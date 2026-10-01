"""
Attribute Worker — Vision Attribute Pipeline
============================================
Consumes person crops from Redis, runs:
  1. OSNet-AIN x1.0 ReID → 512-dim appearance embedding (L2-normalised)
  2. PARModel (PromptPAR or CLIP) → attribute dict
  3. Publishes enriched detection to Redis enriched stream

Batching:
  - Accumulates up to `batch_size` crops before running model inference
  - Falls back to immediate processing if batch timeout exceeded
  - Handles `needs_attribute=False` crops (ReID only, no PAR) for efficiency

Usage:
    python -m pipeline.attribute_worker \
        --config config/config.yaml \
        --worker-id attr-01
"""
import argparse
import io
import os
import signal
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from pipeline.par_model import PARModel
from pipeline.utils.logging_utils import get_logger, load_config, PerfTimer
from pipeline.utils.redis_utils import (
    create_redis_client,
    ensure_consumer_group,
    stream_consume,
    stream_publish,
)

log = get_logger(__name__)
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    log.info("Shutdown signal — stopping attribute worker…")
    _shutdown = True


# ---------------------------------------------------------------------------
# ReID Model wrapper
# ---------------------------------------------------------------------------

class ReIDModel:
    """
    OSNet-AIN x1.0 wrapper using torchreid.
    Produces 512-dim L2-normalised embeddings.

    Pretrained weights: osnet_ain_x1_0 trained on Market1501 + DukeMTMC-reID
    Available on HuggingFace: kaiyangzhou/osnet
    """

    def __init__(self, cfg: dict):
        reid_cfg = cfg.get("models", {}).get("reid", {})
        self.model_name = reid_cfg.get("model_name", "osnet_ain_x1_0")
        self.weights_path = reid_cfg.get("weights_path", "models/osnet_ain_x1_0_market1501.pth")
        self.device = reid_cfg.get("device", "cuda:0")
        self.input_size = reid_cfg.get("input_size", [256, 128])   # H x W
        self.batch_size = int(reid_cfg.get("batch_size", 32))
        self._extractor = None

    def load(self):
        """Loads the OSNet model via torchreid FeatureExtractor."""
        log.info(f"Loading ReID model: {self.model_name}")
        try:
            from torchreid.utils import FeatureExtractor
            device_str = self.device if torch.cuda.is_available() else "cpu"
            model_path = self.weights_path if Path(self.weights_path).exists() else None
            self._extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=model_path,
                device=device_str,
                verbose=False,
            )
            log.info(f"ReID model loaded: {self.model_name} on {device_str}")
        except Exception as e:
            log.error(f"Failed to load ReID model: {e}", exc_info=True)
            self._extractor = None

    @torch.no_grad()
    def extract_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extracts ReID embeddings for a batch of BGR crops.
        Returns (N, 512) float32 array, L2-normalised.
        """
        if self._extractor is None:
            return np.zeros((len(crops), 512), dtype=np.float32)

        pil_images = []
        for crop in crops:
            img = cv2.resize(
                crop,
                (self.input_size[1], self.input_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            pil_images.append(img)

        try:
            with PerfTimer("reid_extract", log):
                features = self._extractor(pil_images)   # (N, 512) tensor
            feats = features.cpu().numpy().astype(np.float32)
            # L2 normalise
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-6)
            return feats / norms
        except Exception as e:
            log.error(f"ReID extraction failed: {e}", exc_info=True)
            return np.zeros((len(crops), 512), dtype=np.float32)


# ---------------------------------------------------------------------------
# Thumbnail saving
# ---------------------------------------------------------------------------

def save_thumbnail(
    crop_bytes: bytes,
    thumbnail_dir: str,
    detection_id: str,
    max_age_hours: int = 24,
) -> str:
    """
    Saves a crop JPEG to the thumbnail directory.
    Returns the saved file path.
    """
    thumb_dir = Path(thumbnail_dir)
    thumb_dir.mkdir(parents=True, exist_ok=True)
    path = thumb_dir / f"{detection_id}.jpg"
    try:
        path.write_bytes(crop_bytes)
        return str(path)
    except Exception as e:
        log.warning(f"Could not save thumbnail {path}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Attribute Worker
# ---------------------------------------------------------------------------

class AttributeWorker:
    def __init__(self, cfg: dict, worker_id: str):
        self.cfg = cfg
        self.worker_id = worker_id
        self.redis = create_redis_client(cfg["redis"])

        streams_cfg = cfg["redis"]["streams"]
        groups_cfg = cfg["redis"]["groups"]
        self.in_stream = streams_cfg["person_crops"]
        self.out_stream = streams_cfg["enriched"]
        self.group_name = groups_cfg["attribute"]
        self.max_len = int(cfg["redis"].get("max_len", 5000))

        par_cfg = cfg.get("models", {}).get("par", {})
        self.par_batch_size = int(par_cfg.get("batch_size", 16))

        reid_cfg = cfg.get("models", {}).get("reid", {})
        self.reid_batch_size = int(reid_cfg.get("batch_size", 32))

        storage_cfg = cfg.get("storage", {})
        self.save_thumbs = bool(storage_cfg.get("save_thumbnails", True))
        self.thumb_dir = storage_cfg.get("thumbnail_dir", "/tmp/thumbnails")

        # Redis cache: deduplicate ReID for same track within embed_cache_ttl
        self.embed_cache_ttl = int(cfg.get("redis", {}).get("embed_cache_ttl", 30))

        # Batch accumulator
        self._pending: List[dict] = []    # {msg_id, payload}
        self._batch_timeout = 0.05        # 50ms max wait for a batch

        # Models (loaded lazily)
        self._reid = ReIDModel(cfg)
        self._par = PARModel(cfg)
        self._models_loaded = False

    def _load_models(self):
        if self._models_loaded:
            return
        self._reid.load()
        backend = self._par.load()
        log.info(
            f"Models loaded. ReID={self._reid.model_name}, PAR={backend}",
            extra={"_worker_id": self.worker_id},
        )
        self._models_loaded = True

    def _decode_crop(self, crop_bytes: bytes) -> Optional[np.ndarray]:
        """Decodes JPEG bytes to BGR numpy array."""
        try:
            buf = np.frombuffer(crop_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            return img
        except Exception:
            return None

    def _get_cached_embed(self, track_id: str) -> Optional[np.ndarray]:
        """Returns cached embedding for track if not expired."""
        key = f"embed_cache:{track_id}"
        raw = self.redis.get(key)
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).copy()

    def _set_cached_embed(self, track_id: str, embed: np.ndarray):
        key = f"embed_cache:{track_id}"
        self.redis.setex(key, self.embed_cache_ttl, embed.tobytes())

    def process_batch(self, messages: List[dict]) -> None:
        """
        Processes a batch of person crop messages.
        Runs ReID + PAR and publishes enriched detections.
        """
        if not messages:
            return

        # Decode crops
        crops = []
        valid_indices = []
        for i, msg in enumerate(messages):
            crop_bytes = msg["payload"].get("crop")
            if crop_bytes is None:
                continue
            crop = self._decode_crop(crop_bytes)
            if crop is not None and crop.size > 0:
                crops.append(crop)
                valid_indices.append(i)

        if not crops:
            return

        # ── ReID embeddings ──────────────────────────────────────────────────
        embeds = np.zeros((len(crops), 512), dtype=np.float32)
        with PerfTimer("reid_batch", log):
            embeds = self._reid.extract_batch(crops)

        # ── PAR attributes ───────────────────────────────────────────────────
        # Only run PAR for crops that need attribute refresh
        par_indices = []
        par_crops = []
        for local_i, global_i in enumerate(valid_indices):
            if messages[global_i]["payload"].get("needs_attribute", True):
                par_indices.append(local_i)
                par_crops.append(crops[local_i])

        par_results: Dict[int, dict] = {}
        if par_crops:
            with PerfTimer("par_batch", log):
                par_attrs = self._par.predict_batch(par_crops)
            for j, local_i in enumerate(par_indices):
                par_results[local_i] = par_attrs[j]

        # ── Assemble and publish enriched detections ─────────────────────────
        for local_i, global_i in enumerate(valid_indices):
            msg = messages[global_i]
            payload = msg["payload"]
            detection_id = str(uuid.uuid4())

            embed = embeds[local_i]
            track_id = payload.get("track_id", "")

            # Use cached embed if fresh and PAR doesn't need refresh
            if not payload.get("needs_attribute", True):
                cached = self._get_cached_embed(track_id)
                if cached is not None:
                    embed = cached
                else:
                    self._set_cached_embed(track_id, embed)
            else:
                self._set_cached_embed(track_id, embed)

            attributes = par_results.get(local_i, {})

            # Save thumbnail
            thumb_path = ""
            if self.save_thumbs:
                crop_bytes = payload.get("crop", b"")
                thumb_path = save_thumbnail(crop_bytes, self.thumb_dir, detection_id)

            enriched = {
                "detection_id":  detection_id,
                "camera_id":     payload.get("camera_id", ""),
                "frame_id":      payload.get("frame_id", ""),
                "timestamp":     payload.get("timestamp", int(time.time() * 1000)),
                "track_id":      track_id,
                "bbox":          payload.get("bbox", []),
                "confidence":    payload.get("confidence", 0.0),
                "reid_embed":    embed.tolist(),
                "attributes":    attributes,
                "thumbnail_path": thumb_path,
                "frame_w":       payload.get("frame_w", 0),
                "frame_h":       payload.get("frame_h", 0),
            }

            stream_publish(
                self.redis,
                self.out_stream,
                enriched,
                maxlen=self.max_len,
            )

        log.debug(
            f"Processed batch of {len(crops)}, PAR run on {len(par_crops)}",
            extra={"_worker_id": self.worker_id},
        )

    def run(self):
        global _shutdown
        self._load_models()
        ensure_consumer_group(self.redis, self.in_stream, self.group_name, start_id="$")

        log.info(
            f"Attribute worker '{self.worker_id}' started",
            extra={"_worker_id": self.worker_id},
        )

        consumer_name = f"attribute-{self.worker_id}"
        pending_msgs: List[dict] = []
        batch_start = time.time()
        total_processed = 0
        error_count = 0

        for msg_id, payload in stream_consume(
            self.redis, self.in_stream, self.group_name, consumer_name,
            batch_size=self.par_batch_size,
        ):
            if _shutdown:
                break

            pending_msgs.append({"msg_id": msg_id, "payload": payload})

            # Flush batch when full OR timeout exceeded
            elapsed = time.time() - batch_start
            if len(pending_msgs) >= self.par_batch_size or elapsed > self._batch_timeout:
                try:
                    self.process_batch(pending_msgs)
                    total_processed += len(pending_msgs)
                    # ACK all messages in the batch
                    for m in pending_msgs:
                        self.redis.xack(self.in_stream, self.group_name, m["msg_id"])
                    if total_processed % 1000 == 0:
                        log.info(
                            f"Attribute worker processed {total_processed} detections, "
                            f"{error_count} errors",
                            extra={"_worker_id": self.worker_id},
                        )
                except Exception as e:
                    error_count += 1
                    log.error(
                        f"Batch processing error: {e}",
                        extra={"_worker_id": self.worker_id},
                        exc_info=True,
                    )
                    for m in pending_msgs:
                        self.redis.xack(self.in_stream, self.group_name, m["msg_id"])

                pending_msgs = []
                batch_start = time.time()

        # Flush remaining
        if pending_msgs and not _shutdown:
            try:
                self.process_batch(pending_msgs)
                for m in pending_msgs:
                    self.redis.xack(self.in_stream, self.group_name, m["msg_id"])
            except Exception as e:
                log.error(f"Final batch flush error: {e}")

        log.info(
            f"Attribute worker '{self.worker_id}' stopped. "
            f"Processed={total_processed}, Errors={error_count}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — Attribute Worker")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--worker-id", default="01")
    parser.add_argument("--metrics-port", type=int, default=9101,
                        help="Prometheus metrics HTTP port (0 to disable)")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    cfg = load_config(args.config)

    # Start Prometheus metrics server
    if args.metrics_port > 0:
        try:
            from pipeline.metrics import PipelineMetrics
            metrics = PipelineMetrics(worker_type="attribute", worker_id=args.worker_id)
            redis_for_metrics = create_redis_client(cfg["redis"])
            metrics.start(
                port=args.metrics_port,
                redis_client=redis_for_metrics,
                streams_cfg=cfg["redis"]["streams"],
            )
            log.info(f"Prometheus metrics on port {args.metrics_port}")
        except Exception as e:
            log.warning(f"Metrics server failed to start: {e}")

    worker = AttributeWorker(cfg, args.worker_id)
    worker.run()


if __name__ == "__main__":
    main()
