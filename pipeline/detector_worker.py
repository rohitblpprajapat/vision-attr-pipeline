"""
Detector Worker — Vision Attribute Pipeline
============================================
Consumes raw frames from Redis, runs YOLOv8x person detection with
BoT-SORT tracking, extracts person crop images, and publishes results
to the person_crops Redis stream.

One worker process handles multiple cameras via a shared GPU.
For >25 cameras, run multiple detector workers (they share the same
consumer group so frames are load-balanced automatically).

Performance targets (T4 GPU, 1280px input):
  - YOLOv8x FP16: ~15ms per frame → ~66 fps single stream
  - With batching: up to 4 frames/batch → ~30ms per batch

Usage:
    python -m pipeline.detector_worker \
        --config config/config.yaml \
        --worker-id detector-01
"""
import argparse
import io
import signal
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    log.info("Shutdown signal — stopping detector worker…")
    _shutdown = True


# ---------------------------------------------------------------------------
# Frame decoder
# ---------------------------------------------------------------------------

def decode_frame(frame_bytes: bytes, width: int, height: int) -> Optional[np.ndarray]:
    """Decodes JPEG bytes back to a numpy BGR array."""
    try:
        buf = np.frombuffer(frame_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        log.error(f"Frame decode failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Tracker state per camera
# ---------------------------------------------------------------------------

class PerCameraTracker:
    """
    Holds per-camera state: frame confirmation counters and track refresh state.
    """

    def __init__(self, confirm_frames: int, refresh_frames: int):
        self.confirm_frames = confirm_frames
        self.refresh_frames = refresh_frames
        # track_id → number of frames seen
        self._frame_counts: Dict[str, int] = {}
        # track_id → frame count at last attribute run
        self._last_attr: Dict[str, int] = {}

    def update(self, track_id: str) -> Tuple[bool, bool]:
        """
        Returns (is_confirmed, needs_attribute_refresh).
        Call once per frame per track.
        """
        count = self._frame_counts.get(track_id, 0) + 1
        self._frame_counts[track_id] = count

        is_confirmed = count >= self.confirm_frames

        last_attr = self._last_attr.get(track_id, -999)
        needs_attr = is_confirmed and (count - last_attr) >= self.refresh_frames

        if needs_attr:
            self._last_attr[track_id] = count

        return is_confirmed, needs_attr

    def evict_old_tracks(self, active_ids: set, camera_id: str):
        """Removes state for tracks no longer active (frees memory)."""
        dead = set(self._frame_counts.keys()) - active_ids
        for tid in dead:
            self._frame_counts.pop(tid, None)
            self._last_attr.pop(tid, None)
        if dead:
            log.debug(
                f"Evicted {len(dead)} stale tracks",
                extra={"_camera_id": camera_id},
            )


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

def extract_crop(
    frame: np.ndarray,
    bbox: List[float],
    pad_ratio: float = 0.15,
) -> np.ndarray:
    """
    Extracts a padded person crop from the frame.
    bbox format: [x1, y1, x2, y2] in pixels.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Padding
    bh, bw = y2 - y1, x2 - x1
    pad_h = int(bh * pad_ratio)
    pad_w = int(bw * pad_ratio)

    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(w, x2 + pad_w)
    y2p = min(h, y2 + pad_h)

    crop = frame[y1p:y2p, x1p:x2p]
    return crop


def encode_crop(crop: np.ndarray, quality: int = 85) -> bytes:
    """Encodes a crop to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Detector Worker
# ---------------------------------------------------------------------------

class DetectorWorker:
    def __init__(self, cfg: dict, worker_id: str):
        self.cfg = cfg
        self.worker_id = worker_id
        self.redis = create_redis_client(cfg["redis"])

        streams_cfg = cfg["redis"]["streams"]
        groups_cfg = cfg["redis"]["groups"]
        self.in_stream = streams_cfg["raw_frames"]
        self.out_stream = streams_cfg["person_crops"]
        self.group_name = groups_cfg["detector"]
        self.max_len = int(cfg["redis"].get("max_len", 5000))

        det_cfg = cfg["models"]["yolo"]
        track_cfg = cfg["models"]["tracker"]
        detect_cfg = cfg.get("detection", {})

        self.conf_threshold = float(det_cfg.get("conf_threshold", 0.40))
        self.iou_threshold = float(det_cfg.get("iou_threshold", 0.45))
        self.min_bbox_area = int(detect_cfg.get("min_bbox_area", 2500))
        self.max_crops = int(detect_cfg.get("max_crops_per_frame", 20))
        self.confirm_frames = int(detect_cfg.get("track_confirm_frames", 3))
        self.refresh_frames = int(detect_cfg.get("attribute_refresh_frames", 30))

        self._camera_trackers: Dict[str, PerCameraTracker] = {}

        # Import and initialise YOLO (deferred to allow worker to start first)
        self._model = None
        self._model_cfg = det_cfg
        self._tracker_type = track_cfg.get("type", "botsort")

    def _get_model(self):
        """Lazy-loads the YOLOv8 model (so import happens inside the worker process)."""
        if self._model is None:
            log.info("Loading YOLOv8 model…")
            from ultralytics import YOLO
            self._model = YOLO(self._model_cfg.get("weights", "yolov8x.pt"))
            log.info("YOLOv8 model loaded")
        return self._model

    def _get_camera_tracker(self, camera_id: str) -> PerCameraTracker:
        if camera_id not in self._camera_trackers:
            self._camera_trackers[camera_id] = PerCameraTracker(
                self.confirm_frames, self.refresh_frames
            )
        return self._camera_trackers[camera_id]

    def _run_detection(
        self, frame: np.ndarray, camera_id: str
    ) -> List[dict]:
        """
        Runs YOLOv8 + BoT-SORT on a frame.
        Returns list of track dicts.
        """
        model = self._get_model()
        det_cfg = self._model_cfg
        device = det_cfg.get("device", "cuda:0")
        imgsz = int(det_cfg.get("imgsz", 640))
        half = det_cfg.get("half", True)

        try:
            with PerfTimer("yolo_track", log):
                results = model.track(
                    source=frame,
                    tracker=f"{self._tracker_type}.yaml",
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    classes=[0],   # person only
                    device=device,
                    imgsz=imgsz,
                    half=half,
                    persist=True,  # BoT-SORT maintains state between calls
                    verbose=False,
                )
        except Exception as e:
            log.error(f"YOLO inference failed: {e}", exc_info=True)
            return []

        tracks = []
        result = results[0] if results else None
        if result is None or result.boxes is None:
            return []

        boxes = result.boxes
        if boxes.id is None:
            return []  # No tracks assigned yet

        active_ids = set()
        cam_tracker = self._get_camera_tracker(camera_id)

        for i, box in enumerate(boxes):
            try:
                track_id = int(box.id.item()) if box.id is not None else None
                if track_id is None:
                    continue

                # Stable track ID string: camera_id + local tracker ID
                stable_id = f"{camera_id}:{track_id}"
                active_ids.add(stable_id)

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                area = (x2 - x1) * (y2 - y1)

                if area < self.min_bbox_area:
                    continue

                is_confirmed, needs_attr = cam_tracker.update(stable_id)
                if not is_confirmed:
                    continue

                crop = extract_crop(frame, [x1, y1, x2, y2])
                if crop.size == 0:
                    continue

                crop_bytes = encode_crop(crop)

                tracks.append({
                    "track_id":      stable_id,
                    "local_track_id": track_id,
                    "bbox":          [x1, y1, x2, y2],
                    "confidence":    conf,
                    "crop":          crop_bytes,
                    "needs_attribute": needs_attr,
                    "frame_w":       frame.shape[1],
                    "frame_h":       frame.shape[0],
                })

            except Exception as e:
                log.warning(f"Error processing track {i}: {e}")
                continue

        cam_tracker.evict_old_tracks(active_ids, camera_id)

        # Limit crops per frame
        if len(tracks) > self.max_crops:
            log.debug(f"Capping {len(tracks)} tracks to {self.max_crops}")
            tracks = tracks[:self.max_crops]

        return tracks

    def process_message(self, msg_id: str, payload: dict) -> None:
        """Processes a single raw frame message."""
        camera_id = payload.get("camera_id", "unknown")
        frame_id = payload.get("frame_id", str(uuid.uuid4()))
        timestamp = payload.get("timestamp", int(time.time() * 1000))
        frame_bytes = payload.get("frame")
        w = payload.get("width", 1280)
        h = payload.get("height", 720)

        if frame_bytes is None:
            log.warning(f"Message {msg_id} has no frame data")
            return

        frame = decode_frame(frame_bytes, w, h)
        if frame is None:
            return

        with PerfTimer(f"full_detect_{camera_id}", log):
            tracks = self._run_detection(frame, camera_id)

        log.debug(
            f"Detected {len(tracks)} tracks",
            extra={
                "_camera_id": camera_id,
                "_frame_id": frame_id,
                "_num_tracks": len(tracks),
            },
        )

        for track in tracks:
            out_payload = {
                "camera_id":       camera_id,
                "frame_id":        frame_id,
                "timestamp":       timestamp,
                "track_id":        track["track_id"],
                "bbox":            track["bbox"],
                "confidence":      track["confidence"],
                "crop":            track["crop"],
                "needs_attribute": track["needs_attribute"],
                "frame_w":         track["frame_w"],
                "frame_h":         track["frame_h"],
            }
            stream_publish(
                self.redis,
                self.out_stream,
                out_payload,
                maxlen=self.max_len,
            )

    def run(self):
        global _shutdown
        ensure_consumer_group(self.redis, self.in_stream, self.group_name, start_id="$")

        log.info(
            f"Detector worker '{self.worker_id}' started, listening on '{self.in_stream}'",
            extra={"_worker_id": self.worker_id},
        )

        consumer_name = f"detector-{self.worker_id}"
        frame_count = 0
        error_count = 0

        for msg_id, payload in stream_consume(
            self.redis, self.in_stream, self.group_name, consumer_name
        ):
            if _shutdown:
                break
            try:
                self.process_message(msg_id, payload)
                self.redis.xack(self.in_stream, self.group_name, msg_id)
                frame_count += 1
                if frame_count % 500 == 0:
                    log.info(
                        f"Processed {frame_count} frames, {error_count} errors",
                        extra={"_worker_id": self.worker_id},
                    )
            except Exception as e:
                error_count += 1
                log.error(
                    f"Error processing message {msg_id}: {e}",
                    extra={"_worker_id": self.worker_id},
                    exc_info=True,
                )
                # ACK even on error to avoid infinite retry loop
                self.redis.xack(self.in_stream, self.group_name, msg_id)

        log.info(
            f"Detector worker '{self.worker_id}' stopped. "
            f"Frames={frame_count}, Errors={error_count}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — Detector Worker")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--worker-id", default="01")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    cfg = load_config(args.config)
    worker = DetectorWorker(cfg, args.worker_id)
    worker.run()


if __name__ == "__main__":
    main()
