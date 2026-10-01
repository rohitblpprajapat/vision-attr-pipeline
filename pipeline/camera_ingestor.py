"""
Camera Ingestor — Vision Attribute Pipeline
============================================
Reads from RTSP streams or video files, samples frames at the configured
rate, compresses to JPEG, and publishes to the Redis raw_frames stream.

Supports:
  - Multiple cameras concurrently (one thread per camera)
  - RTSP reconnection with exponential backoff
  - Frame timestamp + metadata propagation
  - Graceful shutdown on SIGTERM/SIGINT

Usage:
    python -m pipeline.camera_ingestor --config config/config.yaml \
        --cameras cam01:rtsp://192.168.1.10/stream1 \
                  cam02:rtsp://192.168.1.11/stream1 \
                  test:/path/to/video.mp4
"""
import argparse
import base64
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from pipeline.utils.logging_utils import get_logger, load_config, PerfTimer
from pipeline.utils.redis_utils import create_redis_client, stream_publish

log = get_logger(__name__)

_stop_event = threading.Event()


def _signal_handler(sig, frame):
    log.info("Shutdown signal received — stopping ingestors…")
    _stop_event.set()


# ---------------------------------------------------------------------------
# Single-camera ingestor thread
# ---------------------------------------------------------------------------

class CameraIngestor(threading.Thread):
    """
    One thread per camera. Reads frames, applies skip logic, publishes to Redis.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,          # RTSP URL or file path
        redis_client,
        cfg: dict,
    ):
        super().__init__(name=f"ingestor-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.redis = redis_client
        self.cfg = cfg

        ingest_cfg = cfg.get("ingestion", {})
        self.frame_skip = int(ingest_cfg.get("frame_skip", 5))
        self.max_width = int(ingest_cfg.get("max_frame_width", 1280))
        self.jpeg_quality = int(ingest_cfg.get("jpeg_quality", 85))
        self.reconnect_delay = int(ingest_cfg.get("reconnect_delay", 5))

        streams_cfg = cfg.get("redis", {}).get("streams", {})
        self.stream_name = streams_cfg.get("raw_frames", "stream:raw_frames")
        self.max_len = int(cfg.get("redis", {}).get("max_len", 5000))

        self._frame_count = 0
        self._published_count = 0
        self.log = get_logger(f"ingestor.{camera_id}")

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """Opens VideoCapture with retry logic."""
        attempt = 0
        while not _stop_event.is_set():
            cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                self.log.info(
                    f"Camera opened",
                    extra={"_camera_id": self.camera_id, "_source": self.source},
                )
                # Reduce RTSP buffer to minimise latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                return cap
            delay = min(self.reconnect_delay * (2 ** attempt), 60)
            self.log.warning(
                f"Cannot open camera. Retrying in {delay}s…",
                extra={"_camera_id": self.camera_id, "_attempt": attempt},
            )
            time.sleep(delay)
            attempt += 1
        return None

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Proportionally resizes frame to max_width if needed."""
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        scale = self.max_width / w
        return cv2.resize(frame, (self.max_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """JPEG-encodes frame to bytes."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, buf = cv2.imencode(".jpg", frame, encode_params)
        return buf.tobytes()

    def run(self):
        self.log.info(f"Starting camera ingestor for {self.source}")
        while not _stop_event.is_set():
            cap = self._open_capture()
            if cap is None:
                break

            consecutive_failures = 0
            while not _stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        self.log.warning("Too many read failures — reopening capture")
                        break
                    time.sleep(0.05)
                    continue

                consecutive_failures = 0
                self._frame_count += 1

                # Frame sampling
                if self._frame_count % self.frame_skip != 0:
                    continue

                # Resize + encode
                frame_resized = self._resize_frame(frame)
                h, w = frame_resized.shape[:2]
                frame_bytes = self._encode_frame(frame_resized)

                payload = {
                    "camera_id":  self.camera_id,
                    "frame_id":   str(uuid.uuid4()),
                    "timestamp":  int(time.time() * 1000),   # unix ms
                    "frame_num":  self._frame_count,
                    "width":      w,
                    "height":     h,
                    "frame":      frame_bytes,               # raw JPEG bytes
                }

                with PerfTimer("redis_publish_frame"):
                    stream_publish(
                        self.redis,
                        self.stream_name,
                        payload,
                        maxlen=self.max_len,
                    )
                self._published_count += 1

                if self._published_count % 100 == 0:
                    self.log.info(
                        f"Published {self._published_count} frames",
                        extra={
                            "_camera_id": self.camera_id,
                            "_frame_count": self._frame_count,
                        },
                    )

            cap.release()
            if not _stop_event.is_set():
                self.log.info(f"Reconnecting in {self.reconnect_delay}s…")
                time.sleep(self.reconnect_delay)

        self.log.info(
            f"Ingestor stopped. Published {self._published_count} frames total.",
            extra={"_camera_id": self.camera_id},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_camera_args(camera_strings: List[str]) -> Dict[str, str]:
    """
    Parses camera ID:source strings.
    e.g. "cam01:rtsp://192.168.1.10/stream" → {"cam01": "rtsp://..."}
    """
    cameras = {}
    for cs in camera_strings:
        if ":" not in cs:
            log.warning(f"Invalid camera spec '{cs}' — skipping. Expected 'id:source'")
            continue
        cam_id, source = cs.split(":", 1)
        cameras[cam_id.strip()] = source.strip()
    return cameras


def main():
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — Camera Ingestor")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--cameras",
        nargs="+",
        required=True,
        help="Camera specifications as 'id:source' pairs. "
             "Source can be RTSP URL or video file path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cameras = parse_camera_args(args.cameras)

    if not cameras:
        log.error("No valid cameras provided. Exiting.")
        sys.exit(1)

    # Graceful shutdown handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    redis_client = create_redis_client(cfg["redis"])

    log.info(f"Starting {len(cameras)} camera ingestor(s): {list(cameras.keys())}")

    threads: List[CameraIngestor] = []
    for cam_id, source in cameras.items():
        ingestor = CameraIngestor(cam_id, source, redis_client, cfg)
        ingestor.start()
        threads.append(ingestor)

    try:
        while not _stop_event.is_set():
            # Health-check: log stats every 60s
            time.sleep(60)
            for t in threads:
                if t.is_alive():
                    log.info(
                        f"Camera {t.camera_id}: published={t._published_count}, frames={t._frame_count}",
                        extra={"_camera_id": t.camera_id},
                    )
                else:
                    log.warning(f"Camera {t.camera_id} thread has died!")
    except KeyboardInterrupt:
        _stop_event.set()

    for t in threads:
        t.join(timeout=10)
    log.info("All ingestor threads stopped. Exiting.")


if __name__ == "__main__":
    main()
