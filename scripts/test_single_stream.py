"""
Single Stream Test — Vision Attribute Pipeline
===============================================
Runs the full pipeline inline (no Redis/Milvus required) on a single
video file or RTSP stream. Useful for:
  - Verifying model setup and GPU availability
  - Tuning confidence thresholds
  - Visualising detections and attributes in real time

Displays a live OpenCV window showing:
  - Bounding boxes + track IDs
  - Attribute labels overlay
  - FPS + latency counters

Usage:
    python scripts/test_single_stream.py \
        --source test_data/sample.mp4 \
        --config config/config.yaml \
        [--show-window]   \
        [--output output.mp4]  \
        [--max-frames 500]
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Colour palette for track IDs
TRACK_COLOURS = [
    (255, 80, 80),  (80, 255, 80),  (80, 80, 255),
    (255, 255, 80), (255, 80, 255), (80, 255, 255),
    (200, 120, 60), (60, 200, 120), (120, 60, 200),
]


def get_colour(track_id: int):
    return TRACK_COLOURS[int(track_id) % len(TRACK_COLOURS)]


def draw_detections(frame, tracks, attributes):
    """Draws bounding boxes, track IDs, and attribute chips onto a frame copy."""
    vis = frame.copy()
    for track in tracks:
        tid = track.get("track_id", 0)
        bbox = track.get("bbox", [])
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        colour = get_colour(str(tid).split(":")[-1] if ":" in str(tid) else tid)

        # Box
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # Track ID label
        label = f"ID:{str(tid).split(':')[-1]}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Attribute overlay
        attrs = attributes.get(str(tid), {})
        if attrs:
            attr_lines = [
                f"{attrs.get('gender','?')} {attrs.get('age_group','')}",
                f"upper: {attrs.get('upper_color','?')} {attrs.get('upper_type','?')}",
                f"lower: {attrs.get('lower_color','?')} {attrs.get('lower_type','?')}",
                f"glasses:{attrs.get('glasses','?')} hat:{attrs.get('hat','?')}",
            ]
            y_off = y2 + 14
            for line in attr_lines:
                cv2.putText(vis, line, (x1, y_off),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
                y_off += 14
    return vis


def run_test(
    source: str,
    config_path: str,
    show_window: bool = True,
    output_path: str = None,
    max_frames: int = 0,
):
    from pipeline.utils.logging_utils import load_config, get_logger
    log = get_logger("test_single_stream")

    cfg = load_config(config_path)
    det_cfg = cfg["models"]["yolo"]
    device = det_cfg.get("device", "cuda:0")

    print(f"\n{'='*60}")
    print(f"  Vision Attribute Pipeline — Single Stream Test")
    print(f"{'='*60}")
    print(f"  Source:  {source}")
    print(f"  Device:  {device} ({'CUDA available' if torch.cuda.is_available() else 'CPU mode'})")
    print(f"  Config:  {config_path}")
    print()

    # ── Load models ────────────────────────────────────────────────────────
    print("Loading YOLOv8 + BoT-SORT…", end=" ", flush=True)
    from ultralytics import YOLO
    yolo = YOLO(det_cfg.get("weights", "yolov8x.pt"))
    print("✔")

    print("Loading OSNet ReID…", end=" ", flush=True)
    from pipeline.attribute_worker import ReIDModel
    reid = ReIDModel(cfg)
    reid.load()
    print("✔")

    print("Loading PAR model…", end=" ", flush=True)
    from pipeline.par_model import PARModel
    par = PARModel(cfg)
    backend = par.load()
    print(f"✔  [{backend}]")
    print()

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Input:  {w}x{h} @ {fps_in:.1f}fps  ({total_frames} frames)")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))
        print(f"  Output: {output_path}")

    # Per-track attribute cache (refreshed every N frames)
    track_attributes = {}
    track_crops = {}
    ATTR_REFRESH = cfg.get("detection", {}).get("attribute_refresh_frames", 30)
    FRAME_SKIP = cfg.get("ingestion", {}).get("frame_skip", 1)

    frame_num = 0
    processed = 0
    t_start = time.time()
    latencies = []

    print()
    print("Processing… (Press Q to quit)")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if max_frames and frame_num > max_frames:
            break
        if frame_num % FRAME_SKIP != 0:
            continue
        processed += 1

        t0 = time.time()

        # ── Detection + Tracking ──────────────────────────────────────────
        results = yolo.track(
            source=frame,
            tracker="botsort.yaml",
            conf=float(det_cfg.get("conf_threshold", 0.40)),
            iou=float(det_cfg.get("iou_threshold", 0.45)),
            classes=[0],
            device=device,
            imgsz=int(det_cfg.get("imgsz", 640)),
            half=det_cfg.get("half", True) and torch.cuda.is_available(),
            persist=True,
            verbose=False,
        )
        result = results[0]
        tracks = []

        if result.boxes is not None and result.boxes.id is not None:
            from pipeline.detector_worker import extract_crop
            for box in result.boxes:
                tid = int(box.id.item()) if box.id is not None else None
                if tid is None:
                    continue
                bbox = box.xyxy[0].tolist()
                conf = float(box.conf[0].item())
                crop = extract_crop(frame, bbox)
                if crop.size == 0:
                    continue
                tracks.append({
                    "track_id": tid,
                    "bbox": bbox,
                    "confidence": conf,
                    "crop": crop,
                })
                track_crops[tid] = crop

        # ── ReID + PAR (batched) ──────────────────────────────────────────
        refresh_tracks = [
            t for t in tracks
            if t["track_id"] not in track_attributes
            or processed % ATTR_REFRESH == 0
        ]
        if refresh_tracks:
            crops = [t["crop"] for t in refresh_tracks]
            try:
                embeds = reid.extract_batch(crops)
                attrs_list = par.predict_batch(crops)
                for i, t in enumerate(refresh_tracks):
                    track_attributes[t["track_id"]] = attrs_list[i]
            except Exception as e:
                log.warning(f"ReID/PAR failed: {e}")

        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)

        # ── Visualise ────────────────────────────────────────────────────
        vis = draw_detections(frame, tracks, track_attributes)

        # Stats overlay
        avg_lat = sum(latencies[-30:]) / max(len(latencies[-30:]), 1)
        fps_live = processed / max(time.time() - t_start, 0.001)
        stats = [
            f"Frame: {frame_num}/{total_frames}",
            f"FPS: {fps_live:.1f}",
            f"Latency: {avg_lat:.0f}ms",
            f"Tracks: {len(tracks)}",
            f"PAR: {backend[:30]}",
        ]
        for i, s in enumerate(stats):
            cv2.putText(vis, s, (10, 20 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        if writer:
            writer.write(vis)

        if show_window:
            cv2.imshow("VAP — Single Stream Test", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("User quit.")
                break

        # Terminal progress
        if processed % 50 == 0:
            print(
                f"  Frame {frame_num:5d} | "
                f"FPS: {fps_live:5.1f} | "
                f"Latency: {avg_lat:5.0f}ms | "
                f"Tracks: {len(tracks):3d} | "
                f"Cached attrs: {len(track_attributes):4d}"
            )

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    avg_latency = sum(latencies) / max(len(latencies), 1)
    print()
    print("=" * 60)
    print(f"  DONE. Processed {processed} frames in {elapsed:.1f}s")
    print(f"  Average FPS:     {processed/elapsed:.1f}")
    print(f"  Average latency: {avg_latency:.0f}ms per frame")
    print(f"  Unique tracks:   {len(track_attributes)}")
    if output_path:
        print(f"  Output saved to: {output_path}")
    print("=" * 60)
    print()

    # Print a sample of detected attributes
    if track_attributes:
        print("  Sample attribute detections:")
        for tid, attrs in list(track_attributes.items())[:5]:
            print(f"    Track {tid}: {attrs}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — Single Stream Test")
    parser.add_argument("--source", required=True, help="Video file path or RTSP URL")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--show-window", action="store_true", help="Show live OpenCV window")
    parser.add_argument("--output", default=None, help="Save annotated video to this path")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0=all)")
    args = parser.parse_args()

    run_test(
        source=args.source,
        config_path=args.config,
        show_window=args.show_window,
        output_path=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
