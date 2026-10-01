"""
Benchmark — Vision Attribute Pipeline
======================================
Measures the throughput and latency of each pipeline component
on the current hardware. Run this before choosing your EC2 instance size.

Tests:
  1. YOLOv8x detection FPS at various batch sizes
  2. OSNet ReID crops/second at various batch sizes
  3. PromptPAR / CLIP PAR crops/second
  4. Milvus write throughput (vectors/second)
  5. Milvus query latency at various collection sizes
  6. End-to-end simulated pipeline throughput
  7. Maximum concurrent cameras estimate

Outputs a summary table and recommended config settings.

Usage:
    python scripts/benchmark.py \
        --config config/config.yaml \
        [--skip-milvus]  \
        [--frames 200]   \
        [--video test_data/sample.mp4]
"""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
    def info(msg): console.print(f"[cyan]ℹ[/cyan]  {msg}")
    def ok(msg):   console.print(f"[green]✔[/green]  {msg}")
    def warn(msg): console.print(f"[yellow]⚠[/yellow]  {msg}")
    def header(msg): console.rule(f"[bold cyan]{msg}[/bold cyan]")
    def show_table(t): console.print(t)
except ImportError:
    def info(msg): print(f"  {msg}")
    def ok(msg):   print(f"  OK: {msg}")
    def warn(msg): print(f"  WARN: {msg}")
    def header(msg): print(f"\n=== {msg} ===")
    def show_table(t): print(t)
    Console = None
    Table = None


# ── Benchmark helpers ─────────────────────────────────────────────────────────

def generate_test_crops(n: int, h: int = 256, w: int = 128) -> List[np.ndarray]:
    """Generates N synthetic person crop images."""
    return [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def generate_test_frame(h: int = 720, w: int = 1280) -> np.ndarray:
    """Generates a synthetic test frame."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def warmup_gpu():
    """Runs a dummy CUDA op to avoid first-inference cold-start inflation."""
    try:
        import torch
        if torch.cuda.is_available():
            x = torch.randn(100, 100, device="cuda:0")
            _ = x @ x
            torch.cuda.synchronize()
            info("GPU warmed up")
    except Exception:
        pass


def time_fn(fn, *args, n_runs: int = 5, warmup: int = 2, **kwargs) -> Tuple[float, float, float]:
    """
    Times a function over multiple runs.
    Returns (mean_ms, p50_ms, p95_ms).
    """
    times = []
    for i in range(n_runs + warmup):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            times.append(elapsed_ms)

    mean = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]
    return mean, p50, p95


# ── Individual benchmarks ─────────────────────────────────────────────────────

def benchmark_yolo(cfg: dict, video_path: Optional[str], n_frames: int) -> dict:
    header("1. YOLOv8x Detection")
    from ultralytics import YOLO
    import torch

    det_cfg = cfg["models"]["yolo"]
    device = det_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    half = det_cfg.get("half", True) and torch.cuda.is_available()
    imgsz = int(det_cfg.get("imgsz", 640))

    info(f"Device: {device}  |  FP16: {half}  |  imgsz: {imgsz}")
    model = YOLO(det_cfg.get("weights", "yolov8x.pt"))
    warmup_gpu()

    # Load test frames
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        frames = []
        for _ in range(min(n_frames, 100)):
            ret, f = cap.read()
            if not ret: break
            frames.append(f)
        cap.release()
        info(f"Loaded {len(frames)} frames from {video_path}")
    else:
        frames = [generate_test_frame() for _ in range(50)]
        info("Using synthetic test frames (no video provided)")

    if not frames:
        warn("No frames available for YOLO benchmark")
        return {}

    # Benchmark: single frame (no tracking)
    def run_single():
        model(frames[0], device=device, imgsz=imgsz, half=half, verbose=False, classes=[0])

    mean_ms, p50_ms, p95_ms = time_fn(run_single, n_runs=20, warmup=3)
    fps_single = 1000 / mean_ms

    # Benchmark: with tracking (BoT-SORT)
    def run_track():
        model.track(
            source=frames[0],
            tracker="botsort.yaml",
            device=device,
            imgsz=imgsz,
            half=half,
            persist=True,
            verbose=False,
            classes=[0],
        )

    mean_track_ms, _, _ = time_fn(run_track, n_runs=15, warmup=3)
    fps_track = 1000 / mean_track_ms

    results = {
        "detect_mean_ms": round(mean_ms, 1),
        "detect_fps": round(fps_single, 1),
        "track_mean_ms": round(mean_track_ms, 1),
        "track_fps": round(fps_track, 1),
    }

    ok(f"Detection only:    {mean_ms:.1f}ms/frame  →  {fps_single:.0f} FPS")
    ok(f"Detection+Track:   {mean_track_ms:.1f}ms/frame  →  {fps_track:.0f} FPS")
    return results


def benchmark_reid(cfg: dict) -> dict:
    header("2. OSNet-AIN ReID Embeddings")
    from pipeline.attribute_worker import ReIDModel
    reid = ReIDModel(cfg)
    reid.load()
    if reid._extractor is None:
        warn("ReID model not loaded — skipping")
        return {}

    warmup_gpu()
    results = {}

    for batch_size in [1, 8, 16, 32, 64]:
        crops = generate_test_crops(batch_size)
        mean_ms, p50_ms, p95_ms = time_fn(
            reid.extract_batch, crops, n_runs=10, warmup=2
        )
        crops_per_sec = int(batch_size / (mean_ms / 1000))
        results[f"batch_{batch_size}"] = {
            "mean_ms": round(mean_ms, 1),
            "crops_per_sec": crops_per_sec,
        }
        ok(f"batch={batch_size:3d}:  {mean_ms:6.1f}ms  →  {crops_per_sec:5,} crops/s")

    return results


def benchmark_par(cfg: dict) -> dict:
    header("3. PAR Model (PromptPAR / CLIP)")
    from pipeline.par_model import PARModel
    par = PARModel(cfg)
    backend = par.load()
    info(f"Backend: {backend}")
    warmup_gpu()

    results = {"backend": backend}

    for batch_size in [1, 4, 8, 16]:
        crops = generate_test_crops(batch_size)
        mean_ms, p50_ms, p95_ms = time_fn(
            par.predict_batch, crops, n_runs=8, warmup=2
        )
        crops_per_sec = int(batch_size / (mean_ms / 1000))
        results[f"batch_{batch_size}"] = {
            "mean_ms": round(mean_ms, 1),
            "crops_per_sec": crops_per_sec,
        }
        ok(f"batch={batch_size:3d}:  {mean_ms:6.1f}ms  →  {crops_per_sec:5,} crops/s")

    return results


def benchmark_milvus(cfg: dict, n_vectors: int = 10000) -> dict:
    header("4. Milvus Write + Query Throughput")

    try:
        from pipeline.utils.milvus_utils import connect_milvus, ensure_collection, insert_detections
        from pymilvus import utility
    except Exception as e:
        warn(f"Milvus not available: {e}")
        return {}

    try:
        connect_milvus(cfg["milvus"])
    except Exception as e:
        warn(f"Cannot connect to Milvus: {e}")
        return {}

    # Use a benchmark-specific collection to avoid polluting production data
    import copy
    bench_cfg = copy.deepcopy(cfg["milvus"])
    bench_cfg["collection"] = "vap_benchmark_temp"
    bench_cfg["ttl_seconds"] = 300  # 5min TTL for cleanup

    col = ensure_collection(bench_cfg)
    results = {}

    # Write benchmark
    info(f"Writing {n_vectors} test vectors…")
    batch_size = 100
    records = []
    for i in range(n_vectors):
        embed = np.random.randn(512).astype(np.float32)
        embed /= np.linalg.norm(embed)
        records.append({
            "detection_id":   f"bench_{i:08d}",
            "track_id":       f"bench_track_{i % 100}",
            "camera_id":      f"bench_cam_{i % 5}",
            "timestamp":      int(time.time() * 1000) - (i * 1000),
            "reid_embed":     embed.tolist(),
            "attributes":     {"glasses": "on" if i % 3 == 0 else "off", "upper_color": "white"},
            "thumbnail_path": "",
            "bbox":           [100, 100, 200, 400],
            "confidence":     0.9,
        })

    t0 = time.perf_counter()
    batches = [records[i:i+batch_size] for i in range(0, len(records), batch_size)]
    for batch in batches:
        insert_detections(col, batch)
    col.flush()
    write_elapsed = time.perf_counter() - t0
    write_vps = n_vectors / write_elapsed
    results["write_vps"] = round(write_vps, 0)
    ok(f"Write throughput: {write_vps:,.0f} vectors/second ({n_vectors} vectors in {write_elapsed:.2f}s)")

    # Query benchmark
    info("Benchmarking query latency…")
    query_embed = np.random.randn(512).astype(np.float32)
    query_embed /= np.linalg.norm(query_embed)

    from pipeline.utils.milvus_utils import search_by_embedding, search_by_attributes_only

    def run_vector_search():
        search_by_embedding(col, query_embed, top_k=50, cfg=bench_cfg)

    def run_attr_search():
        search_by_attributes_only(col, {"glasses": "on"}, limit=100)

    mean_vec_ms, p50, p95 = time_fn(run_vector_search, n_runs=20, warmup=5)
    mean_attr_ms, _, _ = time_fn(run_attr_search, n_runs=20, warmup=5)

    results["vector_search_mean_ms"] = round(mean_vec_ms, 2)
    results["attr_search_mean_ms"] = round(mean_attr_ms, 2)
    ok(f"Vector search (top-50):     {mean_vec_ms:.2f}ms")
    ok(f"Attribute query (limit-100): {mean_attr_ms:.2f}ms")

    # Cleanup
    try:
        utility.drop_collection("vap_benchmark_temp")
        info("Benchmark collection cleaned up")
    except Exception:
        pass

    return results


def benchmark_end_to_end(cfg: dict, video_path: Optional[str]) -> dict:
    header("5. End-to-End Simulated Pipeline")
    import torch

    if not video_path or not Path(video_path).exists():
        warn("No video path provided — skipping end-to-end benchmark")
        return {}

    from ultralytics import YOLO
    from pipeline.attribute_worker import ReIDModel
    from pipeline.par_model import PARModel
    from pipeline.detector_worker import extract_crop

    det_cfg = cfg["models"]["yolo"]
    device = det_cfg.get("device", "cuda:0") if torch.cuda.is_available() else "cpu"
    half = det_cfg.get("half", True) and torch.cuda.is_available()

    yolo = YOLO(det_cfg.get("weights", "yolov8x.pt"))
    reid = ReIDModel(cfg)
    reid.load()
    par = PARModel(cfg)
    par.load()

    cap = cv2.VideoCapture(video_path)
    warmup_gpu()

    frame_times = []
    processed = 0
    t_start = time.time()
    FRAME_SKIP = int(cfg.get("ingestion", {}).get("frame_skip", 5))

    info(f"Running {min(200 // FRAME_SKIP, 50)} frames with frame_skip={FRAME_SKIP}…")

    for i in range(500):
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret: break

        if i % FRAME_SKIP != 0:
            continue

        t0 = time.perf_counter()

        # Detect + track
        results = yolo.track(
            source=frame, tracker="botsort.yaml", classes=[0],
            device=device, imgsz=int(det_cfg.get("imgsz", 640)),
            half=half, persist=True, verbose=False,
        )
        result = results[0]
        crops = []
        if result.boxes is not None and result.boxes.id is not None:
            for box in result.boxes[:10]:  # cap at 10
                bbox = box.xyxy[0].tolist()
                crop = extract_crop(frame, bbox)
                if crop.size > 0:
                    crops.append(crop)

        # ReID + PAR if we have crops
        if crops:
            reid.extract_batch(crops)
            par.predict_batch(crops[:4])  # PAR is the bottleneck, cap at 4

        elapsed_ms = (time.perf_counter() - t0) * 1000
        frame_times.append(elapsed_ms)
        processed += 1

        if processed >= 50:
            break

    cap.release()

    if not frame_times:
        return {}

    mean_ms = statistics.mean(frame_times)
    p95_ms = sorted(frame_times)[int(len(frame_times) * 0.95)]
    fps = 1000 / mean_ms

    info(f"Processed {processed} frames")
    ok(f"Mean frame latency:  {mean_ms:.1f}ms")
    ok(f"P95 frame latency:   {p95_ms:.1f}ms")
    ok(f"Effective FPS:       {fps:.1f}")

    return {
        "mean_latency_ms": round(mean_ms, 1),
        "p95_latency_ms": round(p95_ms, 1),
        "fps": round(fps, 1),
    }


# ── Recommendations ───────────────────────────────────────────────────────────

def print_recommendations(results: dict, cfg: dict):
    header("Recommendations")

    yolo = results.get("yolo", {})
    reid = results.get("reid", {})
    par = results.get("par", {})
    e2e = results.get("e2e", {})

    track_fps = yolo.get("track_fps", 0)
    par_b16 = par.get("batch_16", {}).get("crops_per_sec", 0)
    reid_b32 = (reid.get("batch_32") or {}).get("crops_per_sec", 0)

    if track_fps > 0 and par_b16 > 0:
        frame_skip = int(cfg.get("ingestion", {}).get("frame_skip", 5))
        attr_refresh = int(cfg.get("detection", {}).get("attribute_refresh_frames", 30))

        # At 25fps with frame_skip, effective processing rate per camera:
        effective_fps_per_cam = 25.0 / frame_skip
        # PAR load per camera (crops/sec, assuming 2 people per frame average):
        avg_people_per_frame = 2
        par_demand_per_cam = (effective_fps_per_cam * avg_people_per_frame) / attr_refresh

        if par_b16 > 0:
            max_cameras = int(par_b16 / par_demand_per_cam)
            info(f"At current settings (frame_skip={frame_skip}, attr_refresh={attr_refresh}):")
            info(f"  PAR demand per camera: ~{par_demand_per_cam:.1f} crops/s")
            info(f"  Estimated max cameras on this GPU: ~{max_cameras}")

            if max_cameras >= 50:
                ok(f"This GPU can handle 50+ cameras ✔")
            elif max_cameras >= 25:
                warn(f"This GPU handles ~{max_cameras} cameras. For 50+, use g5.2xlarge (A10G) or scale attribute workers.")
            else:
                warn(f"This GPU handles ~{max_cameras} cameras. Increase frame_skip or attr_refresh to support more.")

    # Show suggested config tweaks
    info("\nSuggested config.yaml settings for this hardware:")
    if track_fps > 60:
        info("  ingestion.frame_skip: 3   ← GPU is fast enough for higher frame rate")
    elif track_fps < 25:
        info("  ingestion.frame_skip: 8   ← GPU is under pressure, reduce frame rate")
    else:
        info(f"  ingestion.frame_skip: 5   ← Current setting is appropriate")

    if e2e:
        info(f"  Measured end-to-end latency: {e2e.get('mean_latency_ms', '?')}ms "
             f"(P95: {e2e.get('p95_latency_ms', '?')}ms)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — Benchmark")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--video", default=None, help="Test video path")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--milvus-vectors", type=int, default=10000)
    parser.add_argument("--skip-milvus", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--output-json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    from pipeline.utils.logging_utils import load_config
    cfg = load_config(args.config)

    console.print(f"\n[bold]Vision Attribute Pipeline — Benchmark Suite[/bold]")
    import torch
    console.print(f"  CUDA: {'✔ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✖ CPU only'}")
    console.print()

    all_results = {}

    all_results["yolo"] = benchmark_yolo(cfg, args.video, args.frames)
    gc.collect()

    all_results["reid"] = benchmark_reid(cfg)
    gc.collect()

    all_results["par"] = benchmark_par(cfg)
    gc.collect()

    if not args.skip_milvus:
        all_results["milvus"] = benchmark_milvus(cfg, args.milvus_vectors)

    if not args.skip_e2e and args.video:
        all_results["e2e"] = benchmark_end_to_end(cfg, args.video)

    print_recommendations(all_results, cfg)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        ok(f"Results saved to {args.output_json}")

    console.print()


if __name__ == "__main__":
    main()
