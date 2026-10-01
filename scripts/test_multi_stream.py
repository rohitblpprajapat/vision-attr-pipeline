"""
Multi-Stream Test — Vision Attribute Pipeline
==============================================
Phase 1: Ingests a single "reference" video/stream and populates Milvus.
Phase 2: Runs additional streams simultaneously, registers live queries, and
         displays real-time attribute matches across all streams in the terminal.

This is the integration test for the full distributed pipeline — it requires
Redis and Milvus to be running (start them with: docker compose up redis etcd minio milvus -d).

Workflow:
  1. Register one or more live attribute queries (from --queries arg)
  2. Start N streams in parallel (camera_ingestor style, but inline)
  3. Run the full detect → attribute → store → alert pipeline on each stream
  4. Display a live terminal dashboard of incoming matches

Usage:
    # Phase 1: single reference stream (fills the DB)
    python scripts/test_multi_stream.py \
        --config config/config.yaml \
        --streams "cam01:test_data/sample.mp4" \
        --queries '{"upper_color": "white"}' \
        --duration 60

    # Phase 2: multiple streams + live query matching
    python scripts/test_multi_stream.py \
        --config config/config.yaml \
        --streams "cam01:test_data/sample.mp4" "cam02:test_data/sample2.mp4" \
        --queries '{"glasses": "on"}' '{"upper_color": "white", "hat": "on"}' \
        --duration 120 \
        --show-stats
"""
import argparse
import asyncio
import json
import queue
import signal
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

# ── Terminal UI ──────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    class Console:
        def print(self, *a, **kw): print(*a)
    console = Console()

_stop_event = threading.Event()


def _signal_handler(sig, frame):
    console.print("\n[yellow]Shutting down...[/yellow]")
    _stop_event.set()


# ── Shared state ─────────────────────────────────────────────────────────────

@dataclass
class StreamStats:
    camera_id: str
    frames_read: int = 0
    frames_processed: int = 0
    detections: int = 0
    attributes_run: int = 0
    alerts_fired: int = 0
    errors: int = 0
    fps: float = 0.0
    last_detection_ts: float = 0.0
    recent_fps: deque = field(default_factory=lambda: deque(maxlen=30))


@dataclass
class AlertRecord:
    timestamp: float
    camera_id: str
    track_id: str
    query_desc: str
    attributes: dict
    thumbnail_path: str = ""


class SharedState:
    def __init__(self):
        self.stream_stats: Dict[str, StreamStats] = {}
        self.recent_alerts: deque = deque(maxlen=50)
        self.query_ids: Dict[str, str] = {}   # description → query_id
        self.total_stored: int = 0
        self.lock = threading.Lock()

    def add_alert(self, alert: AlertRecord):
        with self.lock:
            self.recent_alerts.appendleft(alert)


# ── Inline pipeline runner per stream ────────────────────────────────────────

class InlineStreamPipeline(threading.Thread):
    """
    Runs the full detect → ReID → PAR → store → alert pipeline on a single
    stream inline (no Redis), writing directly to Milvus.
    Fires alerts by checking registered queries from Redis pub/sub.
    """

    def __init__(
        self,
        camera_id: str,
        source: str,
        cfg: dict,
        shared: SharedState,
        models: dict,   # pre-loaded models passed in to avoid reloading per thread
        alert_queue: queue.Queue,
    ):
        super().__init__(name=f"stream-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.cfg = cfg
        self.shared = shared
        self.models = models
        self.alert_queue = alert_queue

        ingest_cfg = cfg.get("ingestion", {})
        det_cfg = cfg.get("detection", {})
        self.frame_skip = int(ingest_cfg.get("frame_skip", 5))
        self.max_crops = int(det_cfg.get("max_crops_per_frame", 20))
        self.confirm_frames = int(det_cfg.get("track_confirm_frames", 3))
        self.refresh_frames = int(det_cfg.get("attribute_refresh_frames", 30))
        self.min_bbox_area = int(det_cfg.get("min_bbox_area", 2500))

        self._track_counts: Dict[str, int] = {}
        self._track_last_attr: Dict[str, int] = {}
        self._track_attrs: Dict[str, dict] = {}
        self._processed = 0

        with self.shared.lock:
            self.shared.stream_stats[camera_id] = StreamStats(camera_id=camera_id)

    def _track_needs_attr(self, track_id: str) -> bool:
        count = self._track_counts.get(track_id, 0)
        last = self._track_last_attr.get(track_id, -999)
        needs = (count >= self.confirm_frames) and (count - last >= self.refresh_frames)
        if needs:
            self._track_last_attr[track_id] = count
        return needs

    def _track_is_confirmed(self, track_id: str) -> bool:
        return self._track_counts.get(track_id, 0) >= self.confirm_frames

    def run(self):
        from pipeline.detector_worker import extract_crop, encode_crop
        from pipeline.utils.milvus_utils import insert_detections
        from pipeline.utils.redis_utils import create_redis_client

        stats = self.shared.stream_stats[self.camera_id]
        redis_client = create_redis_client(self.cfg["redis"])
        milvus_col = self.models.get("milvus_col")
        yolo = self.models.get("yolo")
        reid = self.models.get("reid")
        par = self.models.get("par")

        storage_cfg = self.cfg.get("storage", {})
        thumb_dir = storage_cfg.get("thumbnail_dir", "/tmp/thumbnails")
        Path(thumb_dir).mkdir(parents=True, exist_ok=True)
        dedup_window = int(storage_cfg.get("dedup_window_seconds", 5))

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            console.print(f"[red]ERROR: Cannot open {self.source}[/red]")
            return

        det_cfg_model = self.cfg["models"]["yolo"]
        device = det_cfg_model.get("device", "cuda:0")
        import torch
        device = device if torch.cuda.is_available() else "cpu"
        half = det_cfg_model.get("half", True) and torch.cuda.is_available()

        frame_num = 0
        t_start = time.time()
        pending_inserts = []
        last_flush = time.time()

        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Loop video for continuous testing
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            frame_num += 1
            stats.frames_read = frame_num

            if frame_num % self.frame_skip != 0:
                continue

            self._processed += 1
            stats.frames_processed = self._processed
            t_frame = time.time()

            # ── Detect + Track ────────────────────────────────────────────
            try:
                results = yolo.track(
                    source=frame,
                    tracker="botsort.yaml",
                    conf=float(det_cfg_model.get("conf_threshold", 0.40)),
                    iou=float(det_cfg_model.get("iou_threshold", 0.45)),
                    classes=[0],
                    device=device,
                    imgsz=int(det_cfg_model.get("imgsz", 640)),
                    half=half,
                    persist=True,
                    verbose=False,
                )
            except Exception as e:
                stats.errors += 1
                continue

            result = results[0] if results else None
            if result is None or result.boxes is None or result.boxes.id is None:
                continue

            tracks_to_process = []
            for box in result.boxes:
                tid = box.id
                if tid is None:
                    continue
                track_id = f"{self.camera_id}:{int(tid.item())}"
                self._track_counts[track_id] = self._track_counts.get(track_id, 0) + 1

                if not self._track_is_confirmed(track_id):
                    continue

                bbox = box.xyxy[0].tolist()
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < self.min_bbox_area:
                    continue

                conf = float(box.conf[0].item())
                crop = extract_crop(frame, bbox)
                if crop.size == 0:
                    continue

                needs_attr = self._track_needs_attr(track_id)
                tracks_to_process.append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "conf": conf,
                    "crop": crop,
                    "needs_attr": needs_attr,
                })
                stats.detections += 1

            if not tracks_to_process:
                continue

            # Limit crops
            tracks_to_process = tracks_to_process[:self.max_crops]

            # ── ReID + PAR ────────────────────────────────────────────────
            attr_tracks = [t for t in tracks_to_process if t["needs_attr"]]
            all_crops = [t["crop"] for t in tracks_to_process]

            try:
                embeds = reid.extract_batch(all_crops)   # (N, 512)
            except Exception as e:
                stats.errors += 1
                continue

            attr_results = {}
            if attr_tracks:
                attr_crops = [t["crop"] for t in attr_tracks]
                try:
                    preds = par.predict_batch(attr_crops)
                    for i, t in enumerate(attr_tracks):
                        self._track_attrs[t["track_id"]] = preds[i]
                        attr_results[t["track_id"]] = preds[i]
                    stats.attributes_run += len(attr_tracks)
                except Exception as e:
                    stats.errors += 1

            # ── Assemble records ──────────────────────────────────────────
            for i, t in enumerate(tracks_to_process):
                track_id = t["track_id"]
                attrs = self._track_attrs.get(track_id, {})

                # Dedup check via Redis
                dedup_key = f"dedup:{self.camera_id}:{track_id}"
                if not redis_client.set(dedup_key, "1", ex=dedup_window, nx=True):
                    continue  # Already stored recently

                detection_id = str(uuid.uuid4())
                crop_bytes = encode_crop(t["crop"])
                thumb_path = str(Path(thumb_dir) / f"{detection_id}.jpg")
                try:
                    Path(thumb_path).write_bytes(crop_bytes)
                except Exception:
                    thumb_path = ""

                pending_inserts.append({
                    "detection_id":   detection_id,
                    "track_id":       track_id,
                    "camera_id":      self.camera_id,
                    "timestamp":      int(time.time() * 1000),
                    "reid_embed":     embeds[i].tolist(),
                    "attributes":     attrs,
                    "thumbnail_path": thumb_path,
                    "bbox":           t["bbox"],
                    "confidence":     t["conf"],
                })

                # ── Live query matching ───────────────────────────────────
                active_queries_raw = redis_client.hgetall("active_queries")
                for qk, qv in active_queries_raw.items():
                    try:
                        qspec = json.loads(qv)
                        qattrs = qspec.get("attributes", {})
                        cam_filter = qspec.get("camera_ids")
                        if cam_filter and self.camera_id not in cam_filter:
                            continue
                        match = all(attrs.get(k) == v for k, v in qattrs.items())
                        if match:
                            stats.alerts_fired += 1
                            alert = AlertRecord(
                                timestamp=time.time(),
                                camera_id=self.camera_id,
                                track_id=track_id,
                                query_desc=qspec.get("description", str(qattrs)),
                                attributes=attrs,
                                thumbnail_path=thumb_path,
                            )
                            self.shared.add_alert(alert)
                            self.alert_queue.put(alert)
                    except Exception:
                        pass

            # ── Batch insert to Milvus ────────────────────────────────────
            if milvus_col and (len(pending_inserts) >= 50 or time.time() - last_flush > 2.0):
                if pending_inserts:
                    try:
                        insert_detections(milvus_col, pending_inserts)
                        with self.shared.lock:
                            self.shared.total_stored += len(pending_inserts)
                        pending_inserts = []
                        last_flush = time.time()
                    except Exception as e:
                        stats.errors += 1

            # FPS tracking
            elapsed = time.time() - t_start
            if elapsed > 0:
                stats.fps = self._processed / elapsed

        # Final flush
        if milvus_col and pending_inserts:
            try:
                insert_detections(milvus_col, pending_inserts)
            except Exception:
                pass

        cap.release()


# ── Terminal dashboard ────────────────────────────────────────────────────────

def build_dashboard(shared: SharedState, queries: List[dict], elapsed: float) -> Layout:
    if not RICH:
        return None

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="streams", ratio=2),
        Layout(name="alerts", ratio=3),
    )

    # Header
    layout["header"].update(Panel(
        Text(
            f"  Vision Attribute Pipeline — Multi-Stream Monitor  |  "
            f"Runtime: {int(elapsed//60):02d}:{int(elapsed%60):02d}  |  "
            f"Total stored: {shared.total_stored:,}",
            style="bold white",
        ),
        style="on dark_blue",
    ))

    # Streams table
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Camera", style="cyan", width=12)
    t.add_column("FPS", justify="right", width=6)
    t.add_column("Frames", justify="right", width=8)
    t.add_column("Detections", justify="right", width=10)
    t.add_column("PAR runs", justify="right", width=8)
    t.add_column("Alerts", justify="right", style="green", width=7)
    t.add_column("Errors", justify="right", style="red", width=7)

    with shared.lock:
        for cam_id, s in shared.stream_stats.items():
            t.add_row(
                cam_id,
                f"{s.fps:.1f}",
                f"{s.frames_processed:,}",
                f"{s.detections:,}",
                f"{s.attributes_run:,}",
                f"[green]{s.alerts_fired}[/green]" if s.alerts_fired else "0",
                f"[red]{s.errors}[/red]" if s.errors else "0",
            )

    layout["streams"].update(Panel(t, title="[bold]Active Streams[/bold]", border_style="blue"))

    # Alerts panel
    alerts_table = Table(box=box.SIMPLE, show_header=True, header_style="bold green", expand=True)
    alerts_table.add_column("Time", width=10)
    alerts_table.add_column("Camera", width=8)
    alerts_table.add_column("Query", width=20)
    alerts_table.add_column("Attributes", min_width=25)

    with shared.lock:
        for alert in list(shared.recent_alerts)[:15]:
            t_str = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
            attr_str = ", ".join(f"{k}={v}" for k, v in list(alert.attributes.items())[:4])
            alerts_table.add_row(
                t_str,
                alert.camera_id,
                Text(alert.query_desc[:20], style="yellow"),
                Text(attr_str, style="dim"),
            )

    layout["alerts"].update(Panel(
        alerts_table,
        title=f"[bold green]Live Alerts[/bold green] ({sum(s.alerts_fired for s in shared.stream_stats.values())} total)",
        border_style="green",
    ))

    # Footer: registered queries
    q_strs = []
    for q in queries:
        q_strs.append(f"[cyan]{q.get('description', str(q.get('attributes', {})))}[/cyan]")
    layout["footer"].update(Panel(
        "  Active queries: " + "  |  ".join(q_strs) if q_strs else "  No queries registered",
        style="dim",
    ))

    return layout


# ── Model loader ──────────────────────────────────────────────────────────────

def load_all_models(cfg: dict) -> dict:
    """Loads all models once — shared across stream threads via the dict."""
    import torch
    models = {}

    console.print("[cyan]Loading YOLOv8…[/cyan]", end=" ")
    from ultralytics import YOLO
    models["yolo"] = YOLO(cfg["models"]["yolo"].get("weights", "yolov8x.pt"))
    console.print("[green]✔[/green]")

    console.print("[cyan]Loading OSNet ReID…[/cyan]", end=" ")
    from pipeline.attribute_worker import ReIDModel
    reid = ReIDModel(cfg)
    reid.load()
    models["reid"] = reid
    console.print("[green]✔[/green]")

    console.print("[cyan]Loading PAR model…[/cyan]", end=" ")
    from pipeline.par_model import PARModel
    par = PARModel(cfg)
    backend = par.load()
    models["par"] = par
    console.print(f"[green]✔[/green]  [{backend[:40]}]")

    # Milvus
    try:
        from pipeline.utils.milvus_utils import connect_milvus, ensure_collection
        connect_milvus(cfg["milvus"])
        col = ensure_collection(cfg["milvus"])
        models["milvus_col"] = col
        console.print("[green]✔  Milvus connected[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠  Milvus not available: {e}. Detections will not be stored.[/yellow]")
        models["milvus_col"] = None

    return models


# ── Main ──────────────────────────────────────────────────────────────────────

def register_queries(api_url: Optional[str], queries: List[dict], redis_client) -> List[dict]:
    """Registers live queries either via API or directly into Redis."""
    registered = []
    for q in queries:
        attrs = q.get("attributes", q)  # allow bare attr dict
        desc = q.get("description", str(attrs))
        query_id = str(uuid.uuid4())
        spec = {
            "query_id":    query_id,
            "attributes":  attrs,
            "camera_ids":  q.get("camera_ids"),
            "description": desc,
            "created_at":  int(time.time() * 1000),
        }
        redis_client.hset("active_queries", query_id, json.dumps(spec))
        registered.append(spec)
        console.print(f"  [green]✔[/green] Query registered: [yellow]{desc}[/yellow] (id: {query_id[:8]}…)")
    return registered


def main():
    parser = argparse.ArgumentParser(description="Multi-stream real-time attribute matching test")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--streams", nargs="+", required=True,
        help="Streams as 'cam_id:source' pairs. e.g. cam01:video.mp4 cam02:rtsp://..."
    )
    parser.add_argument(
        "--queries", nargs="*", default=[],
        help="JSON attribute queries e.g. '{\"upper_color\": \"white\"}'"
    )
    parser.add_argument("--duration", type=int, default=60, help="Run duration in seconds (0 = until Ctrl+C)")
    parser.add_argument("--show-stats", action="store_true", help="Show live terminal dashboard")
    parser.add_argument("--api", default=None, help="API URL if server is running (optional)")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    from pipeline.utils.logging_utils import load_config
    from pipeline.utils.redis_utils import create_redis_client
    cfg = load_config(args.config)

    # Parse streams
    streams = {}
    for s in args.streams:
        if ":" not in s:
            console.print(f"[red]Invalid stream spec: {s}[/red]")
            sys.exit(1)
        parts = s.split(":", 1)
        streams[parts[0]] = parts[1]

    # Parse queries
    queries_raw = []
    for q_str in args.queries:
        try:
            q = json.loads(q_str)
            if not isinstance(q, dict):
                raise ValueError("Must be a JSON object")
            queries_raw.append({"attributes": q, "description": str(q)})
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid query JSON: {q_str} — {e}[/red]")
            sys.exit(1)

    console.print(f"\n[bold]Vision Attribute Pipeline — Multi-Stream Test[/bold]")
    console.print(f"  Streams: {list(streams.keys())}")
    console.print(f"  Queries: {len(queries_raw)}")
    console.print(f"  Duration: {args.duration}s\n")

    # Connect Redis
    redis_client = create_redis_client(cfg["redis"])

    # Load models (once, shared)
    console.print("[bold]Loading models…[/bold]")
    models = load_all_models(cfg)
    console.print()

    # Register queries
    if queries_raw:
        console.print("[bold]Registering live queries…[/bold]")
        registered_queries = register_queries(args.api, queries_raw, redis_client)
        console.print()
    else:
        registered_queries = []
        console.print("[yellow]No queries specified — detections will be stored but no alerts will fire.[/yellow]\n")

    # Shared state
    shared = SharedState()
    alert_queue: queue.Queue = queue.Queue()

    # Launch stream threads
    console.print(f"[bold]Starting {len(streams)} stream(s)…[/bold]")
    pipeline_threads = []
    for cam_id, source in streams.items():
        t = InlineStreamPipeline(cam_id, source, cfg, shared, models, alert_queue)
        t.start()
        pipeline_threads.append(t)
        console.print(f"  [green]✔[/green] Started stream: [cyan]{cam_id}[/cyan] → {source}")
    console.print()

    t_start = time.time()
    deadline = t_start + args.duration if args.duration > 0 else float("inf")

    # ── Live dashboard ────────────────────────────────────────────────────────
    if args.show_stats and RICH:
        with Live(refresh_per_second=2, screen=False) as live:
            while not _stop_event.is_set() and time.time() < deadline:
                elapsed = time.time() - t_start
                dash = build_dashboard(shared, registered_queries, elapsed)
                if dash:
                    live.update(dash)
                time.sleep(0.5)
    else:
        # Simple polling output
        last_print = time.time()
        while not _stop_event.is_set() and time.time() < deadline:
            # Drain alert queue and print
            while not alert_queue.empty():
                alert = alert_queue.get_nowait()
                t_str = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
                console.print(
                    f"  [green]ALERT[/green] [{t_str}] "
                    f"cam=[cyan]{alert.camera_id}[/cyan] "
                    f"query=[yellow]{alert.query_desc}[/yellow] "
                    f"attrs={alert.attributes}"
                )

            if time.time() - last_print > 10:
                elapsed = time.time() - t_start
                with shared.lock:
                    lines = []
                    for cam_id, s in shared.stream_stats.items():
                        lines.append(
                            f"  {cam_id}: fps={s.fps:.1f} detections={s.detections} "
                            f"par={s.attributes_run} alerts={s.alerts_fired}"
                        )
                console.print(f"\n[dim]── {int(elapsed)}s elapsed ──[/dim]")
                for line in lines:
                    console.print(line)
                console.print(f"  Total stored in Milvus: {shared.total_stored:,}")
                last_print = time.time()

            time.sleep(0.1)

    # Shutdown
    _stop_event.set()
    console.print("\n[yellow]Stopping streams…[/yellow]")
    for t in pipeline_threads:
        t.join(timeout=10)

    # Cleanup queries from Redis
    for q in registered_queries:
        redis_client.hdel("active_queries", q["query_id"])

    # Final summary
    elapsed = time.time() - t_start
    console.print(f"\n[bold]── Final Summary ({int(elapsed)}s) ──[/bold]")
    with shared.lock:
        for cam_id, s in shared.stream_stats.items():
            console.print(
                f"  [cyan]{cam_id}[/cyan]: "
                f"processed={s.frames_processed} detections={s.detections} "
                f"par_runs={s.attributes_run} alerts={s.alerts_fired} errors={s.errors}"
            )
    console.print(f"  Total detections stored in Milvus: [green]{shared.total_stored:,}[/green]")
    total_alerts = sum(s.alerts_fired for s in shared.stream_stats.values())
    console.print(f"  Total live alerts fired: [green]{total_alerts}[/green]")
    console.print()


if __name__ == "__main__":
    main()
