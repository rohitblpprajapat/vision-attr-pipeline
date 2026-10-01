"""
Pipeline Monitor — Vision Attribute Pipeline
=============================================
Live terminal dashboard showing real-time health of all pipeline components.

Polls:
  - Redis stream lengths and consumer group lag
  - Active live queries and alert counts
  - API health endpoint
  - GPU memory and utilization (if pynvml available)
  - Recent detections per camera

Usage:
    python scripts/pipeline_monitor.py \
        --config config/config.yaml \
        --api http://localhost:8000 \
        --interval 2
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import requests

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import BarColumn, Progress, TextColumn
    from rich import box
    from rich.columns import Columns
    RICH = True
except ImportError:
    print("This monitor requires 'rich'. Run: pip install rich")
    sys.exit(1)

console = Console()


# ── Data collectors ───────────────────────────────────────────────────────────

def fetch_health(api_url: str) -> dict:
    try:
        resp = requests.get(f"{api_url}/api/v1/health", timeout=3)
        return resp.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


def fetch_live_queries(api_url: str) -> dict:
    try:
        resp = requests.get(f"{api_url}/api/v1/live-queries", timeout=3)
        return resp.json()
    except Exception:
        return {"count": 0, "queries": []}


def fetch_redis_stats(redis_client, streams_cfg: dict) -> dict:
    from pipeline.utils.redis_utils import stream_stats
    stats = {}
    for name, key in streams_cfg.items():
        try:
            stats[name] = stream_stats(redis_client, key)
        except Exception as e:
            stats[name] = {"error": str(e)}
    return stats


def fetch_gpu_stats() -> list:
    try:
        import torch
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_alloc = torch.cuda.memory_allocated(i) / (1024**2)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**2)
                mem_total_mb = torch.cuda.get_device_properties(i).total_memory / (1024**2)
                name = torch.cuda.get_device_name(i)
                util_pct = None
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_pct = util.gpu
                except Exception:
                    pass
                gpus.append({
                    "index": i,
                    "name": name,
                    "mem_alloc_mb": mem_alloc,
                    "mem_reserved_mb": mem_reserved,
                    "mem_total_mb": mem_total_mb,
                    "util_pct": util_pct,
                })
        return gpus
    except Exception:
        return []


def fetch_alert_counts(redis_client) -> dict:
    try:
        # Count total alerts in pub/sub via a ZSET we can maintain, or just count keys
        raw = redis_client.hgetall("active_queries")
        return {"active_queries": len(raw)}
    except Exception:
        return {"active_queries": 0}


# ── Dashboard builders ────────────────────────────────────────────────────────

def build_header(health: dict, now: datetime) -> Panel:
    status = health.get("status", "unknown")
    color = "green" if status == "ok" else "red" if status == "unreachable" else "yellow"
    return Panel(
        Text(
            f"  Vision Attribute Pipeline Monitor  |  "
            f"{now.strftime('%Y-%m-%d  %H:%M:%S UTC')}  |  "
            f"Status: [{color}]{status.upper()}[/{color}]",
            style="bold",
        ),
        style=f"on {'dark_blue' if status == 'ok' else 'dark_red'}",
    )


def build_components_table(health: dict) -> Panel:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Component", style="cyan", width=20)
    t.add_column("Status", width=10)
    t.add_column("Detail")

    for comp, info in health.get("components", {}).items():
        status = info.get("status", "?")
        color = "green" if status == "ok" else "red"
        detail = ""
        if "stats" in info:
            s = info["stats"]
            detail = f"entities={s.get('num_entities', '?'):,}"
        elif "error" in info:
            detail = str(info["error"])[:60]
        t.add_row(comp, f"[{color}]{status}[/{color}]", detail)

    return Panel(t, title="[bold]Components[/bold]", border_style="blue")


def build_streams_table(redis_stats: dict) -> Panel:
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", expand=True)
    t.add_column("Stream", style="cyan", width=20)
    t.add_column("Length", justify="right", width=8)
    t.add_column("Groups/Lag", width=35)

    for stream_name, stats in redis_stats.items():
        if "error" in stats:
            t.add_row(stream_name, "[red]ERR[/red]", str(stats["error"])[:40])
            continue
        length = stats.get("length", 0)
        length_color = "green" if length < 1000 else "yellow" if length < 4000 else "red"
        groups = stats.get("groups", [])
        group_str = "  ".join(
            f"{g.get('name','?').split(':')[-1]}:lag=[{'green' if g.get('lag',0)==0 else 'yellow'}]{g.get('lag',0)}[/{'green' if g.get('lag',0)==0 else 'yellow'}]"
            for g in groups
        ) or "—"
        t.add_row(
            stream_name,
            f"[{length_color}]{length:,}[/{length_color}]",
            group_str,
        )

    return Panel(t, title="[bold]Redis Streams[/bold]", border_style="blue")


def build_gpu_panel(gpus: list) -> Panel:
    if not gpus:
        return Panel("[dim]No GPU detected or torch not installed[/dim]", title="[bold]GPU[/bold]", border_style="yellow")

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold", expand=True)
    t.add_column("GPU", style="cyan")
    t.add_column("VRAM Used", justify="right", width=12)
    t.add_column("VRAM Total", justify="right", width=12)
    t.add_column("Utilization", width=25)

    for gpu in gpus:
        mem = gpu["mem_alloc_mb"]
        total = gpu["mem_total_mb"]
        pct = (mem / total * 100) if total > 0 else 0
        mem_color = "green" if pct < 70 else "yellow" if pct < 90 else "red"

        util_str = "N/A"
        if gpu["util_pct"] is not None:
            u = gpu["util_pct"]
            util_color = "green" if u < 70 else "yellow" if u < 90 else "red"
            bar = "█" * int(u / 5) + "░" * (20 - int(u / 5))
            util_str = f"[{util_color}]{bar}[/{util_color}] {u}%"

        t.add_row(
            f"cuda:{gpu['index']} {gpu['name'][:20]}",
            f"[{mem_color}]{mem:,.0f} MB[/{mem_color}]",
            f"{total:,.0f} MB",
            util_str,
        )

    return Panel(t, title="[bold]GPU[/bold]", border_style="yellow")


def build_queries_panel(live_queries: dict) -> Panel:
    queries = live_queries.get("queries", [])
    count = live_queries.get("count", 0)

    if not queries:
        return Panel(
            "[dim]No live queries registered.\n"
            "Register via: POST /api/v1/live-queries[/dim]",
            title=f"[bold]Live Queries[/bold] (0)",
            border_style="green",
        )

    t = Table(box=box.SIMPLE, show_header=True, header_style="bold green", expand=True)
    t.add_column("ID", style="dim", width=10)
    t.add_column("Description", width=25)
    t.add_column("Attributes")
    t.add_column("Cameras", width=15)

    for q in queries[:10]:
        attrs = q.get("attributes", {})
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items())
        cams = q.get("camera_ids")
        cam_str = ", ".join(cams) if cams else "all"
        t.add_row(
            q.get("query_id", "")[:8] + "…",
            q.get("description", "—")[:25],
            Text(attr_str, style="yellow"),
            cam_str,
        )

    return Panel(t, title=f"[bold]Live Queries[/bold] ({count})", border_style="green")


def build_footer(poll_count: int, interval: float) -> Panel:
    return Panel(
        f"  Poll #{poll_count}  |  Refresh every {interval}s  |  "
        "[bold]Ctrl+C[/bold] to exit  |  "
        "Docs: [cyan]http://localhost:8000/docs[/cyan]",
        style="dim",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — Live Monitor")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--interval", type=float, default=2.0, help="Refresh interval seconds")
    args = parser.parse_args()

    from pipeline.utils.logging_utils import load_config
    from pipeline.utils.redis_utils import create_redis_client
    cfg = load_config(args.config)
    redis_client = create_redis_client(cfg["redis"])
    streams_cfg = cfg["redis"]["streams"]

    poll_count = 0

    def make_layout() -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="top_row", size=12),
            Layout(name="mid_row", size=12),
            Layout(name="footer", size=3),
        )
        layout["top_row"].split_row(
            Layout(name="components", ratio=1),
            Layout(name="streams", ratio=2),
        )
        layout["mid_row"].split_row(
            Layout(name="gpu", ratio=1),
            Layout(name="queries", ratio=2),
        )
        return layout

    with Live(make_layout(), refresh_per_second=1, screen=True) as live:
        while True:
            try:
                poll_count += 1
                now = datetime.now(timezone.utc)

                health = fetch_health(args.api)
                live_queries = fetch_live_queries(args.api)
                redis_stats = fetch_redis_stats(redis_client, streams_cfg)
                gpus = fetch_gpu_stats()

                layout = make_layout()
                layout["header"].update(build_header(health, now))
                layout["components"].update(build_components_table(health))
                layout["streams"].update(build_streams_table(redis_stats))
                layout["gpu"].update(build_gpu_panel(gpus))
                layout["queries"].update(build_queries_panel(live_queries))
                layout["footer"].update(build_footer(poll_count, args.interval))
                live.update(layout)

                time.sleep(args.interval)

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Monitor error: {e}[/red]")
                time.sleep(args.interval)

    console.print("\n[yellow]Monitor stopped.[/yellow]")


if __name__ == "__main__":
    main()
