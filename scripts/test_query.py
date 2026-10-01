"""
Query Test Script — Vision Attribute Pipeline
=============================================
Tests all search engine query types against a running pipeline.

Requires:
  - Docker Compose stack running (redis + milvus + workers)
  - At least some data ingested into Milvus (run a video through the pipeline first)

Tests:
  1. Attribute-only search (e.g. upper_color=white)
  2. Embedding similarity search (upload a person crop)
  3. Live query registration + WebSocket alert reception
  4. API health check

Usage:
    python scripts/test_query.py --api http://localhost:8000
    python scripts/test_query.py --api http://localhost:8000 --image path/to/person_crop.jpg
"""
import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import requests
import websockets

try:
    from rich.console import Console
    from rich.table import Table
    from rich.json import JSON
    console = Console()
    def header(msg): console.rule(f"[bold cyan]{msg}[/bold cyan]")
    def ok(msg):     console.print(f"  [green]✔[/green] {msg}")
    def fail(msg):   console.print(f"  [red]✖[/red] {msg}")
    def info(msg):   console.print(f"  [blue]ℹ[/blue] {msg}")
    def show_json(d): console.print(JSON(json.dumps(d, default=str, indent=2)))
except ImportError:
    def header(msg): print(f"\n=== {msg} ===")
    def ok(msg):     print(f"  [OK]   {msg}")
    def fail(msg):   print(f"  [FAIL] {msg}")
    def info(msg):   print(f"  [INFO] {msg}")
    def show_json(d): print(json.dumps(d, default=str, indent=2))


# ── Test helpers ──────────────────────────────────────────────────────────

def assert_200(resp, label: str):
    if resp.status_code == 200 or resp.status_code == 201:
        ok(f"{label} — HTTP {resp.status_code}")
        return True
    else:
        fail(f"{label} — HTTP {resp.status_code}: {resp.text[:200]}")
        return False


def test_health(api: str):
    header("1. Health Check")
    resp = requests.get(f"{api}/api/v1/health", timeout=10)
    if assert_200(resp, "GET /api/v1/health"):
        data = resp.json()
        status = data.get("status", "unknown")
        colour = "green" if status == "ok" else "yellow"
        info(f"Pipeline status: [{colour}]{status}[/{colour}]")
        for component, cinfo in data.get("components", {}).items():
            s = cinfo.get("status", "?")
            c = "green" if s == "ok" else "red"
            info(f"  {component}: [{c}]{s}[/{c}]")
        return True
    return False


def test_schema(api: str):
    header("2. Attribute Schema")
    resp = requests.get(f"{api}/api/v1/attributes/schema", timeout=10)
    if assert_200(resp, "GET /api/v1/attributes/schema"):
        schema = resp.json()
        info(f"Available attributes: {list(schema.keys())}")
        return schema
    return {}


def test_attribute_search(api: str, attrs: dict, hours: float = 24):
    header(f"3. Attribute Search: {attrs}")
    payload = {
        "attributes": attrs,
        "time_window_hours": hours,
        "limit": 20,
    }
    t0 = time.time()
    resp = requests.post(f"{api}/api/v1/search/attributes", json=payload, timeout=30)
    latency_ms = (time.time() - t0) * 1000
    info(f"Query latency: {latency_ms:.0f}ms")

    if assert_200(resp, "POST /api/v1/search/attributes"):
        data = resp.json()
        count = data.get("count", 0)
        info(f"Results returned: {count}")
        results = data.get("results", [])
        if results:
            ok(f"First result:")
            r = results[0]
            info(f"  camera_id:  {r.get('camera_id')}")
            info(f"  track_id:   {r.get('track_id')}")
            info(f"  timestamp:  {r.get('datetime_utc')}")
            info(f"  attributes: {r.get('attributes')}")
            info(f"  thumbnail:  {r.get('thumbnail_url')}")
        else:
            info("No results — the pipeline may not have ingested data yet")
        return data
    return {}


def test_image_search(api: str, image_path: str):
    header(f"4. Image Similarity Search: {image_path}")
    img_bytes = Path(image_path).read_bytes()
    b64 = base64.b64encode(img_bytes).decode()
    payload = {
        "image_base64": b64,
        "time_window_hours": 24,
        "top_k": 20,
        "similarity_threshold": 0.50,
    }
    t0 = time.time()
    resp = requests.post(f"{api}/api/v1/search/image", json=payload, timeout=30)
    latency_ms = (time.time() - t0) * 1000
    info(f"Query latency: {latency_ms:.0f}ms")

    if assert_200(resp, "POST /api/v1/search/image"):
        data = resp.json()
        count = data.get("count", 0)
        info(f"Similar persons found: {count}")
        results = data.get("results", [])
        if results:
            ok(f"Top match (score={results[0].get('score', 0):.3f}):")
            r = results[0]
            info(f"  camera_id:  {r.get('camera_id')}")
            info(f"  track_id:   {r.get('track_id')}")
            info(f"  attributes: {r.get('attributes')}")
        return data
    return {}


def test_hybrid_search(api: str, attrs: dict, image_path: str = None):
    header(f"5. Hybrid Search: attrs={attrs}")
    payload = {
        "attributes": attrs,
        "time_window_hours": 24,
        "top_k": 20,
        "similarity_threshold": 0.50,
    }
    if image_path:
        img_bytes = Path(image_path).read_bytes()
        payload["image_base64"] = base64.b64encode(img_bytes).decode()

    t0 = time.time()
    resp = requests.post(f"{api}/api/v1/search/hybrid", json=payload, timeout=30)
    latency_ms = (time.time() - t0) * 1000
    info(f"Query latency: {latency_ms:.0f}ms")

    if assert_200(resp, "POST /api/v1/search/hybrid"):
        data = resp.json()
        info(f"Hybrid results: {data.get('count', 0)}")
        return data
    return {}


def test_live_query(api: str, attrs: dict):
    header(f"6. Live Query Registration: {attrs}")

    # Register query
    payload = {
        "attributes": attrs,
        "description": f"Test query: {attrs}",
    }
    resp = requests.post(f"{api}/api/v1/live-queries", json=payload, timeout=10)
    if not assert_200(resp, "POST /api/v1/live-queries"):
        return None
    data = resp.json()
    query_id = data.get("query_id")
    ws_url = data.get("ws_url")
    ok(f"Query registered: {query_id}")
    info(f"WebSocket URL: {ws_url}")

    # List queries
    resp2 = requests.get(f"{api}/api/v1/live-queries", timeout=5)
    if assert_200(resp2, "GET /api/v1/live-queries"):
        all_queries = resp2.json()
        info(f"Total active queries: {all_queries.get('count', 0)}")

    return query_id


async def test_websocket_alerts(api: str, query_id: str, wait_seconds: int = 10):
    header(f"7. WebSocket Alerts (waiting {wait_seconds}s for matches)")
    ws_host = api.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_host}/ws/alerts/{query_id}"
    info(f"Connecting to: {ws_url}")

    received = []
    try:
        async with websockets.connect(ws_url, ping_interval=5) as ws:
            ok("WebSocket connected")
            deadline = time.time() + wait_seconds
            while time.time() < deadline:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    if data.get("type") == "subscribed":
                        info("Subscription confirmed")
                    else:
                        received.append(data)
                        ok(f"Alert received! camera={data.get('camera_id')} track={data.get('track_id')}")
                        info(f"  attributes: {data.get('attributes')}")
                except asyncio.TimeoutError:
                    pass
    except Exception as e:
        fail(f"WebSocket error: {e}")

    if received:
        ok(f"Total alerts received: {len(received)}")
    else:
        info(f"No alerts in {wait_seconds}s — pipeline may need data to process")
    return received


def cleanup_live_query(api: str, query_id: str):
    resp = requests.delete(f"{api}/api/v1/live-queries/{query_id}", timeout=5)
    if assert_200(resp, f"DELETE /api/v1/live-queries/{query_id}"):
        ok(f"Query {query_id} deleted")


def test_error_handling(api: str):
    header("8. Error Handling Tests")

    # Invalid attribute key
    resp = requests.post(
        f"{api}/api/v1/search/attributes",
        json={"attributes": {"invalid_key": "value"}},
        timeout=5,
    )
    if resp.status_code == 400:
        ok(f"Invalid attribute key → 400 (correct)")
    else:
        fail(f"Expected 400, got {resp.status_code}")

    # Invalid attribute value
    resp2 = requests.post(
        f"{api}/api/v1/search/attributes",
        json={"attributes": {"glasses": "maybe"}},
        timeout=5,
    )
    if resp2.status_code == 400:
        ok(f"Invalid attribute value → 400 (correct)")
    else:
        fail(f"Expected 400, got {resp2.status_code}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — Query Test")
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--image", default=None, help="Path to a person crop image for similarity search")
    parser.add_argument("--wait-alerts", type=int, default=10, help="Seconds to wait for live alerts")
    parser.add_argument("--skip-ws", action="store_true", help="Skip WebSocket test")
    args = parser.parse_args()

    print()
    console.print(
        f"[bold]Vision Attribute Pipeline — Query Test Suite[/bold]\n"
        f"API: [cyan]{args.api}[/cyan]",
        markup=True,
    )
    print()

    # 1. Health
    if not test_health(args.api):
        fail("API is not reachable. Is the pipeline running?")
        sys.exit(1)

    # 2. Schema
    schema = test_schema(args.api)

    # 3. Attribute searches (diverse test cases)
    test_cases = [
        {"glasses": "on"},
        {"upper_color": "white"},
        {"gender": "female", "hat": "on"},
        {"bag_type": "backpack"},
        {"upper_color": "black", "lower_type": "jeans"},
    ]
    for attrs in test_cases:
        test_attribute_search(args.api, attrs, hours=24)

    # 4. Image search (if image provided)
    if args.image:
        test_image_search(args.api, args.image)

        # 5. Hybrid search
        test_hybrid_search(args.api, {"glasses": "on"}, args.image)

    # 6. Live query + WebSocket
    query_id = test_live_query(args.api, {"upper_color": "white"})
    if query_id and not args.skip_ws:
        asyncio.run(test_websocket_alerts(args.api, query_id, args.wait_alerts))
        cleanup_live_query(args.api, query_id)

    # 7. Error handling
    test_error_handling(args.api)

    print()
    console.rule("[bold green]All tests complete[/bold green]")
    print()


if __name__ == "__main__":
    main()
