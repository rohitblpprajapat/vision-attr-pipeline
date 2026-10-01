"""
API Server — Vision Attribute Pipeline
=======================================
FastAPI application providing:
  - REST endpoints for attribute and embedding search
  - WebSocket endpoints for real-time alert subscriptions
  - Pipeline health monitoring endpoints
  - Simple web UI at GET /

REST Endpoints:
  POST /api/v1/search/attributes     — Attribute-only query
  POST /api/v1/search/image          — Appearance similarity query
  POST /api/v1/search/hybrid         — Combined query
  GET  /api/v1/attributes/schema     — Valid attribute keys/values
  POST /api/v1/live-queries          — Register live query
  DELETE /api/v1/live-queries/{id}   — Remove live query
  GET  /api/v1/live-queries          — List live queries
  GET  /api/v1/health                — Pipeline health

WebSocket:
  WS /ws/alerts/{query_id}           — Stream alerts for a live query
  WS /ws/alerts/global               — Stream all alerts

Usage:
    python -m api.api_server --config config/config.yaml
    # or via uvicorn:
    uvicorn api.api_server:app --host 0.0.0.0 --port 8000 --workers 4
"""
import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    File,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline.utils.logging_utils import get_logger, load_config
from pipeline.utils.redis_utils import create_redis_client, subscribe_alerts
from search.search_engine import SearchEngine, QueryValidationError

log = get_logger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vision Attribute Pipeline API",
    description="Real-time person attribute search across multi-camera streams",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (initialised at startup)
_cfg: dict = {}
_search_engine: Optional[SearchEngine] = None
_redis = None
_active_ws_connections: Dict[str, Set[WebSocket]] = {}


# ── Startup / Shutdown ───────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global _cfg, _search_engine, _redis
    config_path = os.environ.get("VAP_CONFIG", "config/config.yaml")
    _cfg = load_config(config_path)
    _redis = create_redis_client(_cfg["redis"])
    _search_engine = SearchEngine(_cfg)
    try:
        _search_engine.connect()
        log.info("Search engine connected at startup")
    except Exception as e:
        log.error(f"Search engine failed to connect at startup: {e}")


@app.on_event("shutdown")
async def shutdown():
    log.info("API server shutting down")


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class AttributeQuery(BaseModel):
    attributes: Dict[str, str] = Field(
        ...,
        example={"upper_color": "white", "glasses": "on"},
        description="Attribute key→value pairs to match",
    )
    camera_ids: Optional[List[str]] = Field(None, example=["cam01", "cam02"])
    time_window_hours: float = Field(24.0, ge=0.1, le=168.0)
    limit: int = Field(100, ge=1, le=500)


class ImageQuery(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded person crop image (JPEG/PNG)")
    attribute_filter: Optional[Dict[str, str]] = None
    camera_ids: Optional[List[str]] = None
    time_window_hours: float = Field(24.0, ge=0.1, le=168.0)
    top_k: int = Field(50, ge=1, le=200)
    similarity_threshold: float = Field(0.60, ge=0.0, le=1.0)


class HybridQuery(BaseModel):
    image_base64: Optional[str] = None
    attributes: Optional[Dict[str, str]] = None
    camera_ids: Optional[List[str]] = None
    time_window_hours: float = 24.0
    top_k: int = 50
    similarity_threshold: float = 0.55


class LiveQueryRequest(BaseModel):
    attributes: Dict[str, str] = Field(
        ...,
        example={"upper_color": "red", "hat": "on"},
    )
    camera_ids: Optional[List[str]] = None
    description: str = ""


class DetectionResult(BaseModel):
    detection_id: str
    track_id: str
    camera_id: str
    timestamp: int
    score: float
    attributes: dict
    thumbnail_path: Optional[str]
    bbox: List[float]
    confidence: Optional[float]
    thumbnail_url: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _enrich_results(results: List[dict]) -> List[dict]:
    """Adds thumbnail_url to results for UI display."""
    for r in results:
        thumb = r.get("thumbnail_path")
        if thumb and Path(thumb).exists():
            detection_id = r.get("detection_id", "")
            r["thumbnail_url"] = f"/api/v1/thumbnails/{Path(thumb).name}"
        else:
            r["thumbnail_url"] = None
    return results


def _format_ts(ts_ms: int) -> str:
    import datetime
    return datetime.datetime.utcfromtimestamp(ts_ms / 1000).isoformat() + "Z"


def _enrich_with_datetime(results: List[dict]) -> List[dict]:
    for r in results:
        ts = r.get("timestamp")
        if ts:
            r["datetime_utc"] = _format_ts(ts)
    return results


# ── REST Endpoints ───────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serves the embedded web UI."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(content=ui_path.read_text(), status_code=200)
    return HTMLResponse(content="<h1>VAP API</h1><p>UI not found. Use /docs</p>", status_code=200)


@app.get("/api/v1/health")
async def health():
    """Returns health status of all pipeline components."""
    status = {"status": "ok", "timestamp": int(time.time() * 1000), "components": {}}

    # Redis
    try:
        _redis.ping()
        status["components"]["redis"] = {"status": "ok"}
    except Exception as e:
        status["components"]["redis"] = {"status": "error", "detail": str(e)}
        status["status"] = "degraded"

    # Milvus / Search engine
    try:
        stats = _search_engine.stats()
        status["components"]["milvus"] = {"status": "ok", "stats": stats}
    except Exception as e:
        status["components"]["milvus"] = {"status": "error", "detail": str(e)}
        status["status"] = "degraded"

    # Redis streams
    from pipeline.utils.redis_utils import stream_stats
    streams_cfg = _cfg.get("redis", {}).get("streams", {})
    for sname, skey in streams_cfg.items():
        try:
            s = stream_stats(_redis, skey)
            status["components"][f"stream_{sname}"] = s
        except Exception:
            pass

    return JSONResponse(content=status)


@app.get("/api/v1/attributes/schema")
async def get_attribute_schema():
    """Returns all valid attribute keys and their possible values."""
    return _search_engine.get_valid_attributes()


@app.post("/api/v1/search/attributes")
async def search_attributes(query: AttributeQuery):
    """
    Search for detections matching the given attributes.

    Example:
        POST /api/v1/search/attributes
        {
            "attributes": {"upper_color": "white", "glasses": "on"},
            "time_window_hours": 6
        }
    """
    if _search_engine is None:
        raise HTTPException(503, "Search engine not ready")
    try:
        results = _search_engine.query_by_attributes(
            attributes=query.attributes,
            camera_ids=query.camera_ids,
            time_window_hours=query.time_window_hours,
            limit=query.limit,
        )
        results = _enrich_results(results)
        results = _enrich_with_datetime(results)
        return {"count": len(results), "results": results, "query": query.dict()}
    except QueryValidationError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        log.error(f"Attribute search error: {e}", exc_info=True)
        raise HTTPException(500, detail="Search failed")


@app.post("/api/v1/search/image")
async def search_by_image(query: ImageQuery):
    """
    Find people who look visually similar to the provided person crop.
    Image should be a cropped person image, base64-encoded.
    """
    if _search_engine is None:
        raise HTTPException(503, "Search engine not ready")
    try:
        image_bytes = base64.b64decode(query.image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data")
    try:
        results = _search_engine.query_by_image(
            image_bytes=image_bytes,
            attribute_filter=query.attribute_filter,
            camera_ids=query.camera_ids,
            time_window_hours=query.time_window_hours,
            top_k=query.top_k,
            similarity_threshold=query.similarity_threshold,
        )
        results = _enrich_results(results)
        results = _enrich_with_datetime(results)
        return {"count": len(results), "results": results}
    except QueryValidationError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        log.error(f"Image search error: {e}", exc_info=True)
        raise HTTPException(500, detail="Search failed")


@app.post("/api/v1/search/image-upload")
async def search_by_image_upload(
    file: UploadFile = File(...),
    attribute_filter: Optional[str] = Query(None, description="JSON-encoded attribute filter"),
    time_window_hours: float = Query(24.0),
    top_k: int = Query(50),
    similarity_threshold: float = Query(0.60),
):
    """Upload a person crop image file and find similar people."""
    if _search_engine is None:
        raise HTTPException(503, "Search engine not ready")
    image_bytes = await file.read()
    attrs = None
    if attribute_filter:
        try:
            attrs = json.loads(attribute_filter)
        except Exception:
            raise HTTPException(400, "Invalid attribute_filter JSON")
    try:
        results = _search_engine.query_by_image(
            image_bytes=image_bytes,
            attribute_filter=attrs,
            time_window_hours=time_window_hours,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        results = _enrich_results(results)
        results = _enrich_with_datetime(results)
        return {"count": len(results), "results": results}
    except Exception as e:
        log.error(f"Image upload search error: {e}", exc_info=True)
        raise HTTPException(500, detail="Search failed")


@app.post("/api/v1/search/hybrid")
async def search_hybrid(query: HybridQuery):
    """Combined attribute + appearance similarity search."""
    if _search_engine is None:
        raise HTTPException(503, "Search engine not ready")
    image_bytes = None
    if query.image_base64:
        try:
            image_bytes = base64.b64decode(query.image_base64)
        except Exception:
            raise HTTPException(400, "Invalid base64 image data")
    try:
        results = _search_engine.query_hybrid(
            image_bytes=image_bytes,
            attributes=query.attributes,
            camera_ids=query.camera_ids,
            time_window_hours=query.time_window_hours,
            top_k=query.top_k,
            similarity_threshold=query.similarity_threshold,
        )
        results = _enrich_results(results)
        results = _enrich_with_datetime(results)
        return {"count": len(results), "results": results}
    except QueryValidationError as e:
        raise HTTPException(400, detail=str(e))
    except Exception as e:
        log.error(f"Hybrid search error: {e}", exc_info=True)
        raise HTTPException(500, detail="Search failed")


# ── Live Queries ─────────────────────────────────────────────────────────────

@app.post("/api/v1/live-queries", status_code=201)
async def create_live_query(request: LiveQueryRequest):
    """
    Register a live attribute query.
    Matching detections will be pushed to WS /ws/alerts/{query_id} in real-time.
    """
    if _search_engine is None:
        raise HTTPException(503, "Search engine not ready")
    try:
        query_id = _search_engine.register_live_query(
            _redis,
            attributes=request.attributes,
            camera_ids=request.camera_ids,
            description=request.description,
        )
        return {
            "query_id": query_id,
            "ws_url": f"/ws/alerts/{query_id}",
            "attributes": request.attributes,
            "camera_ids": request.camera_ids,
        }
    except QueryValidationError as e:
        raise HTTPException(400, detail=str(e))


@app.delete("/api/v1/live-queries/{query_id}")
async def delete_live_query(query_id: str):
    """Removes a live query."""
    removed = _search_engine.unregister_live_query(_redis, query_id)
    if not removed:
        raise HTTPException(404, "Query not found")
    return {"deleted": query_id}


@app.get("/api/v1/live-queries")
async def list_live_queries():
    """Lists all active live queries."""
    queries = _search_engine.list_live_queries(_redis)
    return {"count": len(queries), "queries": queries}


# ── Thumbnail serving ─────────────────────────────────────────────────────────

@app.get("/api/v1/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    """Serves a saved person crop thumbnail."""
    thumb_dir = _cfg.get("storage", {}).get("thumbnail_dir", "/tmp/thumbnails")
    path = Path(thumb_dir) / filename
    if not path.exists() or not path.suffix in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(404, "Thumbnail not found")
    return FileResponse(str(path), media_type="image/jpeg")


# ── WebSocket Alert Streams ──────────────────────────────────────────────────

@app.websocket("/ws/alerts/{query_id}")
async def ws_query_alerts(websocket: WebSocket, query_id: str):
    """
    WebSocket stream for real-time alerts matching a specific live query.

    Subscribe to this after POST /api/v1/live-queries.
    Each message is a JSON alert object:
    {
        "type": "attribute_match",
        "query_id": "...",
        "detection_id": "...",
        "camera_id": "...",
        "timestamp": 1700000000000,
        "attributes": {"glasses": "on", ...},
        "thumbnail_url": "/api/v1/thumbnails/xxx.jpg",
        ...
    }
    """
    await websocket.accept()
    channel = f"alerts:{query_id}"
    log.info(f"WebSocket connected: query={query_id}")

    # Track connection
    if query_id not in _active_ws_connections:
        _active_ws_connections[query_id] = set()
    _active_ws_connections[query_id].add(websocket)

    try:
        # Subscribe to Redis pub/sub
        pubsub = _redis.pubsub()
        pubsub.subscribe(channel)

        await websocket.send_json({"type": "subscribed", "query_id": query_id})

        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.05)
            if message and message["type"] == "message":
                try:
                    alert = json.loads(message["data"])
                    # Add thumbnail URL
                    thumb = alert.get("thumbnail_path")
                    if thumb and Path(thumb).exists():
                        alert["thumbnail_url"] = f"/api/v1/thumbnails/{Path(thumb).name}"
                    await websocket.send_json(alert)
                except Exception as e:
                    log.warning(f"WS send failed: {e}")

            # Small async yield
            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected: query={query_id}")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
    finally:
        _active_ws_connections.get(query_id, set()).discard(websocket)
        try:
            pubsub.unsubscribe(channel)
        except Exception:
            pass


@app.websocket("/ws/alerts/global")
async def ws_global_alerts(websocket: WebSocket):
    """WebSocket stream for ALL alerts across all live queries."""
    await websocket.accept()
    log.info("Global alerts WebSocket connected")

    try:
        pubsub = _redis.pubsub()
        pubsub.subscribe("alerts:global")
        await websocket.send_json({"type": "subscribed", "scope": "global"})

        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.05)
            if message and message["type"] == "message":
                try:
                    alert = json.loads(message["data"])
                    thumb = alert.get("thumbnail_path")
                    if thumb and Path(thumb).exists():
                        alert["thumbnail_url"] = f"/api/v1/thumbnails/{Path(thumb).name}"
                    await websocket.send_json(alert)
                except Exception as e:
                    log.warning(f"Global WS send failed: {e}")
            await asyncio.sleep(0.02)

    except WebSocketDisconnect:
        log.info("Global alerts WebSocket disconnected")
    except Exception as e:
        log.error(f"Global WS error: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — API Server")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["VAP_CONFIG"] = args.config
    uvicorn.run(
        "api.api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers if not args.reload else 1,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
