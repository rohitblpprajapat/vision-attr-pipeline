"""
Milvus vector database utilities for the Vision Attribute Pipeline.

Schema:
  - detection_id (VARCHAR pk)  — UUID of this specific detection event
  - track_id     (VARCHAR)     — persistent track across frames
  - camera_id    (VARCHAR)     — source camera identifier
  - timestamp    (INT64)       — unix timestamp ms
  - reid_embed   (FLOAT_VECTOR 512) — OSNet ReID embedding (L2-normalised → cosine via IP)
  - attributes   (VARCHAR)     — JSON-encoded PAR attribute dict
  - thumbnail_path (VARCHAR)   — path to saved crop thumbnail
  - bbox_json    (VARCHAR)     — JSON [x1,y1,x2,y2] in original frame coords

Indexes:
  - HNSW on reid_embed for ANN search
  - Scalar index on camera_id, timestamp for filtered queries
"""
import json
import time
import uuid
from typing import Dict, List, Optional, Any
import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
    MilvusException,
)
from pipeline.utils.logging_utils import get_logger, PerfTimer

log = get_logger(__name__)

COLLECTION_NAME = "person_detections"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect_milvus(cfg: dict, max_retries: int = 10) -> None:
    """Establishes Milvus connection with exponential backoff."""
    host = cfg.get("host", "localhost")
    port = int(cfg.get("port", 19530))
    for attempt in range(max_retries):
        try:
            connections.connect("default", host=host, port=port)
            log.info(f"Milvus connected at {host}:{port}")
            return
        except MilvusException as e:
            wait = min(2 ** attempt, 30)
            log.warning(f"Milvus connect attempt {attempt+1}/{max_retries} failed: {e}. Retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Could not connect to Milvus after retries")


# ---------------------------------------------------------------------------
# Schema & collection creation
# ---------------------------------------------------------------------------

def _build_schema(reid_dim: int) -> CollectionSchema:
    fields = [
        FieldSchema("detection_id",   DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
        FieldSchema("track_id",        DataType.VARCHAR, max_length=64),
        FieldSchema("camera_id",       DataType.VARCHAR, max_length=64),
        FieldSchema("timestamp",       DataType.INT64),           # unix ms
        FieldSchema("reid_embed",      DataType.FLOAT_VECTOR,  dim=reid_dim),
        FieldSchema("attributes",      DataType.VARCHAR, max_length=4096),  # JSON
        FieldSchema("thumbnail_path",  DataType.VARCHAR, max_length=512),
        FieldSchema("bbox_json",       DataType.VARCHAR, max_length=128),
        FieldSchema("confidence",      DataType.FLOAT),
    ]
    return CollectionSchema(fields, description="Person detection events with attributes")


def ensure_collection(cfg: dict) -> Collection:
    """
    Creates the collection and required indexes if they don't exist.
    Returns the Collection object (loaded into memory).
    """
    collection_name = cfg.get("collection", COLLECTION_NAME)
    reid_dim = int(cfg.get("reid_dim", 512))
    ttl_seconds = int(cfg.get("ttl_seconds", 604800))  # 7 days

    if utility.has_collection(collection_name):
        log.info(f"Using existing Milvus collection '{collection_name}'")
        col = Collection(collection_name)
        col.load()
        return col

    log.info(f"Creating Milvus collection '{collection_name}'")
    schema = _build_schema(reid_dim)
    col = Collection(
        name=collection_name,
        schema=schema,
        properties={"collection.ttl.seconds": ttl_seconds},
    )

    # Vector index — HNSW for <1ms latency at millions of vectors
    idx_type = cfg.get("index_type", "HNSW")
    idx_params = cfg.get("index_params", {"M": 32, "efConstruction": 200})
    col.create_index(
        field_name="reid_embed",
        index_params={"index_type": idx_type, "metric_type": "IP", "params": idx_params},
    )

    # Scalar indexes for fast filtering
    col.create_index("camera_id",  index_params={"index_type": "Trie"})
    col.create_index("track_id",   index_params={"index_type": "Trie"})
    col.create_index("timestamp",  index_params={"index_type": "STL_SORT"})

    col.load()
    log.info(f"Milvus collection '{collection_name}' created and loaded")
    return col


# ---------------------------------------------------------------------------
# Insert
# ---------------------------------------------------------------------------

def insert_detections(col: Collection, records: List[dict]) -> int:
    """
    Bulk-inserts a list of detection records into Milvus.

    Each record dict must have:
        detection_id, track_id, camera_id, timestamp,
        reid_embed (list/ndarray, float32, L2-normed),
        attributes (dict), thumbnail_path, bbox, confidence

    Returns number of inserted rows.
    """
    if not records:
        return 0

    data = {
        "detection_id":  [],
        "track_id":      [],
        "camera_id":     [],
        "timestamp":     [],
        "reid_embed":    [],
        "attributes":    [],
        "thumbnail_path":[],
        "bbox_json":     [],
        "confidence":    [],
    }

    for r in records:
        embed = np.array(r["reid_embed"], dtype=np.float32)
        # L2-normalise to enable cosine similarity via inner product
        norm = np.linalg.norm(embed)
        if norm > 1e-6:
            embed = embed / norm

        data["detection_id"].append(r.get("detection_id") or str(uuid.uuid4()))
        data["track_id"].append(str(r.get("track_id", "")))
        data["camera_id"].append(str(r.get("camera_id", "")))
        data["timestamp"].append(int(r.get("timestamp", time.time() * 1000)))
        data["reid_embed"].append(embed.tolist())
        data["attributes"].append(json.dumps(r.get("attributes", {}), default=str))
        data["thumbnail_path"].append(str(r.get("thumbnail_path", "")))
        data["bbox_json"].append(json.dumps(r.get("bbox", [])))
        data["confidence"].append(float(r.get("confidence", 0.0)))

    try:
        with PerfTimer("milvus_insert", log):
            mut = col.insert(list(data.values()))
        count = len(mut.primary_keys)
        log.debug(f"Inserted {count} detection(s) into Milvus")
        return count
    except MilvusException as e:
        log.error(f"Milvus insert failed: {e}")
        return 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def attribute_to_filter(attributes: dict) -> str:
    """
    Converts a user attribute query dict to a Milvus scalar filter expression.

    The attributes are stored as JSON strings. We use Milvus's JSON_CONTAINS
    approach via string matching in the JSON field.

    Note: For complex attribute filtering, this performs a JSON_CONTAINS
    style check. Milvus 2.4+ supports JSON path indexing for faster filtering.

    Example:
        {"glasses": "on", "upper_color": "white"}
        → 'attributes like "%\"glasses\": \"on\"%" and attributes like "%\"upper_color\": \"white\"%"'
    """
    if not attributes:
        return ""
    parts = []
    for key, value in attributes.items():
        # Simple string match on the JSON blob — reliable for known key/value combos
        # For high-QPS deployments, consider storing attributes as separate scalar fields
        escaped = json.dumps({key: value})[1:-1].strip()  # Remove outer braces
        parts.append(f'attributes like "%{escaped.replace(chr(34), chr(34))[:100]}%"')
    return " and ".join(parts)


def search_by_embedding(
    col: Collection,
    query_embed: np.ndarray,
    top_k: int = 50,
    attribute_filter: Optional[dict] = None,
    camera_ids: Optional[List[str]] = None,
    time_range: Optional[tuple] = None,   # (start_ms, end_ms)
    cfg: Optional[dict] = None,
) -> List[dict]:
    """
    ANN vector search with optional scalar filters.

    Returns list of result dicts sorted by similarity score (descending).
    """
    if cfg is None:
        cfg = {}
    search_params = cfg.get("search_params", {"ef": 128})

    # L2-normalise query
    q = np.array(query_embed, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm > 1e-6:
        q = q / norm

    # Build filter expression
    filter_parts = []

    if attribute_filter:
        attr_expr = attribute_to_filter(attribute_filter)
        if attr_expr:
            filter_parts.append(f"({attr_expr})")

    if camera_ids:
        cam_list = ", ".join([f'"{c}"' for c in camera_ids])
        filter_parts.append(f"camera_id in [{cam_list}]")

    if time_range:
        start_ms, end_ms = time_range
        filter_parts.append(f"timestamp >= {int(start_ms)} and timestamp <= {int(end_ms)}")

    filter_expr = " and ".join(filter_parts) if filter_parts else None

    output_fields = [
        "detection_id", "track_id", "camera_id",
        "timestamp", "attributes", "thumbnail_path",
        "bbox_json", "confidence",
    ]

    try:
        with PerfTimer("milvus_search", log):
            results = col.search(
                data=[q.tolist()],
                anns_field="reid_embed",
                param={"metric_type": "IP", "params": search_params},
                limit=top_k,
                expr=filter_expr,
                output_fields=output_fields,
            )

        hits = []
        for hit in results[0]:
            try:
                attrs = json.loads(hit.entity.get("attributes") or "{}")
            except Exception:
                attrs = {}
            hits.append({
                "detection_id":   hit.entity.get("detection_id"),
                "track_id":       hit.entity.get("track_id"),
                "camera_id":      hit.entity.get("camera_id"),
                "timestamp":      hit.entity.get("timestamp"),
                "score":          float(hit.score),
                "attributes":     attrs,
                "thumbnail_path": hit.entity.get("thumbnail_path"),
                "bbox":           json.loads(hit.entity.get("bbox_json") or "[]"),
                "confidence":     hit.entity.get("confidence"),
            })
        return hits

    except MilvusException as e:
        log.error(f"Milvus search failed: {e}")
        return []


def search_by_attributes_only(
    col: Collection,
    attribute_filter: dict,
    camera_ids: Optional[List[str]] = None,
    time_range: Optional[tuple] = None,
    limit: int = 200,
) -> List[dict]:
    """
    Pure scalar query (no vector) — used for broad attribute sweeps.
    Returns up to `limit` matching records ordered by timestamp descending.
    """
    filter_parts = []

    if attribute_filter:
        attr_expr = attribute_to_filter(attribute_filter)
        if attr_expr:
            filter_parts.append(f"({attr_expr})")

    if camera_ids:
        cam_list = ", ".join([f'"{c}"' for c in camera_ids])
        filter_parts.append(f"camera_id in [{cam_list}]")

    if time_range:
        start_ms, end_ms = time_range
        filter_parts.append(f"timestamp >= {int(start_ms)} and timestamp <= {int(end_ms)}")

    if not filter_parts:
        log.warning("search_by_attributes_only called with no filters — refusing full scan")
        return []

    filter_expr = " and ".join(filter_parts)
    output_fields = [
        "detection_id", "track_id", "camera_id",
        "timestamp", "attributes", "thumbnail_path",
        "bbox_json", "confidence",
    ]

    try:
        with PerfTimer("milvus_query", log):
            results = col.query(
                expr=filter_expr,
                output_fields=output_fields,
                limit=limit,
            )
        hits = []
        for r in results:
            try:
                attrs = json.loads(r.get("attributes") or "{}")
            except Exception:
                attrs = {}
            hits.append({
                "detection_id":   r.get("detection_id"),
                "track_id":       r.get("track_id"),
                "camera_id":      r.get("camera_id"),
                "timestamp":      r.get("timestamp"),
                "score":          1.0,  # No similarity score for scalar-only
                "attributes":     attrs,
                "thumbnail_path": r.get("thumbnail_path"),
                "bbox":           json.loads(r.get("bbox_json") or "[]"),
                "confidence":     r.get("confidence"),
            })
        # Sort by timestamp descending
        hits.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
        return hits

    except MilvusException as e:
        log.error(f"Milvus query failed: {e}")
        return []


def collection_stats(col: Collection) -> dict:
    """Returns basic collection statistics for health monitoring."""
    try:
        return {
            "num_entities": col.num_entities,
            "name": col.name,
            "schema": [f.name for f in col.schema.fields],
        }
    except Exception as e:
        return {"error": str(e)}
