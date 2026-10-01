"""
Search Engine — Vision Attribute Pipeline
==========================================
Provides the query interface for:
  1. Attribute-only search: "show all people wearing white upper + glasses ON"
  2. Appearance similarity search: "find people who look like this image crop"
  3. Hybrid search: attribute filter + embedding similarity combined

This module is imported by the API server and can also be used as a library.

Example:
    engine = SearchEngine(cfg)
    engine.connect()
    results = engine.query_by_attributes(
        {"upper_color": "white", "glasses": "on"},
        time_window_hours=6,
    )
"""
import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pipeline.utils.logging_utils import get_logger, PerfTimer
from pipeline.utils.milvus_utils import (
    connect_milvus,
    ensure_collection,
    search_by_embedding,
    search_by_attributes_only,
)
from pymilvus import Collection

log = get_logger(__name__)

# Valid attribute values for query validation
VALID_ATTRIBUTES = {
    "gender":       {"male", "female"},
    "age_group":    {"young", "adult", "old"},
    "upper_color":  {"black", "white", "red", "blue", "green", "yellow", "gray", "purple", "brown", "pink", "orange"},
    "lower_color":  {"black", "white", "blue", "gray", "brown", "green"},
    "upper_type":   {"tshirt", "shirt", "jacket", "coat", "dress", "sweater", "hoodie", "vest", "other"},
    "lower_type":   {"pants", "jeans", "shorts", "skirt", "leggings"},
    "hair_length":  {"short", "medium", "long", "bald"},
    "glasses":      {"on", "off"},
    "hat":          {"on", "off"},
    "bag_type":     {"backpack", "handbag", "shoulder_bag", "none"},
    "action":       {"walking", "standing", "running", "sitting", "cycling"},
}


class QueryValidationError(ValueError):
    pass


def validate_attributes(attrs: dict) -> dict:
    """
    Validates and normalises an attribute query dict.
    Raises QueryValidationError on unknown keys or invalid values.
    Returns cleaned dict.
    """
    cleaned = {}
    for key, value in attrs.items():
        if key not in VALID_ATTRIBUTES:
            raise QueryValidationError(
                f"Unknown attribute '{key}'. Valid keys: {sorted(VALID_ATTRIBUTES.keys())}"
            )
        valid_vals = VALID_ATTRIBUTES[key]
        if isinstance(value, list):
            for v in value:
                if v not in valid_vals:
                    raise QueryValidationError(
                        f"Invalid value '{v}' for '{key}'. Valid: {sorted(valid_vals)}"
                    )
            cleaned[key] = value
        else:
            value = str(value).lower().strip()
            if value not in valid_vals:
                raise QueryValidationError(
                    f"Invalid value '{value}' for '{key}'. Valid: {sorted(valid_vals)}"
                )
            cleaned[key] = value
    return cleaned


class SearchEngine:
    """
    Main search interface. Connect once and reuse.
    Thread-safe for read operations.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._col: Optional[Collection] = None
        self._reid_model = None   # Loaded lazily for image-based search
        self._connected = False

    def connect(self):
        """Connects to Milvus and prepares the collection."""
        connect_milvus(self.cfg["milvus"])
        self._col = ensure_collection(self.cfg["milvus"])
        self._connected = True
        log.info("SearchEngine connected to Milvus")

    def _ensure_connected(self):
        if not self._connected:
            self.connect()

    # -----------------------------------------------------------------------
    # 1. Attribute-only search
    # -----------------------------------------------------------------------

    def query_by_attributes(
        self,
        attributes: dict,
        camera_ids: Optional[List[str]] = None,
        time_window_hours: float = 24,
        limit: int = 200,
    ) -> List[dict]:
        """
        Search for detections matching given attributes.

        Args:
            attributes: Dict of attribute_key → value.
                        e.g. {"upper_color": "white", "glasses": "on"}
            camera_ids: Optional list of camera IDs to restrict search.
            time_window_hours: How far back to search (default: last 24h).
            limit: Max results to return.

        Returns:
            List of detection dicts, sorted by recency (newest first).
        """
        self._ensure_connected()

        try:
            attributes = validate_attributes(attributes)
        except QueryValidationError as e:
            raise

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(time_window_hours * 3600 * 1000)
        time_range = (start_ms, now_ms)

        with PerfTimer("attr_query", log):
            results = search_by_attributes_only(
                self._col,
                attribute_filter=attributes,
                camera_ids=camera_ids,
                time_range=time_range,
                limit=limit,
            )

        log.info(
            f"Attribute query returned {len(results)} results",
            extra={"_attrs": attributes, "_num_results": len(results)},
        )
        return results

    # -----------------------------------------------------------------------
    # 2. Embedding / appearance similarity search
    # -----------------------------------------------------------------------

    def query_by_image(
        self,
        image_bytes: bytes,
        attribute_filter: Optional[dict] = None,
        camera_ids: Optional[List[str]] = None,
        time_window_hours: float = 24,
        top_k: int = 50,
        similarity_threshold: float = 0.60,
    ) -> List[dict]:
        """
        Find people who look similar to the provided person crop image.

        Args:
            image_bytes: JPEG/PNG bytes of a person crop.
            attribute_filter: Optional attribute constraints on results.
            camera_ids: Optional camera filter.
            time_window_hours: Search window in hours.
            top_k: Number of ANN results to return.
            similarity_threshold: Minimum cosine similarity (0–1).

        Returns:
            List of results sorted by similarity (highest first).
        """
        self._ensure_connected()

        embed = self._extract_query_embedding(image_bytes)
        if embed is None:
            raise ValueError("Could not extract embedding from provided image")

        if attribute_filter:
            attribute_filter = validate_attributes(attribute_filter)

        now_ms = int(time.time() * 1000)
        start_ms = now_ms - int(time_window_hours * 3600 * 1000)

        with PerfTimer("embed_query", log):
            results = search_by_embedding(
                self._col,
                query_embed=embed,
                top_k=top_k,
                attribute_filter=attribute_filter,
                camera_ids=camera_ids,
                time_range=(start_ms, now_ms),
                cfg=self.cfg.get("milvus"),
            )

        # Filter by similarity threshold
        results = [r for r in results if r["score"] >= similarity_threshold]
        log.info(
            f"Image similarity query returned {len(results)} results "
            f"above threshold {similarity_threshold}",
        )
        return results

    # -----------------------------------------------------------------------
    # 3. Hybrid search (attributes + image)
    # -----------------------------------------------------------------------

    def query_hybrid(
        self,
        image_bytes: Optional[bytes] = None,
        attributes: Optional[dict] = None,
        camera_ids: Optional[List[str]] = None,
        time_window_hours: float = 24,
        top_k: int = 50,
        similarity_threshold: float = 0.55,
    ) -> List[dict]:
        """
        Hybrid: combine attribute filtering with embedding similarity.
        If no image provided, falls back to attribute-only search.
        """
        if image_bytes is None and attributes:
            return self.query_by_attributes(
                attributes, camera_ids, time_window_hours
            )
        elif image_bytes is not None:
            return self.query_by_image(
                image_bytes, attributes, camera_ids, time_window_hours, top_k, similarity_threshold
            )
        else:
            raise ValueError("Must provide at least 'attributes' or 'image_bytes'")

    # -----------------------------------------------------------------------
    # Active query registration (for real-time alerts)
    # -----------------------------------------------------------------------

    def register_live_query(
        self,
        redis_client,
        attributes: dict,
        camera_ids: Optional[List[str]] = None,
        description: str = "",
    ) -> str:
        """
        Registers a live query. Matching detections will be published
        to Redis pub/sub channel 'alerts:{query_id}' in real-time.

        Returns the query_id.
        """
        attributes = validate_attributes(attributes)
        query_id = str(uuid.uuid4())
        query_spec = {
            "query_id":    query_id,
            "attributes":  attributes,
            "camera_ids":  camera_ids,
            "description": description,
            "created_at":  int(time.time() * 1000),
        }
        redis_client.hset("active_queries", query_id, json.dumps(query_spec))
        log.info(f"Registered live query {query_id}: {attributes}")
        return query_id

    def unregister_live_query(self, redis_client, query_id: str) -> bool:
        """Removes a live query. Returns True if it existed."""
        removed = redis_client.hdel("active_queries", query_id)
        return removed > 0

    def list_live_queries(self, redis_client) -> List[dict]:
        """Returns all currently active live queries."""
        raw = redis_client.hgetall("active_queries")
        queries = []
        for k, v in raw.items():
            try:
                queries.append(json.loads(v))
            except Exception:
                pass
        return queries

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_reid_model(self):
        """Lazily loads ReID model for query-time embedding extraction."""
        if self._reid_model is None:
            log.info("Loading ReID model for query-time embedding extraction…")
            try:
                from torchreid.utils import FeatureExtractor
                import torch
                reid_cfg = self.cfg.get("models", {}).get("reid", {})
                weights = reid_cfg.get("weights_path", "models/osnet_ain_x1_0_market1501.pth")
                device = reid_cfg.get("device", "cuda:0")
                device_str = device if torch.cuda.is_available() else "cpu"
                model_path = weights if Path(weights).exists() else None
                self._reid_model = FeatureExtractor(
                    model_name=reid_cfg.get("model_name", "osnet_ain_x1_0"),
                    model_path=model_path,
                    device=device_str,
                    verbose=False,
                )
                log.info("ReID model loaded for search engine")
            except Exception as e:
                log.error(f"Failed to load ReID model: {e}")
                self._reid_model = None
        return self._reid_model

    def _extract_query_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Decodes an image and extracts its ReID embedding."""
        import torch
        try:
            buf = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                return None
            reid = self._get_reid_model()
            if reid is None:
                return None
            feats = reid([img])
            embed = feats.cpu().numpy().astype(np.float32).squeeze()
            norm = np.linalg.norm(embed)
            if norm > 1e-6:
                embed = embed / norm
            return embed
        except Exception as e:
            log.error(f"Embedding extraction failed: {e}", exc_info=True)
            return None

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def get_valid_attributes(self) -> dict:
        """Returns the attribute schema (valid keys and values)."""
        return {k: sorted(v) for k, v in VALID_ATTRIBUTES.items()}

    def stats(self) -> dict:
        """Returns collection statistics."""
        if self._col is None:
            return {"status": "not_connected"}
        try:
            from pipeline.utils.milvus_utils import collection_stats
            return collection_stats(self._col)
        except Exception as e:
            return {"error": str(e)}
