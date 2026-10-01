"""
Storage Worker — Vision Attribute Pipeline
==========================================
Consumes enriched detections from Redis and writes them to Milvus.
Also publishes real-time alerts to Redis pub/sub for active queries.

Features:
  - Batch writes to Milvus (configurable batch size + timeout)
  - Deduplication: skips same track_id within configurable window
  - Active query matching for real-time alerting
  - Periodic Milvus flush to ensure data visibility

Usage:
    python -m pipeline.storage_worker \
        --config config/config.yaml \
        --worker-id storage-01
"""
import argparse
import json
import signal
import time
import uuid
from typing import Dict, List, Optional, Set

from pipeline.utils.logging_utils import get_logger, load_config, PerfTimer
from pipeline.utils.milvus_utils import connect_milvus, ensure_collection, insert_detections
from pipeline.utils.redis_utils import (
    create_redis_client,
    ensure_consumer_group,
    stream_consume,
    publish_alert,
)

log = get_logger(__name__)
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    log.info("Shutdown signal — stopping storage worker…")
    _shutdown = True


# ---------------------------------------------------------------------------
# Active query registry (in-memory, synced from Redis hash)
# ---------------------------------------------------------------------------

class ActiveQueryRegistry:
    """
    Tracks live attribute queries registered by the API.
    Each query has an attribute filter and a Redis pub/sub channel for alerts.

    Queries are stored in Redis hash:  active_queries:{query_id} → JSON
    TTL per query = 24h (queries auto-expire)
    """

    QUERIES_KEY = "active_queries"

    def __init__(self, redis_client):
        self.redis = redis_client
        self._cache: Dict[str, dict] = {}
        self._last_refresh = 0.0
        self._refresh_interval = 10.0   # seconds

    def refresh(self):
        """Pulls active queries from Redis."""
        now = time.time()
        if now - self._last_refresh < self._refresh_interval:
            return
        try:
            raw = self.redis.hgetall(self.QUERIES_KEY)
            queries = {}
            for k, v in raw.items():
                key = k.decode() if isinstance(k, bytes) else k
                try:
                    queries[key] = json.loads(v)
                except Exception:
                    pass
            self._cache = queries
            self._last_refresh = now
        except Exception as e:
            log.warning(f"Failed to refresh active queries: {e}")

    def register(self, query_id: str, query_spec: dict, ttl: int = 86400):
        """Registers a new active query."""
        self.redis.hset(self.QUERIES_KEY, query_id, json.dumps(query_spec))
        self._cache[query_id] = query_spec

    def unregister(self, query_id: str):
        """Removes a query."""
        self.redis.hdel(self.QUERIES_KEY, query_id)
        self._cache.pop(query_id, None)

    def match(self, detection: dict) -> List[dict]:
        """
        Returns list of (query_id, query_spec) for queries that match this detection.
        Matching is attribute-based (AND logic).
        """
        self.refresh()
        matches = []
        det_attrs = detection.get("attributes", {})

        for qid, qspec in self._cache.items():
            filters = qspec.get("attributes", {})
            if not filters:
                continue

            matched = True
            for attr_key, attr_val in filters.items():
                det_val = det_attrs.get(attr_key)
                if isinstance(attr_val, list):
                    if det_val not in attr_val:
                        matched = False
                        break
                else:
                    if det_val != attr_val:
                        matched = False
                        break

            if matched:
                camera_filter = qspec.get("camera_ids")
                if camera_filter and detection.get("camera_id") not in camera_filter:
                    matched = False

            if matched:
                matches.append({"query_id": qid, "query_spec": qspec})

        return matches


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TrackDeduplicator:
    """
    Prevents storing redundant detections for the same track within a time window.
    Uses Redis for persistence across worker restarts.
    """

    def __init__(self, redis_client, window_seconds: int = 5):
        self.redis = redis_client
        self.window = window_seconds

    def should_store(self, track_id: str, camera_id: str) -> bool:
        """Returns True if this track should be stored (not a recent duplicate)."""
        key = f"dedup:{camera_id}:{track_id}"
        result = self.redis.set(key, "1", ex=self.window, nx=True)
        return result is True   # True = key was set (not duplicate); None = already existed


# ---------------------------------------------------------------------------
# Storage Worker
# ---------------------------------------------------------------------------

class StorageWorker:
    def __init__(self, cfg: dict, worker_id: str):
        self.cfg = cfg
        self.worker_id = worker_id
        self.redis = create_redis_client(cfg["redis"])

        streams_cfg = cfg["redis"]["streams"]
        groups_cfg = cfg["redis"]["groups"]
        self.in_stream = streams_cfg["enriched"]
        self.group_name = groups_cfg["storage"]
        self.alerts_channel = streams_cfg.get("alerts", "stream:alerts")

        storage_cfg = cfg.get("storage", {})
        self.dedup_window = int(storage_cfg.get("dedup_window_seconds", 5))

        # Batch write config
        self.batch_size = 50
        self.batch_timeout = 1.0    # seconds

        # Milvus
        self._col = None
        self._query_registry = ActiveQueryRegistry(self.redis)
        self._dedup = TrackDeduplicator(self.redis, self.dedup_window)

        # Cooldown tracker for alert spam prevention
        self._alert_cooldowns: Dict[str, float] = {}
        self.alert_cooldown = int(cfg.get("alerts", {}).get("cooldown_seconds", 60))

    def _load_milvus(self):
        milvus_cfg = self.cfg["milvus"]
        connect_milvus(milvus_cfg)
        self._col = ensure_collection(milvus_cfg)
        log.info("Milvus collection ready", extra={"_worker_id": self.worker_id})

    def _should_alert(self, track_id: str, query_id: str) -> bool:
        key = f"{track_id}:{query_id}"
        last_alert = self._alert_cooldowns.get(key, 0)
        if time.time() - last_alert > self.alert_cooldown:
            self._alert_cooldowns[key] = time.time()
            # Clean up old cooldowns periodically
            if len(self._alert_cooldowns) > 10000:
                cutoff = time.time() - self.alert_cooldown * 2
                self._alert_cooldowns = {
                    k: v for k, v in self._alert_cooldowns.items() if v > cutoff
                }
            return True
        return False

    def process_batch(self, records: List[dict]) -> int:
        """Writes a batch to Milvus and fires any matching alerts."""
        if not records:
            return 0

        # Dedup and prepare for Milvus
        to_insert = []
        for rec in records:
            track_id = rec.get("track_id", "")
            camera_id = rec.get("camera_id", "")
            if not track_id:
                continue
            if not self._dedup.should_store(track_id, camera_id):
                continue
            to_insert.append(rec)

        inserted = 0
        if to_insert and self._col is not None:
            with PerfTimer("milvus_batch_insert", log):
                inserted = insert_detections(self._col, to_insert)

        # Alert matching (all records, not just deduped ones)
        for rec in records:
            matches = self._query_registry.match(rec)
            for match in matches:
                qid = match["query_id"]
                track_id = rec.get("track_id", "")
                if self._should_alert(track_id, qid):
                    alert = {
                        "type":           "attribute_match",
                        "query_id":       qid,
                        "detection_id":   rec.get("detection_id"),
                        "track_id":       track_id,
                        "camera_id":      rec.get("camera_id"),
                        "timestamp":      rec.get("timestamp"),
                        "attributes":     rec.get("attributes", {}),
                        "thumbnail_path": rec.get("thumbnail_path"),
                        "bbox":           rec.get("bbox"),
                        "confidence":     rec.get("confidence"),
                        "alert_ts":       int(time.time() * 1000),
                    }
                    try:
                        publish_alert(self.redis, f"alerts:{qid}", alert)
                        publish_alert(self.redis, "alerts:global", alert)
                        log.info(
                            f"Alert fired: query={qid} track={track_id} camera={rec.get('camera_id')}",
                            extra={"_query_id": qid, "_track_id": track_id},
                        )
                    except Exception as e:
                        log.warning(f"Failed to publish alert: {e}")

        return inserted

    def run(self):
        global _shutdown
        self._load_milvus()

        ensure_consumer_group(self.redis, self.in_stream, self.group_name, start_id="$")
        log.info(
            f"Storage worker '{self.worker_id}' started",
            extra={"_worker_id": self.worker_id},
        )

        consumer_name = f"storage-{self.worker_id}"
        pending: List[dict] = []
        msg_ids: List[str] = []
        batch_start = time.time()
        total_stored = 0
        flush_count = 0

        for msg_id, payload in stream_consume(
            self.redis, self.in_stream, self.group_name, consumer_name,
            batch_size=self.batch_size,
        ):
            if _shutdown:
                break

            pending.append(payload)
            msg_ids.append(msg_id)

            elapsed = time.time() - batch_start
            if len(pending) >= self.batch_size or elapsed > self.batch_timeout:
                try:
                    inserted = self.process_batch(pending)
                    total_stored += inserted
                    for mid in msg_ids:
                        self.redis.xack(self.in_stream, self.group_name, mid)
                    flush_count += 1
                    if flush_count % 20 == 0:
                        log.info(
                            f"Storage worker: {total_stored} detections stored",
                            extra={"_worker_id": self.worker_id},
                        )
                        # Periodic Milvus flush to ensure strong consistency
                        if self._col:
                            try:
                                self._col.flush()
                            except Exception:
                                pass
                except Exception as e:
                    log.error(f"Batch storage error: {e}", exc_info=True)
                    for mid in msg_ids:
                        self.redis.xack(self.in_stream, self.group_name, mid)

                pending = []
                msg_ids = []
                batch_start = time.time()

        # Final flush
        if pending:
            try:
                self.process_batch(pending)
                for mid in msg_ids:
                    self.redis.xack(self.in_stream, self.group_name, mid)
            except Exception as e:
                log.error(f"Final flush error: {e}")

        log.info(
            f"Storage worker '{self.worker_id}' stopped. Total stored: {total_stored}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vision Attribute Pipeline — Storage Worker")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--worker-id", default="01")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    cfg = load_config(args.config)
    worker = StorageWorker(cfg, args.worker_id)
    worker.run()


if __name__ == "__main__":
    main()
