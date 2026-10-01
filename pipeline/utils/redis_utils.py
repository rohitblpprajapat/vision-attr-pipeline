"""
Redis Streams utilities for the Vision Attribute Pipeline.

Wraps redis-py to provide:
  - Frame/detection publishing
  - Consumer group management
  - Backpressure via MAXLEN
  - Serialization (msgpack for speed, JSON fallback)
"""
import os
import time
import json
import msgpack
from typing import Any, Dict, Generator, List, Optional, Tuple
import redis
from pipeline.utils.logging_utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def create_redis_client(cfg: dict) -> redis.Redis:
    """Creates and returns a Redis client, with retry on first connect."""
    client = redis.Redis(
        host=cfg.get("host", "localhost"),
        port=int(cfg.get("port", 6379)),
        password=cfg.get("password") or None,
        db=int(cfg.get("db", 0)),
        decode_responses=False,   # We handle bytes ourselves
        socket_connect_timeout=10,
        socket_keepalive=True,
        retry_on_timeout=True,
        health_check_interval=30,
    )
    # Verify connection
    for attempt in range(10):
        try:
            client.ping()
            log.info("Redis connected", extra={"_host": cfg.get("host"), "_port": cfg.get("port")})
            return client
        except redis.exceptions.ConnectionError as e:
            log.warning(f"Redis connect attempt {attempt+1}/10 failed: {e}. Retrying in 3s…")
            time.sleep(3)
    raise RuntimeError("Could not connect to Redis after 10 attempts")


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def pack(payload: dict) -> bytes:
    """Serialize dict → msgpack bytes (fast, compact)."""
    return msgpack.packb(payload, use_bin_type=True)


def unpack(raw: bytes) -> dict:
    """Deserialize msgpack bytes → dict."""
    return msgpack.unpackb(raw, raw=False)


# ---------------------------------------------------------------------------
# Stream producer
# ---------------------------------------------------------------------------

def stream_publish(
    client: redis.Redis,
    stream_name: str,
    payload: dict,
    maxlen: int = 5000,
    approximate: bool = True,
) -> str:
    """
    Publish a payload dict to a Redis Stream.

    Returns the message ID.
    MAXLEN trims the stream to prevent unbounded growth.
    """
    encoded = pack(payload)
    msg_id = client.xadd(
        stream_name,
        {"data": encoded},
        maxlen=maxlen,
        approximate=approximate,
    )
    return msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)


# ---------------------------------------------------------------------------
# Consumer group management
# ---------------------------------------------------------------------------

def ensure_consumer_group(
    client: redis.Redis,
    stream_name: str,
    group_name: str,
    start_id: str = "0",
) -> None:
    """
    Creates stream + consumer group if they don't exist.
    `start_id="$"` means only new messages; `"0"` means replay all pending.
    """
    # Create stream if absent (MKSTREAM)
    try:
        client.xgroup_create(stream_name, group_name, id=start_id, mkstream=True)
        log.info(f"Created consumer group '{group_name}' on '{stream_name}'")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            pass  # Group already exists — expected on restart
        else:
            raise


# ---------------------------------------------------------------------------
# Consumer (blocking read loop)
# ---------------------------------------------------------------------------

def stream_consume(
    client: redis.Redis,
    stream_name: str,
    group_name: str,
    consumer_name: str,
    batch_size: int = 10,
    block_ms: int = 1000,
) -> Generator[Tuple[str, dict], None, None]:
    """
    Generator: yields (msg_id, payload_dict) from a Redis Stream consumer group.

    Caller is responsible for ACKing after successful processing:
        client.xack(stream_name, group_name, msg_id)

    Handles:
      - Claiming stale pending messages (dead-consumer recovery)
      - Graceful empty-stream handling
    """
    # First, try to reclaim any pending messages that timed out (> 30s)
    try:
        pending = client.xpending_range(
            stream_name, group_name, "-", "+", count=100
        )
        for entry in pending:
            msg_id = entry["message_id"]
            idle_ms = entry["time_since_delivered"]
            if idle_ms > 30_000:  # 30 seconds stale
                claimed = client.xautoclaim(
                    stream_name, group_name, consumer_name, 30_000, msg_id, count=1
                )
                log.warning(f"Reclaimed stale message {msg_id} (idle {idle_ms}ms)")
    except Exception as e:
        log.debug(f"Pending reclaim skipped: {e}")

    while True:
        try:
            results = client.xreadgroup(
                group_name,
                consumer_name,
                {stream_name: ">"},
                count=batch_size,
                block=block_ms,
            )
            if not results:
                continue
            for _stream, messages in results:
                for msg_id, fields in messages:
                    msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    raw = fields.get(b"data") or fields.get("data")
                    if raw is None:
                        client.xack(stream_name, group_name, msg_id)
                        continue
                    try:
                        payload = unpack(raw)
                    except Exception as e:
                        log.error(f"Failed to decode message {msg_id_str}: {e}")
                        client.xack(stream_name, group_name, msg_id)
                        continue
                    yield msg_id_str, payload

        except redis.exceptions.ConnectionError as e:
            log.error(f"Redis connection lost: {e}. Reconnecting in 5s…")
            time.sleep(5)
        except Exception as e:
            log.error(f"Unexpected error in stream_consume: {e}")
            time.sleep(1)


# ---------------------------------------------------------------------------
# Alert pub/sub (for real-time WebSocket push)
# ---------------------------------------------------------------------------

def publish_alert(client: redis.Redis, channel: str, alert: dict) -> None:
    """Publishes an alert to a Redis pub/sub channel."""
    client.publish(channel, json.dumps(alert, default=str))


def subscribe_alerts(client: redis.Redis, channel: str):
    """Returns a Redis pub/sub subscriber for alert consumption."""
    pubsub = client.pubsub()
    pubsub.subscribe(channel)
    return pubsub


# ---------------------------------------------------------------------------
# Health check helper
# ---------------------------------------------------------------------------

def stream_stats(client: redis.Redis, stream_name: str) -> dict:
    """Returns stream length and group lag info."""
    try:
        info = client.xinfo_stream(stream_name)
        groups = client.xinfo_groups(stream_name)
        return {
            "length": info.get("length", 0),
            "groups": [
                {
                    "name": g.get("name", b"").decode() if isinstance(g.get("name"), bytes) else g.get("name"),
                    "pending": g.get("pending", 0),
                    "lag": g.get("lag", 0),
                }
                for g in groups
            ],
        }
    except Exception as e:
        return {"error": str(e)}
