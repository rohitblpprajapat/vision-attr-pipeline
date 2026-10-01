"""
Metrics — Vision Attribute Pipeline
=====================================
Prometheus metrics for all pipeline workers.
Each worker imports this module and calls the appropriate counters.

A /metrics HTTP endpoint is exposed on a configurable port (default: 9100)
so Prometheus can scrape it.

Usage in a worker:
    from pipeline.metrics import PipelineMetrics
    metrics = PipelineMetrics(worker_type="detector", worker_id="01")
    metrics.start_http_server(port=9100)

    # Then inside processing loops:
    metrics.frames_processed.inc()
    with metrics.inference_latency.labels(model="yolo").time():
        result = model.infer(frame)

Grafana dashboard JSON is in docs/grafana_dashboard.json
"""
import os
import time
import threading
from typing import Optional

# Prometheus client — graceful fallback if not installed
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        start_http_server,
        CollectorRegistry,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Stub implementations so workers don't crash if prometheus_client isn't installed
    class _Stub:
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def set(self, val): pass
        def observe(self, val): pass
        def labels(self, **kw): return self
        def time(self):
            class _Ctx:
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return _Ctx()

    Counter = Gauge = Histogram = Summary = lambda *a, **kw: _Stub()
    def start_http_server(*a, **kw): pass
    REGISTRY = None


from pipeline.utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Metric definitions ────────────────────────────────────────────────────────
# All metrics use the "vap_" prefix (Vision Attribute Pipeline)

# Ingestion
FRAMES_INGESTED = Counter(
    "vap_frames_ingested_total",
    "Total frames published to Redis by the camera ingestor",
    ["camera_id"],
) if PROMETHEUS_AVAILABLE else Counter("", "")

FRAMES_SKIPPED = Counter(
    "vap_frames_skipped_total",
    "Frames skipped due to frame_skip sampling",
    ["camera_id"],
) if PROMETHEUS_AVAILABLE else Counter("", "")

CAMERA_RECONNECTS = Counter(
    "vap_camera_reconnects_total",
    "Number of times a camera stream was reconnected",
    ["camera_id"],
) if PROMETHEUS_AVAILABLE else Counter("", "")

# Detection
DETECTIONS_TOTAL = Counter(
    "vap_detections_total",
    "Total person detections (confirmed tracks)",
    ["camera_id"],
) if PROMETHEUS_AVAILABLE else Counter("", "")

TRACKS_ACTIVE = Gauge(
    "vap_tracks_active",
    "Currently active tracks per camera",
    ["camera_id"],
) if PROMETHEUS_AVAILABLE else Gauge("", "")

DETECTION_LATENCY = Histogram(
    "vap_detection_latency_seconds",
    "YOLOv8 + BoT-SORT inference time per frame",
    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5],
) if PROMETHEUS_AVAILABLE else Histogram("", "")

# Attribute recognition
PAR_RUNS_TOTAL = Counter(
    "vap_par_runs_total",
    "Total PAR model invocations (batches)",
    ["backend"],
) if PROMETHEUS_AVAILABLE else Counter("", "")

PAR_CROPS_TOTAL = Counter(
    "vap_par_crops_total",
    "Total person crops processed by PAR",
) if PROMETHEUS_AVAILABLE else Counter("", "")

PAR_LATENCY = Histogram(
    "vap_par_latency_seconds",
    "PAR inference time per batch",
    buckets=[0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
) if PROMETHEUS_AVAILABLE else Histogram("", "")

REID_LATENCY = Histogram(
    "vap_reid_latency_seconds",
    "ReID embedding extraction time per batch",
    buckets=[0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15],
) if PROMETHEUS_AVAILABLE else Histogram("", "")

REID_BATCH_SIZE = Histogram(
    "vap_reid_batch_size",
    "Number of crops per ReID batch",
    buckets=[1, 2, 4, 8, 16, 32, 64],
) if PROMETHEUS_AVAILABLE else Histogram("", "")

# Storage
MILVUS_INSERTS_TOTAL = Counter(
    "vap_milvus_inserts_total",
    "Total detection records written to Milvus",
) if PROMETHEUS_AVAILABLE else Counter("", "")

MILVUS_INSERT_LATENCY = Histogram(
    "vap_milvus_insert_latency_seconds",
    "Milvus batch insert latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
) if PROMETHEUS_AVAILABLE else Histogram("", "")

DEDUP_SKIPS = Counter(
    "vap_dedup_skips_total",
    "Detections skipped due to deduplication window",
) if PROMETHEUS_AVAILABLE else Counter("", "")

MILVUS_ENTITIES = Gauge(
    "vap_milvus_entities_total",
    "Total entities in Milvus collection",
) if PROMETHEUS_AVAILABLE else Gauge("", "")

# Alerting
ALERTS_FIRED = Counter(
    "vap_alerts_fired_total",
    "Total live query alerts fired",
    ["query_id"],
) if PROMETHEUS_AVAILABLE else Counter("", "")

ACTIVE_LIVE_QUERIES = Gauge(
    "vap_active_live_queries",
    "Number of currently registered live queries",
) if PROMETHEUS_AVAILABLE else Gauge("", "")

# Redis queue health
REDIS_STREAM_LENGTH = Gauge(
    "vap_redis_stream_length",
    "Current length of a Redis stream",
    ["stream_name"],
) if PROMETHEUS_AVAILABLE else Gauge("", "")

REDIS_STREAM_LAG = Gauge(
    "vap_redis_stream_lag",
    "Consumer group lag (unread messages) on a Redis stream",
    ["stream_name", "group_name"],
) if PROMETHEUS_AVAILABLE else Gauge("", "")

REDIS_ERRORS = Counter(
    "vap_redis_errors_total",
    "Redis connection/operation errors",
    ["worker_type"],
) if PROMETHEUS_AVAILABLE else Counter("", "")

# Search queries
SEARCH_REQUESTS = Counter(
    "vap_search_requests_total",
    "Total search API requests",
    ["query_type"],   # attribute | image | hybrid
) if PROMETHEUS_AVAILABLE else Counter("", "")

SEARCH_LATENCY = Histogram(
    "vap_search_latency_seconds",
    "Search query end-to-end latency",
    ["query_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
) if PROMETHEUS_AVAILABLE else Histogram("", "")

SEARCH_RESULTS = Histogram(
    "vap_search_results_count",
    "Number of results returned per search query",
    ["query_type"],
    buckets=[0, 1, 5, 10, 25, 50, 100, 200, 500],
) if PROMETHEUS_AVAILABLE else Histogram("", "")

# GPU / system resources
GPU_MEMORY_USED_MB = Gauge(
    "vap_gpu_memory_used_mb",
    "GPU VRAM currently used in MB",
    ["device"],
) if PROMETHEUS_AVAILABLE else Gauge("", "")

GPU_UTILIZATION_PCT = Gauge(
    "vap_gpu_utilization_pct",
    "GPU compute utilization percentage",
    ["device"],
) if PROMETHEUS_AVAILABLE else Gauge("", "")

# Worker errors
WORKER_ERRORS = Counter(
    "vap_worker_errors_total",
    "Unhandled errors in worker processing loops",
    ["worker_type", "worker_id"],
) if PROMETHEUS_AVAILABLE else Counter("", "")


# ── GPU stats collector ───────────────────────────────────────────────────────

class GPUStatsCollector(threading.Thread):
    """
    Background thread that periodically polls NVIDIA GPU stats
    and updates Prometheus gauges.
    """

    def __init__(self, interval: float = 10.0):
        super().__init__(daemon=True, name="gpu-stats-collector")
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        device_name = f"cuda:{i}"
                        mem_used = torch.cuda.memory_allocated(i) / (1024 * 1024)
                        GPU_MEMORY_USED_MB.labels(device=device_name).set(mem_used)

                        # pynvml for utilization
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            GPU_UTILIZATION_PCT.labels(device=device_name).set(util.gpu)
                        except ImportError:
                            pass  # pynvml optional
                        except Exception:
                            pass
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ── Redis stream health poller ────────────────────────────────────────────────

class RedisHealthPoller(threading.Thread):
    """
    Polls Redis stream lengths and consumer group lag every 15s,
    updates Prometheus gauges.
    """

    def __init__(self, redis_client, streams_cfg: dict, interval: float = 15.0):
        super().__init__(daemon=True, name="redis-health-poller")
        self.redis = redis_client
        self.streams = streams_cfg
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        from pipeline.utils.redis_utils import stream_stats
        while not self._stop.is_set():
            for stream_name, stream_key in self.streams.items():
                try:
                    stats = stream_stats(self.redis, stream_key)
                    REDIS_STREAM_LENGTH.labels(stream_name=stream_name).set(
                        stats.get("length", 0)
                    )
                    for group in stats.get("groups", []):
                        REDIS_STREAM_LAG.labels(
                            stream_name=stream_name,
                            group_name=group.get("name", "unknown"),
                        ).set(group.get("lag", 0))
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ── Milvus entity count poller ────────────────────────────────────────────────

class MilvusCountPoller(threading.Thread):
    """Polls Milvus entity count periodically."""

    def __init__(self, collection, interval: float = 60.0):
        super().__init__(daemon=True, name="milvus-count-poller")
        self.col = collection
        self.interval = interval
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                MILVUS_ENTITIES.set(self.col.num_entities)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


# ── Unified metrics manager ───────────────────────────────────────────────────

class PipelineMetrics:
    """
    Convenience class — instantiate once per worker, start background threads.

    Example:
        metrics = PipelineMetrics(worker_type="attribute", worker_id="01")
        metrics.start(port=9101, redis_client=redis, streams_cfg=cfg["redis"]["streams"])
    """

    def __init__(self, worker_type: str, worker_id: str):
        self.worker_type = worker_type
        self.worker_id = worker_id
        self._threads = []
        self._port = None
        self.available = PROMETHEUS_AVAILABLE

        if not PROMETHEUS_AVAILABLE:
            log.warning(
                "prometheus_client not installed — metrics disabled. "
                "Run: pip install prometheus-client"
            )

    def start(
        self,
        port: int = 9100,
        redis_client=None,
        streams_cfg: Optional[dict] = None,
        milvus_col=None,
    ):
        """Starts the metrics HTTP server and background pollers."""
        if not PROMETHEUS_AVAILABLE:
            return

        self._port = port
        try:
            start_http_server(port)
            log.info(f"Prometheus metrics server started on port {port}")
        except OSError as e:
            log.warning(f"Could not start metrics server on port {port}: {e}")

        # GPU stats
        gpu_collector = GPUStatsCollector()
        gpu_collector.start()
        self._threads.append(gpu_collector)

        # Redis health
        if redis_client and streams_cfg:
            redis_poller = RedisHealthPoller(redis_client, streams_cfg)
            redis_poller.start()
            self._threads.append(redis_poller)

        # Milvus count
        if milvus_col:
            milvus_poller = MilvusCountPoller(milvus_col)
            milvus_poller.start()
            self._threads.append(milvus_poller)

        log.info(
            f"PipelineMetrics started for worker_type={self.worker_type} "
            f"worker_id={self.worker_id} port={port}"
        )

    def stop(self):
        for t in self._threads:
            if hasattr(t, "stop"):
                t.stop()

    # ── Shortcut methods for common recording patterns ──────────────────────

    def record_detection(self, camera_id: str):
        DETECTIONS_TOTAL.labels(camera_id=camera_id).inc()

    def record_frame(self, camera_id: str):
        FRAMES_INGESTED.labels(camera_id=camera_id).inc()

    def record_worker_error(self):
        WORKER_ERRORS.labels(
            worker_type=self.worker_type,
            worker_id=self.worker_id,
        ).inc()

    def record_par_batch(self, batch_size: int, backend: str, elapsed_s: float):
        PAR_RUNS_TOTAL.labels(backend=backend).inc()
        PAR_CROPS_TOTAL.inc(batch_size)
        PAR_LATENCY.observe(elapsed_s)

    def record_reid_batch(self, batch_size: int, elapsed_s: float):
        REID_LATENCY.observe(elapsed_s)
        REID_BATCH_SIZE.observe(batch_size)

    def record_milvus_insert(self, count: int, elapsed_s: float):
        MILVUS_INSERTS_TOTAL.inc(count)
        MILVUS_INSERT_LATENCY.observe(elapsed_s)

    def record_alert(self, query_id: str):
        ALERTS_FIRED.labels(query_id=query_id).inc()

    def record_search(self, query_type: str, result_count: int, elapsed_s: float):
        SEARCH_REQUESTS.labels(query_type=query_type).inc()
        SEARCH_LATENCY.labels(query_type=query_type).observe(elapsed_s)
        SEARCH_RESULTS.labels(query_type=query_type).observe(result_count)
