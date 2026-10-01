# Architecture — Vision Attribute Pipeline

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  50+ IP Cameras (RTSP) / Video Files                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │  Frame bytes (JPEG compressed)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  camera_ingestor.py                                                 │
│  - One thread per camera                                            │
│  - Samples every Nth frame (frame_skip=5 → 5fps at 25fps input)     │
│  - Resizes to max 1280px wide                                       │
│  - JPEG-compresses (85% quality)                                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │  stream:raw_frames (Redis Stream)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  detector_worker.py                                                 │
│  - YOLOv8x FP16 @ 640px → person bboxes                            │
│  - BoT-SORT tracking → stable track IDs per camera                 │
│  - Per-camera tracker state (confirm after 3 frames)                │
│  - Crops person bboxes (with 15% padding)                           │
│  - Marks crops: needs_attribute=True (refresh every 30 frames)      │
└────────────────────────────┬────────────────────────────────────────┘
                             │  stream:person_crops (Redis Stream)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  attribute_worker.py                                                │
│  - Batches up to 16 crops (50ms timeout)                            │
│  - OSNet-AIN x1.0 → 512-dim L2-normed ReID embedding               │
│  - PromptPAR (or CLIP) → 11 attribute categories                    │
│  - Redis embedding cache: skip re-embedding same track (TTL=30s)    │
│  - Saves JPEG thumbnail to /tmp/thumbnails/{detection_id}.jpg       │
└────────────────────────────┬────────────────────────────────────────┘
                             │  stream:enriched_detections (Redis Stream)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  storage_worker.py                                                  │
│  - Deduplicates: skip same track_id within 5s window                │
│  - Batch writes to Milvus (50 records/batch, 1s timeout)            │
│  - Matches against active live queries (in-memory cache + Redis)    │
│  - Publishes alerts to Redis pub/sub: alerts:{query_id}             │
└────────────────────────────┬────────────────────────────────────────┘
                             │  HNSW index in Milvus (512-dim vectors)
                             │  + scalar indexes (camera_id, timestamp)
                             │  + 7-day TTL auto-expiry
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  api_server.py  (FastAPI + WebSocket)                               │
│  - POST /search/attributes  → Milvus scalar query                   │
│  - POST /search/image       → ANN vector search                     │
│  - POST /search/hybrid      → vector + scalar combined              │
│  - POST /live-queries       → register alert rule in Redis          │
│  - WS /ws/alerts/{query_id} → subscribe to real-time matches        │
│  - GET /                    → Web UI                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Redis Streams

Each stage communicates through Redis Streams with consumer groups:

| Stream | Producer | Consumer | Payload |
|--------|----------|----------|---------|
| `stream:raw_frames` | camera_ingestor | detector_worker | camera_id, frame_id, timestamp, JPEG bytes, dimensions |
| `stream:person_crops` | detector_worker | attribute_worker | track_id, bbox, confidence, crop JPEG, needs_attribute flag |
| `stream:enriched_detections` | attribute_worker | storage_worker | + reid_embed (512d), attributes dict, thumbnail_path |

**Backpressure**: Each stream is capped at `max_len=5000` entries. If a downstream worker falls behind, the producer silently drops the oldest frames (MAXLEN APPROX). This is intentional — old unprocessed frames have diminishing value.

**Dead-message recovery**: `stream_consume()` in `redis_utils.py` automatically reclaims messages pending >30s from dead consumers, preventing queue pile-ups on worker crash.

### Milvus Schema

```
Collection: person_detections
  detection_id    VARCHAR(64)   PK     UUID of this detection event
  track_id        VARCHAR(64)          "cam01:42" — camera-scoped tracker ID
  camera_id       VARCHAR(64)          Index: Trie
  timestamp       INT64                Unix ms; Index: STL_SORT
  reid_embed      FLOAT_VECTOR(512)    Index: HNSW (M=32, efConstruction=200)
  attributes      VARCHAR(4096)        JSON blob of attr dict
  thumbnail_path  VARCHAR(512)         Path to saved JPEG crop
  bbox_json       VARCHAR(128)         JSON [x1, y1, x2, y2]
  confidence      FLOAT                YOLOv8 detection confidence

TTL: 604800 seconds (7 days)  — automatic Milvus collection TTL
```

**Query strategy**:
- Attribute-only queries → Milvus `collection.query()` with scalar filter on `attributes` JSON string
- Image similarity → HNSW ANN search on `reid_embed` with optional scalar filter
- Hybrid → ANN with scalar filter expression (`expr=` parameter)

### PAR Model: PromptPAR vs CLIP

The `PARModel` class tries PromptPAR first, then CLIP ViT-L/14 as fallback.

**PromptPAR** (primary):
- Architecture: Swin-B backbone + learnable text prompts
- Training: PA100K (94,090 images, 26 binary attributes)
- Benchmark: 91.4% mean accuracy on PA100K (vs ~89% for older methods)
- Inference: ~120 crops/s at batch=16 on T4

**CLIP ViT-L/14** (fallback):
- Zero-shot via prompt ensembles per attribute
- No pretrained weights file needed — downloads automatically
- ~80–85% accuracy (lower than PromptPAR, but works immediately)
- Inference: ~60 crops/s at batch=16 (single-image loop for CLIP)

### OSNet-AIN ReID

- 512-dimensional embeddings, L2-normalised for cosine similarity via inner product
- Trained on Market1501 + DukeMTMC-reID combined (180K images, 1,812 identities)
- The embedding enables both deduplication across cameras and similarity search
- `embed_cache_ttl=30s` in Redis prevents re-embedding the same track ID in burst scenarios

### Attribute-to-Milvus Mapping

Attributes are stored as a JSON string in a single `VARCHAR(4096)` field. Queries use `like '%"glasses": "on"%'` style matching. This trades schema flexibility for query speed — for deployments exceeding 50M daily detections, consider migrating to a dedicated scalar field per attribute.

---

## Latency Budget

```
Frame ingestion + queue:     2–5ms
YOLOv8x inference (FP16):   15–25ms
BoT-SORT tracking overhead:  3–5ms
Crop extraction + encode:    1–2ms
Redis publish/consume:       2–4ms  (per hop)
OSNet ReID (batch=32):       8–15ms total → ~0.5ms/crop
PromptPAR (batch=16):        30–50ms total → ~2–3ms/crop
Milvus insert (batch=50):    5–10ms
──────────────────────────────────────
End-to-end detection→stored: 80–150ms

Milvus HNSW query:           1–3ms
API request overhead:        2–5ms
──────────────────────────────────────
Query latency (P99):         <10ms
```

### Real-time alert path

```
detection arrives → storage_worker.match() → Redis PUBLISH → WS listener → client
Total:  <50ms from detection to browser alert
```

---

## Cost Optimisation

For 50 cameras at 25fps with `frame_skip=5`:
- Effective rate: 5fps × 50 cameras = **250 frames/second** into the pipeline
- With `min_bbox_area` filtering and `track_confirm_frames=3`, only ~40% of frames generate crops
- With `attribute_refresh_frames=30`, PAR only runs on ~1/30 of confirmed crops
- Net PAR load: ≈ 250 × 0.4 / 30 ≈ **3 PAR inferences/second** — very manageable

**Milvus storage estimate** (7-day retention):
- 3 PAR inferences/s × 86400s/day × 7 days = ~1.8M vectors
- Each vector: 512 × 4 bytes + metadata ≈ 3KB → ~5.4GB total — well within single-node Milvus

**Redis memory** (`max_len=5000` per stream, 3 streams):
- Peak: 15,000 messages × ~10KB each ≈ 150MB — negligible

**Thumbnail storage** (`thumbnail_dir`, 24h max-age):
- 3 saves/s × 86400s × ~30KB JPEG ≈ 7.8GB/day — use EBS gp3 for this directory
