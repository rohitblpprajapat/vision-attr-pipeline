# API Reference — Vision Attribute Pipeline

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## Endpoints

### Health

#### `GET /api/v1/health`

Returns pipeline component status.

**Response**
```json
{
  "status": "ok",
  "timestamp": 1700000000000,
  "components": {
    "redis":   { "status": "ok" },
    "milvus":  { "status": "ok", "stats": { "num_entities": 142837 } },
    "stream_raw_frames":   { "length": 142, "groups": [{"name": "group:detector", "pending": 0, "lag": 0}] },
    "stream_enriched":     { "length": 89, ... }
  }
}
```

---

### Attribute Schema

#### `GET /api/v1/attributes/schema`

Returns all valid attribute keys and their permitted values.

**Response**
```json
{
  "action":       ["cycling", "running", "sitting", "standing", "walking"],
  "age_group":    ["adult", "old", "young"],
  "bag_type":     ["backpack", "handbag", "none", "shoulder_bag"],
  "gender":       ["female", "male"],
  "glasses":      ["off", "on"],
  "hat":          ["off", "on"],
  "hair_length":  ["bald", "long", "medium", "short"],
  "lower_color":  ["black", "blue", "brown", "gray", "green", "white"],
  "lower_type":   ["jeans", "leggings", "pants", "shorts", "skirt"],
  "upper_color":  ["black", "blue", "brown", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"],
  "upper_type":   ["coat", "dress", "hoodie", "jacket", "other", "shirt", "sweater", "tshirt", "vest"]
}
```

---

### Search

#### `POST /api/v1/search/attributes`

Search for detections matching attribute criteria. Returns results sorted by recency.

**Request body**
```json
{
  "attributes": {
    "upper_color": "white",
    "glasses": "on"
  },
  "camera_ids": ["cam01", "cam03"],
  "time_window_hours": 6,
  "limit": 100
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `attributes` | object | Yes | — | Key-value attribute pairs (AND logic) |
| `camera_ids` | string[] | No | all | Filter to specific cameras |
| `time_window_hours` | float | No | 24 | Search window (max 168 = 7 days) |
| `limit` | int | No | 100 | Max results (max 500) |

**Response**
```json
{
  "count": 23,
  "query": { "attributes": {"upper_color": "white", "glasses": "on"}, ... },
  "results": [
    {
      "detection_id":   "a4f8c3d2-...",
      "track_id":       "cam01:42",
      "camera_id":      "cam01",
      "timestamp":      1700000000000,
      "datetime_utc":   "2024-11-14T12:00:00.000Z",
      "score":          1.0,
      "confidence":     0.92,
      "attributes": {
        "gender":      "male",
        "age_group":   "adult",
        "upper_color": "white",
        "upper_type":  "shirt",
        "lower_color": "black",
        "lower_type":  "pants",
        "glasses":     "on",
        "hat":         "off",
        "bag_type":    "none",
        "action":      "walking"
      },
      "bbox":           [120.5, 85.2, 220.1, 410.8],
      "thumbnail_path": "/tmp/thumbnails/a4f8c3d2.jpg",
      "thumbnail_url":  "/api/v1/thumbnails/a4f8c3d2.jpg"
    }
  ]
}
```

**Error: 400 Bad Request** — invalid attribute key or value
```json
{ "detail": "Invalid value 'maybe' for 'glasses'. Valid: ['off', 'on']" }
```

---

#### `POST /api/v1/search/image`

Find people who look visually similar to the provided person crop image.

**Request body**
```json
{
  "image_base64": "<base64-encoded-jpeg>",
  "attribute_filter": { "gender": "male" },
  "camera_ids": null,
  "time_window_hours": 24,
  "top_k": 50,
  "similarity_threshold": 0.60
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `image_base64` | string | Yes | — | Base64 JPEG/PNG of a person crop |
| `attribute_filter` | object | No | — | Optional additional attribute constraints |
| `top_k` | int | No | 50 | Number of ANN results before threshold filter |
| `similarity_threshold` | float | No | 0.60 | Min cosine similarity (0–1) |

**Response** — same structure as `/search/attributes` but with `score` = cosine similarity (0–1).

---

#### `POST /api/v1/search/image-upload`

Same as `/search/image` but accepts a multipart file upload instead of base64.

**Request** — `multipart/form-data`
```
file=@person_crop.jpg
attribute_filter={"glasses": "on"}
time_window_hours=24
top_k=50
similarity_threshold=0.60
```

---

#### `POST /api/v1/search/hybrid`

Combined attribute filtering + embedding similarity. If both `image_base64` and `attributes` are provided, performs vector search with attribute pre-filter.

**Request body**
```json
{
  "image_base64": "<base64-jpeg>",
  "attributes": { "hat": "on", "gender": "male" },
  "camera_ids": ["cam01"],
  "time_window_hours": 12,
  "top_k": 50,
  "similarity_threshold": 0.55
}
```

---

### Live Queries

Live queries enable real-time alerting: register a query, then subscribe to a WebSocket stream to receive matching detections as they happen.

#### `POST /api/v1/live-queries` → 201 Created

Register a live attribute query.

**Request body**
```json
{
  "attributes": { "upper_color": "red", "hat": "on" },
  "camera_ids": ["cam01", "cam02"],
  "description": "Person in red with hat"
}
```

**Response**
```json
{
  "query_id": "b7e2a4f1-...",
  "ws_url":   "/ws/alerts/b7e2a4f1-...",
  "attributes": { "upper_color": "red", "hat": "on" },
  "camera_ids": ["cam01", "cam02"]
}
```

---

#### `GET /api/v1/live-queries`

List all active live queries.

**Response**
```json
{
  "count": 2,
  "queries": [
    {
      "query_id":    "b7e2a4f1-...",
      "attributes":  { "upper_color": "red", "hat": "on" },
      "camera_ids":  ["cam01", "cam02"],
      "description": "Person in red with hat",
      "created_at":  1700000000000
    }
  ]
}
```

---

#### `DELETE /api/v1/live-queries/{query_id}` → 200

Remove a live query (stops alerting).

---

### Thumbnails

#### `GET /api/v1/thumbnails/{filename}`

Serves a saved person crop thumbnail as JPEG.

The `thumbnail_url` field in search results points to this endpoint.

---

## WebSocket — Real-time Alerts

### `WS /ws/alerts/{query_id}`

Subscribe to real-time alerts for a registered live query.

**Connection sequence**
```
Client → connect ws://host/ws/alerts/{query_id}
Server → {"type": "subscribed", "query_id": "..."}
Server → {"type": "attribute_match", ...}  ← whenever a match occurs
Server → {"type": "attribute_match", ...}
```

**Alert message format**
```json
{
  "type":           "attribute_match",
  "query_id":       "b7e2a4f1-...",
  "detection_id":   "a4f8c3d2-...",
  "track_id":       "cam01:42",
  "camera_id":      "cam01",
  "timestamp":      1700000000000,
  "attributes": {
    "upper_color": "red",
    "hat":         "on",
    "gender":      "male"
  },
  "thumbnail_url":  "/api/v1/thumbnails/a4f8c3d2.jpg",
  "bbox":           [120.5, 85.2, 220.1, 410.8],
  "confidence":     0.91,
  "alert_ts":       1700000000050
}
```

**Alert cooldown**: The same `track_id` will not fire the same `query_id` alert again within 60 seconds (configurable via `alerts.cooldown_seconds`).

### `WS /ws/alerts/global`

Receives ALL alerts from ALL live queries. Useful for a global monitoring dashboard.

---

## Code Examples

### Python — attribute search
```python
import requests

resp = requests.post("http://localhost:8000/api/v1/search/attributes", json={
    "attributes": {"upper_color": "white", "glasses": "on"},
    "time_window_hours": 6,
})
data = resp.json()
print(f"Found {data['count']} matches")
for r in data["results"]:
    print(f"  {r['camera_id']} @ {r['datetime_utc']} — {r['attributes']}")
```

### Python — image similarity search
```python
import base64, requests
from pathlib import Path

img_b64 = base64.b64encode(Path("person_crop.jpg").read_bytes()).decode()
resp = requests.post("http://localhost:8000/api/v1/search/image", json={
    "image_base64": img_b64,
    "similarity_threshold": 0.65,
    "time_window_hours": 24,
})
for r in resp.json()["results"]:
    print(f"  score={r['score']:.3f} cam={r['camera_id']} track={r['track_id']}")
```

### JavaScript — live query + WebSocket
```javascript
// Register query
const resp = await fetch("/api/v1/live-queries", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    attributes: { upper_color: "red", hat: "on" },
    description: "Person in red hat"
  })
});
const { query_id } = await resp.json();

// Subscribe to alerts
const ws = new WebSocket(`ws://localhost:8000/ws/alerts/${query_id}`);
ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  if (alert.type === "attribute_match") {
    console.log(`Match! Camera: ${alert.camera_id}, Track: ${alert.track_id}`);
    console.log("Attributes:", alert.attributes);
  }
};
```

### curl — quick test
```bash
# Attribute search
curl -s -X POST http://localhost:8000/api/v1/search/attributes \
  -H "Content-Type: application/json" \
  -d '{"attributes": {"glasses": "on"}, "time_window_hours": 1}' \
  | python -m json.tool

# Live query registration
curl -s -X POST http://localhost:8000/api/v1/live-queries \
  -H "Content-Type: application/json" \
  -d '{"attributes": {"upper_color": "white"}, "description": "white shirt"}' \
  | python -m json.tool
```
