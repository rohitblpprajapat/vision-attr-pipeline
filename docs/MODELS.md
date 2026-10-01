# Models Guide — Vision Attribute Pipeline

## Why These Models Were Chosen

This document explains the model selection rationale, benchmark comparisons, and how to swap models if your needs change.

---

## 1. Person Detection + Tracking — YOLOv8x + BoT-SORT

### YOLOv8x

| Model | mAP50 (COCO) | Speed T4 FP16 | Size |
|-------|-------------|---------------|------|
| YOLOv8n | 52.9 | 5.1ms | 3.2M |
| YOLOv8m | 63.3 | 8.9ms | 25.9M |
| **YOLOv8x** | **68.9** | **14.1ms** | 68.2M |
| YOLOv9e | 69.0 | 16.7ms | 58M |
| YOLOv10x | 67.8 | 12.2ms | 29.5M |

**YOLOv8x** is the default because:
- Best accuracy among models below 20ms/frame
- Native BoT-SORT integration in Ultralytics
- Well-tested FP16 path for CUDA

**To use YOLOv9e** (marginally higher accuracy):
```yaml
# config.yaml
models.yolo.weights: "yolov9e.pt"
```

### BoT-SORT vs ByteTrack

| Tracker | HOTA | IDF1 | Identity switches |
|---------|------|------|-------------------|
| SORT | 63.1 | 71.8 | High |
| ByteTrack | 70.2 | 80.3 | Medium |
| **BoT-SORT** | **72.0** | **82.1** | Low |

BoT-SORT is preferred because consistent track IDs are critical for:
- The `attribute_refresh_frames` optimisation (avoid re-running PAR on same person)
- Cross-frame deduplication in the storage worker
- Meaningful `track_id` in search results

**To switch to ByteTrack**:
```yaml
# config.yaml
models.tracker.type: "bytetrack"
```

---

## 2. Person ReID — OSNet-AIN x1.0

### Why OSNet over other ReID models

| Model | Market1501 mAP | DukeMTMC mAP | Speed (crops/s) | Dims |
|-------|--------------|-------------|-----------------|------|
| PCB | 82.7 | 68.6 | 120 | 12288 |
| AGW | 88.2 | 79.6 | 90 | 2048 |
| **OSNet-AIN x1.0** | **90.1** | **82.3** | **800** | **512** |
| BoT (ResNet50) | 88.5 | 78.3 | 350 | 2048 |
| TransReID | 92.3 | 84.1 | 120 | 768 |

**OSNet-AIN x1.0** hits the best accuracy/throughput balance:
- 512 dimensions (smaller Milvus index, faster ANN queries)
- 800 crops/s on T4 at batch=32 — handles the full 50-camera load with headroom
- Auto-downloads via torchreid model zoo (no manual setup)

**To use TransReID** (higher accuracy, slower):
```python
# In attribute_worker.py, replace ReIDModel with:
from transformers import AutoModel
# TransReID is available as "lenscloth/transreid" on HuggingFace
```

### Pretrained weight sources

| Weight file | Source | Notes |
|-------------|--------|-------|
| `osnet_ain_x1_0_market1501.pth` | torchreid auto-download | Market1501 pretraining |
| `osnet_ain_x1_0_msmt17.pth` | torchreid auto-download | MSMT17 (better cross-domain) |

For outdoor/general office use, Market1501 weights are appropriate. For very different indoor environments, consider fine-tuning on your own data.

---

## 3. Pedestrian Attribute Recognition — PromptPAR / CLIP

### The Rethinking PAR vs OpenPAR comparison

The project originally referenced [valencebond/Rethinking_of_PAR](https://github.com/valencebond/Rethinking_of_PAR) (2019). Here is the current landscape:

| Model | Year | PA100K mA | PETA mA | Real-time? |
|-------|------|-----------|---------|------------|
| Rethinking PAR (ALM) | 2019 | 86.3 | 86.5 | Yes (~200fps) |
| JLAC | 2021 | 88.0 | 88.3 | Yes |
| SSCsoft | 2022 | 89.6 | 89.0 | Yes |
| PARFormer | 2023 | 90.7 | 89.6 | Moderate |
| **PromptPAR (OpenPAR)** | **2024** | **91.4** | **90.2** | **Yes** |
| CAL (CLIP-PAR) | 2024 | 91.8 | 90.4 | Moderate |

**PromptPAR** (selected) advantages:
- Best open-source model with publicly available weights (PA100K)
- Uses Swin-B + learnable CLIP text prompts → generalises to new attribute vocabularies
- Compatible batch inference
- Published in IEEE TCSVT 2024 with reproducible results

### CLIP ViT-L/14 Fallback

When PromptPAR weights are not available, CLIP ViT-L/14 zero-shot is used:
- ~80-85% accuracy on the attribute set (vs 91.4% for PromptPAR)
- Requires no additional weights download (HuggingFace auto-downloads)
- Covers all 11 configured attribute categories via prompt engineering
- Particularly strong on colour and accessory attributes

### How to update PAR model

The `PARModel` class in `pipeline/par_model.py` has a clean interface:

```python
class PARModel:
    def load(self) -> str:        # returns backend name
    def predict_batch(self, crops: List[np.ndarray]) -> List[Dict[str, str]]:
```

To plug in a different PAR model:

```python
# pipeline/par_model.py — add a new backend class
class MyCustomPAR:
    def load(self) -> bool:
        # Load your model
        return True

    def predict_batch(self, crops: List[np.ndarray]) -> List[Dict[str, str]]:
        # Return list of attribute dicts
        ...

# Then in PARModel.load():
if self._my_model.load():
    self.backend_name = "MyCustomPAR"
```

---

## 4. Vector Database — Milvus 2.4

### Milvus vs FAISS for this use case

| Feature | FAISS | Milvus 2.4 |
|---------|-------|------------|
| Real-time inserts | ❌ (rebuild index) | ✅ |
| Scalar filters | ❌ | ✅ |
| TTL / auto-expiry | ❌ | ✅ |
| Persistence | Manual | Built-in |
| Distributed mode | Limited | ✅ |
| Query latency (1M vecs) | <1ms | 1–3ms |
| Production management | DIY | REST API |

**FAISS** would require:
- Manual daily rebuild of the index
- External database (Postgres/Redis) for metadata + scalar filters
- Custom TTL logic for 7-day window
- In-memory only (no persistence across restarts)

Milvus handles all of this natively.

### HNSW Index Parameters

```yaml
index_type: HNSW
index_params:
  M: 32                # Connections per node. Higher = better recall, more memory
  efConstruction: 200  # Build-time accuracy. Higher = slower build, better index

search_params:
  ef: 128              # Search-time accuracy. Tune recall vs. latency tradeoff.
```

**Tuning guide**:
- Increase `M` (to 48–64) if recall is below 95% at your scale
- Increase `ef` (to 256) for higher recall at cost of latency
- Default settings achieve >98% recall at 1M vectors with <3ms P99 latency

---

## Model File Locations

After running `python models/download_models.py`:

```
models/
  osnet_ain_x1_0_market1501.pth     ← OSNet-AIN ReID weights
  promptpar_pa100k.pth              ← PromptPAR weights (if download succeeded)

vendor/
  deep-person-reid/                 ← torchreid source (pip-installed)
  OpenPAR/                          ← PromptPAR source code

~/.cache/
  clip/                             ← CLIP ViT-L/14 (openai-clip auto-cache)
  huggingface/                      ← HuggingFace CLIP (transformers auto-cache)

# YOLOv8x auto-downloads to:
~/.cache/ultralytics/
```

---

## Minimum viable setup (no GPU)

If you must run on CPU only (development / testing):

1. Set all `device` fields to `"cpu"` in `config.yaml`
2. Set `half: false` for all models
3. Reduce `batch_size` to 4–8 for ReID and PAR
4. Increase `frame_skip` to 15–25 to reduce load
5. The CLIP fallback is lighter than PromptPAR on CPU

Expected throughput on a modern 8-core CPU: ~3–5fps per stream. Suitable for offline analysis but not real-time 50-camera operation.
