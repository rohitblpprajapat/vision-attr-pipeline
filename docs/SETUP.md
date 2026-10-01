# Setup Guide — Vision Attribute Pipeline

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development)
3. [Model Downloads](#model-downloads)
4. [Running the Pipeline](#running-the-pipeline)
5. [AWS g4dn/g5 Deployment](#aws-deployment)
6. [Scaling Workers](#scaling-workers)
7. [Configuration Reference](#configuration-reference)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware

- **GPU**: NVIDIA T4 (g4dn) or A10G (g5) recommended. 8GB VRAM minimum.
- **RAM**: 16GB minimum (32GB recommended for 50+ cameras)
- **Storage**: 50GB for models + logs + thumbnails

### Software

- Docker 24+ with `docker compose` plugin
- NVIDIA Container Toolkit (for GPU passthrough to containers)
- Python 3.10+ (for local/non-Docker setup)
- CUDA 12.1+ driver

On Windows, make sure Docker Desktop is running and that the Linux engine context is available before starting the stack. A missing `dockerDesktopLinuxEngine` pipe means Docker is installed but not currently running.

### Verify GPU setup

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## Local Development

### 1. Clone and install

```bash
git clone <repo-url> vision-attr-pipeline
cd vision-attr-pipeline

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 12.1 first
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Start infrastructure (Redis + Milvus) via Docker

```bash
# Start only the infrastructure services
docker compose up redis etcd minio milvus -d

# Verify they are healthy
docker compose ps
# All should show "healthy" after ~60 seconds
```

### 3. Environment variables

```bash
cp .env.example .env
# Edit .env — minimum required:
#   REDIS_HOST=localhost
#   MILVUS_HOST=localhost
#   CAMERA_SOURCES=test:/path/to/your/video.mp4
```

Load them in your shell:

```bash
export $(cat .env | grep -v '^#' | xargs)
```

---

## Model Downloads

Run the model downloader once before starting the pipeline:

```bash
python models/download_models.py
```

This script:

1. Clones and installs **torchreid** (OSNet-AIN) from GitHub
2. Auto-downloads **OSNet-AIN x1.0** weights via torchreid's model zoo
3. Clones **OpenPAR** and downloads **PromptPAR** weights (PA100K)
4. Pre-downloads **CLIP ViT-L/14** (fallback PAR model)
5. Pre-downloads **YOLOv8x** weights via Ultralytics

Expected output:

```
✔  torchreid (OSNet)          ✔ OK
✔  OSNet-AIN x1.0 weights     ✔ OK
✔  PromptPAR (OpenPAR)        ✔ OK      ← if GitHub download succeeds
✔  CLIP ViT-L/14              ✔ OK      ← always available as fallback
✔  YOLOv8x                    ✔ OK
```

### If PromptPAR download fails

The pipeline will automatically use **CLIP ViT-L/14** as the PAR backend — this requires no manual setup. You only lose ~2% mA accuracy vs. PromptPAR. The CLIP fallback handles all 11 attribute categories.

To manually get PromptPAR weights:

```bash
cd vendor
git clone https://github.com/Event-AHU/OpenPAR.git
# Download weights from the OpenPAR releases page:
# https://github.com/Event-AHU/OpenPAR/releases
# Place the .pth file at: models/promptpar_pa100k.pth
```

---

## Running the Pipeline

### Test mode — single stream, no queues

Best for verifying model setup and GPU:

```bash
# Download a test video first
wget -O test_data/sample.mp4 \
  https://www.pexels.com/video/crowd-of-people-walking-852395/download

python scripts/test_single_stream.py \
    --source test_data/sample.mp4 \
    --config config/config.yaml \
    --output test_data/output_annotated.mp4 \
    --max-frames 300
```

This runs inline (no Redis/Milvus) and prints performance stats.

### Production mode — full distributed pipeline

Start each worker in a separate terminal (or let Docker Compose handle it):

**Terminal 1 — Camera Ingestor**

```bash
python -m pipeline.camera_ingestor \
    --config config/config.yaml \
    --cameras cam01:rtsp://192.168.1.10/stream1 \
              cam02:rtsp://192.168.1.11/stream1 \
              test:test_data/sample.mp4
```

**Terminal 2 — Detector Worker**

```bash
python -m pipeline.detector_worker \
    --config config/config.yaml \
    --worker-id 01
```

**Terminal 3 — Attribute Worker**

```bash
python -m pipeline.attribute_worker \
    --config config/config.yaml \
    --worker-id 01
```

**Terminal 4 — Storage Worker**

```bash
python -m pipeline.storage_worker \
    --config config/config.yaml \
    --worker-id 01
```

**Terminal 5 — API Server**

```bash
python -m api.api_server \
    --config config/config.yaml \
    --host 0.0.0.0 \
    --port 8000

# Open: http://localhost:8000
```

### Docker Compose (all at once)

```bash
# Set your cameras in .env
echo 'CAMERA_SOURCES=cam01:rtsp://192.168.1.10/stream1' >> .env

# Full stack
docker compose up -d

# Watch logs
docker compose logs -f attribute_worker

# Check health
curl http://localhost:8000/api/v1/health | python -m json.tool
```

---

## AWS Deployment

### Instance recommendations

| Cameras  | Instance                   | vRAM    | Notes                       |
| -------- | -------------------------- | ------- | --------------------------- |
| Up to 10 | g4dn.xlarge                | 16GB T4 | Single node, 2 workers each |
| 10–25    | g4dn.2xlarge               | 16GB T4 | 3 attribute workers         |
| 25–50+   | g4dn.4xlarge or g5.2xlarge | 24–48GB | Scale workers               |

### Step-by-step EC2 setup

```bash
# 1. Launch EC2 instance with Deep Learning AMI (Ubuntu 22.04)
#    AMI: "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)"
#    Attach 100GB gp3 EBS volume

# 2. SSH in and verify GPU
ssh -i your-key.pem ubuntu@<ec2-ip>
nvidia-smi   # Should show T4 or A10G

# 3. Install Docker GPU support (if not on Deep Learning AMI)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 4. Clone and build
git clone <your-repo> ~/vision-attr-pipeline
cd ~/vision-attr-pipeline
cp .env.example .env
# Edit .env with your RTSP camera URLs and Redis password

# 5. Build Docker image
docker compose build

# 6. Download models (on host, before starting containers)
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python models/download_models.py

# 7. Start full stack
docker compose up -d

# 8. Open API (set your security group to allow port 8000)
curl http://localhost:8000/api/v1/health
```

### Security group ports

| Port  | Service                  |
| ----- | ------------------------ |
| 8000  | API + Web UI (public)    |
| 6379  | Redis (internal only)    |
| 19530 | Milvus (internal only)   |
| 9091  | Milvus health (internal) |

### Persistent data (EBS)

```bash
# Mount EBS for thumbnails and logs
sudo mkfs -t ext4 /dev/nvme1n1
sudo mount /dev/nvme1n1 /data
sudo mkdir -p /data/thumbnails /data/logs

# Update docker-compose.yml volumes to point to /data
# Or update config.yaml:
#   storage.thumbnail_dir: /data/thumbnails
#   logging.log_dir: /data/logs
```

---

## Scaling Workers

The Redis Stream consumer group automatically load-balances messages across workers. You can add workers at any time without restarting existing ones.

```bash
# Scale to 2 attribute workers and 2 storage workers
docker compose up --scale attribute_worker=2 --scale storage_worker=2 -d

# Scale detector workers for more cameras
docker compose up --scale detector_worker=2 -d
```

### Worker-to-camera ratio guidelines

| Cameras | detector_worker | attribute_worker | storage_worker |
| ------- | --------------- | ---------------- | -------------- |
| 1–10    | 1               | 1                | 1              |
| 10–25   | 1               | 2                | 1              |
| 25–50   | 2               | 2–3              | 2              |
| 50+     | 3               | 4                | 2              |

---

## Configuration Reference

All settings are in `config/config.yaml`. Key settings to tune:

```yaml
# Reduce this to process more frames (higher GPU load)
ingestion.frame_skip: 5 # Process every 5th frame (5fps at 25fps source)

# Increase this for faster attribute refresh (higher GPU load)
detection.attribute_refresh_frames: 30 # Re-run PAR every 30 processed frames

# Tune for your accuracy/performance tradeoff
models.yolo.conf_threshold: 0.40
models.par.confidence_threshold: 0.50

# Adjust for your storage budget
milvus.ttl_seconds: 604800 # 7 days
```

---

## Troubleshooting

### "Cannot open camera" errors

```bash
# Test RTSP connectivity
ffprobe rtsp://192.168.1.10/stream1
# or
vlc rtsp://192.168.1.10/stream1
```

### "Milvus not connecting"

```bash
docker compose logs milvus
# Common fix: wait longer for Milvus to start (can take 2-3 minutes on first run)
docker compose restart milvus
```

### GPU out of memory

```bash
# Check VRAM usage
nvidia-smi

# Reduce batch sizes in config.yaml:
#   models.reid.batch_size: 16   (from 32)
#   models.par.batch_size: 8    (from 16)
# Or enable CPU offload for PAR:
#   models.par.device: cpu
```

### "torchreid not installed"

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
# then re-run: python models/download_models.py --skip-par --skip-yolo
```

### No results in search

After starting the pipeline, allow 2–5 minutes for detections to populate Milvus. Verify data is flowing:

```bash
# Check stream lengths
redis-cli -h localhost XLEN stream:raw_frames
redis-cli -h localhost XLEN stream:enriched_detections

# Check Milvus entity count via API
curl http://localhost:8000/api/v1/health
```
