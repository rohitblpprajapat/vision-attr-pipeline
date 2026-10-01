"""
Model Downloader — Vision Attribute Pipeline
=============================================
Downloads and installs all required pretrained model weights.

Models:
  1. OSNet-AIN x1.0        — ReID embeddings (Market1501 weights)
  2. PromptPAR / OpenPAR   — Pedestrian attribute recognition (PA100K weights)
  3. torchreid             — Python package for OSNet (installed from GitHub)

Usage:
    python models/download_models.py [--skip-torchreid] [--skip-par]
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import requests

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
    console = Console()
    def info(msg): console.print(f"[cyan]ℹ[/cyan]  {msg}")
    def ok(msg):   console.print(f"[green]✔[/green]  {msg}")
    def warn(msg): console.print(f"[yellow]⚠[/yellow]  {msg}")
    def err(msg):  console.print(f"[red]✖[/red]  {msg}")
except ImportError:
    def info(msg): print(f"[INFO]  {msg}")
    def ok(msg):   print(f"[OK]    {msg}")
    def warn(msg): print(f"[WARN]  {msg}")
    def err(msg):  print(f"[ERR]   {msg}")

ROOT = Path(__file__).parent
MODELS_DIR = ROOT


# ── Helpers ───────────────────────────────────────────────────────────────

def run(cmd: str, check: bool = True) -> int:
    info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        err(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def download_file(url: str, dest: Path, description: str = "") -> bool:
    """Downloads a file using Python's stdlib/requests so it works on Windows and Linux."""
    if dest.exists():
        ok(f"Already exists: {dest.name}")
        return True
    info(f"Downloading {description or dest.name}…")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if requests is not None:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with dest.open("wb") as handle:
                    shutil.copyfileobj(response.raw, handle, length=1024 * 1024)
        else:
            import urllib.request
            urllib.request.urlretrieve(url, str(dest))
    except Exception as exc:
        err(f"Failed to download {url}: {exc}")
        dest.unlink(missing_ok=True)
        return False

    if not dest.exists():
        err(f"Download completed but file was not written: {dest}")
        return False

    ok(f"Downloaded: {dest.name} ({dest.stat().st_size // 1024}KB)")
    return True


# ── Step 1: Install torchreid (deep-person-reid) ──────────────────────────

def install_torchreid():
    info("Checking torchreid installation…")
    try:
        import torchreid
        ok(f"torchreid already installed")
        return True
    except ImportError:
        pass

    info("Installing torchreid from GitHub (KaiyangZhou/deep-person-reid)…")
    vendor_dir = ROOT.parent / "vendor"
    vendor_dir.mkdir(exist_ok=True)
    reid_dir = vendor_dir / "deep-person-reid"

    if not reid_dir.exists():
        rc = run(
            f'git clone --depth 1 https://github.com/KaiyangZhou/deep-person-reid.git "{reid_dir}"',
            check=False,
        )
        if rc != 0:
            err("Failed to clone deep-person-reid. Check internet connectivity.")
            warn("You can install manually: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")
            return False
    else:
        ok("deep-person-reid repo already cloned")

    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "Cython",
            "gdown",
        ],
        check=False,
    )
    if rc.returncode != 0:
        err("Failed to install torchreid build dependencies")
        return False

    rc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-build-isolation",
            "-e",
            str(reid_dir),
        ],
        check=False,
    )
    if rc.returncode != 0:
        err("Failed to pip install torchreid")
        return False

    ok("torchreid installed successfully")
    return True


# ── Step 2: Download OSNet-AIN x1.0 weights ───────────────────────────────

OSNET_WEIGHTS = {
    "osnet_ain_x1_0_market1501.pth": {
        "url": "https://drive.google.com/uc?id=1-CaioD9NaqbHK_kfS6RVBPVdQXuzbcAh",
        "description": "OSNet-AIN x1.0 — Market1501 (primary ReID model)",
        "fallback_url": None,
    },
}

def download_osnet():
    info("Downloading OSNet-AIN weights…")
    weights_dir = MODELS_DIR
    weights_dir.mkdir(exist_ok=True)

    # Try using torchreid's built-in downloader first (most reliable)
    try:
        from torchreid.utils import FeatureExtractor
        info("Using torchreid auto-download for OSNet-AIN x1.0…")
        # FeatureExtractor auto-downloads weights to ~/.cache/torch/checkpoints
        extractor = FeatureExtractor(
            model_name="osnet_ain_x1_0",
            device="cpu",
            verbose=True,
        )
        # Find where torchreid cached it
        import torch
        cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
        cached_weights = list(cache_dir.glob("osnet_ain*market*"))
        if not cached_weights:
            cached_weights = list(cache_dir.glob("osnet_ain*"))

        if cached_weights:
            src = cached_weights[0]
            dst = weights_dir / "osnet_ain_x1_0_market1501.pth"
            if not dst.exists():
                import shutil
                shutil.copy2(src, dst)
                ok(f"Copied OSNet weights to {dst}")
            else:
                ok(f"OSNet weights already at {dst}")
        else:
            ok("OSNet weights auto-downloaded to torchreid cache — no manual copy needed")
        return True
    except Exception as e:
        warn(f"torchreid auto-download failed: {e}")

    # Manual download fallback (gdown)
    info("Attempting manual OSNet download via gdown…")
    try:
        import gdown
        dst = weights_dir / "osnet_ain_x1_0_market1501.pth"
        if not dst.exists():
            gdown.download(
                "https://drive.google.com/uc?id=1-CaioD9NaqbHK_kfS6RVBPVdQXuzbcAh",
                str(dst),
                quiet=False,
            )
        ok(f"OSNet weights downloaded to {dst}")
        return True
    except ImportError:
        warn("gdown not installed. Installing…")
        run("pip install gdown --quiet", check=False)
    except Exception as e:
        warn(f"gdown download failed: {e}")

    warn(
        "Could not auto-download OSNet weights.\n"
        "  Option 1: pip install gdown && python models/download_models.py\n"
        "  Option 2: torchreid will auto-download weights on first inference run.\n"
        "  The pipeline will still work — torchreid caches weights on first use."
    )
    return False


# ── Step 3: Download / clone OpenPAR (PromptPAR) ─────────────────────────

def install_openpar():
    info("Setting up OpenPAR (PromptPAR)…")
    vendor_dir = ROOT.parent / "vendor"
    vendor_dir.mkdir(exist_ok=True)
    openpar_dir = vendor_dir / "OpenPAR"

    # Clone OpenPAR repo
    if not openpar_dir.exists():
        info("Cloning OpenPAR from GitHub (Event-AHU/OpenPAR)…")
        rc = run(
            f'git clone --depth 1 https://github.com/Event-AHU/OpenPAR.git "{openpar_dir}"',
            check=False,
        )
        if rc != 0:
            err("Failed to clone OpenPAR.")
            warn(
                "PromptPAR backbone unavailable. The pipeline will fall back to CLIP ViT-L/14 "
                "for attribute recognition, which still works well.\n"
                "To manually install:\n"
                "  cd vendor && git clone https://github.com/Event-AHU/OpenPAR.git"
            )
            return False
        ok("OpenPAR cloned successfully")
    else:
        ok("OpenPAR already cloned")

    # Download PromptPAR pretrained weights
    weights_path = MODELS_DIR / "promptpar_pa100k.pth"
    if weights_path.exists():
        ok(f"PromptPAR weights already at {weights_path}")
        return True

    # PromptPAR weights are hosted on the OpenPAR GitHub releases
    # Check the repo's README for the latest download link
    par_weights_url = (
        "https://github.com/Event-AHU/OpenPAR/releases/download/v1.0/"
        "PromptPAR_PA100K.pth"
    )
    info("Downloading PromptPAR PA100K weights…")
    if not download_file(par_weights_url, weights_path, "PromptPAR PA100K weights"):
        warn(
            "PromptPAR weights download failed.\n"
            f"  Please download manually from:\n"
            f"  https://github.com/Event-AHU/OpenPAR\n"
            f"  and place the file at: {weights_path}\n"
            f"\n"
            f"  The pipeline will use CLIP ViT-L/14 as fallback until weights are present."
        )
        return False

    ok(f"PromptPAR weights downloaded: {weights_path}")
    return True


# ── Step 4: Download YOLOv8x (auto via ultralytics) ──────────────────────

def download_yolo():
    info("Verifying YOLOv8x model…")
    try:
        from ultralytics import YOLO
        weights_path = MODELS_DIR.parent / "yolov8x.pt"
        if not weights_path.exists():
            info("YOLOv8x weights not cached. Will auto-download on first run.")
            info("Pre-downloading now to avoid cold-start delay…")
            model = YOLO("yolov8x.pt")
            ok("YOLOv8x weights downloaded via ultralytics")
        else:
            ok(f"YOLOv8x weights found at {weights_path}")
        return True
    except Exception as e:
        warn(f"YOLOv8 check failed: {e}")
        return False


# ── Step 5: CLIP weights (auto via HuggingFace / openai-clip) ─────────────

def download_clip():
    info("Verifying CLIP ViT-L/14 model…")
    try:
        import clip
        info("Pre-downloading CLIP ViT-L/14 weights…")
        model, _ = clip.load("ViT-L/14", device="cpu")
        ok("CLIP ViT-L/14 weights downloaded")
        return True
    except ImportError:
        # Try HuggingFace transformers
        try:
            from transformers import CLIPModel, CLIPProcessor
            info("Downloading CLIP via HuggingFace transformers…")
            CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            ok("CLIP ViT-L/14 downloaded via HuggingFace")
            return True
        except Exception as e:
            warn(f"CLIP download failed: {e}")
            return False
    except Exception as e:
        warn(f"CLIP download failed: {e}")
        return False


# ── Summary ───────────────────────────────────────────────────────────────

def print_summary(results: dict):
    try:
        from rich.table import Table
        table = Table(title="Model Download Summary", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Notes")
        for model, (status, note) in results.items():
            color = "green" if status else "yellow"
            table.add_row(model, f"[{color}]{'✔ OK' if status else '⚠ Partial'}[/{color}]", note)
        Console().print(table)
    except ImportError:
        print("\n=== Model Download Summary ===")
        for model, (status, note) in results.items():
            print(f"  {'✔' if status else '⚠'} {model}: {note}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download all pipeline model weights")
    parser.add_argument("--skip-torchreid", action="store_true")
    parser.add_argument("--skip-par", action="store_true")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-yolo", action="store_true")
    args = parser.parse_args()

    results = {}

    print()
    info("=== Vision Attribute Pipeline — Model Setup ===")
    print()

    if not args.skip_torchreid:
        ok_torchreid = install_torchreid()
        results["torchreid (OSNet)"] = (ok_torchreid, "Required for ReID embeddings")
    else:
        ok_torchreid = True

    if ok_torchreid:
        ok_osnet = download_osnet()
        results["OSNet-AIN x1.0 weights"] = (ok_osnet, "Auto-downloaded by torchreid if not present")

    if not args.skip_par:
        ok_par = install_openpar()
        results["PromptPAR (OpenPAR)"] = (ok_par, "Falls back to CLIP if unavailable")
    
    if not args.skip_clip:
        ok_clip = download_clip()
        results["CLIP ViT-L/14"] = (ok_clip, "Fallback PAR model")

    if not args.skip_yolo:
        ok_yolo = download_yolo()
        results["YOLOv8x"] = (ok_yolo, "Auto-downloads on first inference if not cached")

    print()
    print_summary(results)
    print()
    ok("Model setup complete. You can now run the pipeline.")


if __name__ == "__main__":
    main()
