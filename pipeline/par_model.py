"""
PAR Model Wrapper — Vision Attribute Pipeline
==============================================
Wraps PromptPAR (OpenPAR, Event-AHU, IEEE TCSVT 2024) for pedestrian
attribute recognition. Falls back to a CLIP ViT-L/14 zero-shot approach
if PromptPAR weights are not available.

PromptPAR Reference:
  "Pedestrian Attribute Recognition via CLIP based Prompt Vision-Language Fusion"
  Xiao Wang et al., IEEE TCSVT 2024.
  GitHub: https://github.com/Event-AHU/OpenPAR
  Trained on PA100K (100K images, 26 binary attributes) and PETA (65 attrs)

Attributes recognized (standardized superset):
  - gender:       male | female
  - age_group:    young | adult | old
  - upper_color:  black | white | red | blue | green | yellow | gray | purple | brown | pink | orange
  - lower_color:  black | white | blue | gray | brown | green | other
  - upper_type:   tshirt | shirt | jacket | coat | dress | sweater | vest | hoodie
  - lower_type:   pants | jeans | shorts | skirt | leggings
  - hair_length:  short | medium | long | bald
  - glasses:      on | off
  - hat:          on | off
  - bag_type:     backpack | handbag | none | shoulder_bag
  - action:       walking | standing | running | sitting | cycling
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


# ---------------------------------------------------------------------------
# Attribute definitions (maps model output indices to human-readable labels)
# ---------------------------------------------------------------------------

# PA100K attribute set (26 binary attributes) — used by PromptPAR
PA100K_ATTRS = [
    "Female",
    "AgeOver60", "Age18-60", "AgeLess18",
    "Front", "Side", "Back",
    "Hat",
    "Glasses",
    "HandBag", "ShoulderBag", "Backpack",
    "HoldObjectsInFront",
    "ShortSleeve", "LongSleeve",
    "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern",
    "LongCoat",
    "Trousers", "Shorts", "Skirt&Dress",
    "boots",
]

# Mapping from PA100K binary outputs → our attribute schema
# Format: (pa100k_index, our_key, positive_label, negative_label)
PA100K_TO_SCHEMA = [
    (0,  "gender",      "female", "male"),
    (1,  "age_group",   "old",    None),
    (2,  "age_group",   "adult",  None),
    (3,  "age_group",   "young",  None),
    (7,  "hat",         "on",     "off"),
    (8,  "glasses",     "on",     "off"),
    (9,  "bag_type",    "handbag", None),
    (10, "bag_type",    "shoulder_bag", None),
    (11, "bag_type",    "backpack", None),
    (13, "upper_type",  "tshirt", None),
    (14, "upper_type",  "shirt",  None),   # long sleeve → shirt
    (21, "upper_type",  "coat",   None),
    (22, "lower_type",  "pants",  None),
    (23, "lower_type",  "shorts", None),
    (24, "lower_type",  "skirt",  None),
]

# CLIP-based zero-shot prompts for each attribute
CLIP_PROMPTS = {
    "gender": [
        ("male",   ["a photo of a man", "a photo of a male person"]),
        ("female", ["a photo of a woman", "a photo of a female person"]),
    ],
    "age_group": [
        ("young", ["a photo of a young person under 30", "a photo of a teenager"]),
        ("adult", ["a photo of an adult person aged 30-60"]),
        ("old",   ["a photo of an elderly person over 60", "a photo of an old person"]),
    ],
    "upper_color": [
        ("black",  ["a person wearing a black top"]),
        ("white",  ["a person wearing a white top", "a person in white shirt"]),
        ("red",    ["a person wearing a red top"]),
        ("blue",   ["a person wearing a blue top"]),
        ("green",  ["a person wearing a green top"]),
        ("yellow", ["a person wearing a yellow top"]),
        ("gray",   ["a person wearing a gray top"]),
        ("purple", ["a person wearing a purple top"]),
        ("brown",  ["a person wearing a brown top"]),
        ("pink",   ["a person wearing a pink top"]),
        ("orange", ["a person wearing an orange top"]),
    ],
    "lower_color": [
        ("black", ["a person wearing black pants or skirt"]),
        ("white", ["a person wearing white pants or skirt"]),
        ("blue",  ["a person wearing blue jeans or pants"]),
        ("gray",  ["a person wearing gray pants"]),
        ("brown", ["a person wearing brown pants"]),
        ("green", ["a person wearing green pants"]),
    ],
    "upper_type": [
        ("tshirt",   ["a person wearing a t-shirt"]),
        ("shirt",    ["a person wearing a long sleeve shirt"]),
        ("jacket",   ["a person wearing a jacket"]),
        ("coat",     ["a person wearing a long coat"]),
        ("dress",    ["a person wearing a dress"]),
        ("sweater",  ["a person wearing a sweater"]),
        ("hoodie",   ["a person wearing a hoodie"]),
    ],
    "lower_type": [
        ("pants",    ["a person wearing pants or trousers"]),
        ("jeans",    ["a person wearing jeans"]),
        ("shorts",   ["a person wearing shorts"]),
        ("skirt",    ["a person wearing a skirt"]),
        ("leggings", ["a person wearing leggings"]),
    ],
    "hair_length": [
        ("short",  ["a person with short hair"]),
        ("medium", ["a person with medium length hair"]),
        ("long",   ["a person with long hair"]),
        ("bald",   ["a bald person"]),
    ],
    "glasses": [
        ("on",  ["a person wearing glasses", "a person with spectacles"]),
        ("off", ["a person without glasses"]),
    ],
    "hat": [
        ("on",  ["a person wearing a hat or cap", "a person with a hat"]),
        ("off", ["a person without a hat"]),
    ],
    "bag_type": [
        ("backpack",     ["a person carrying a backpack on their back"]),
        ("handbag",      ["a person carrying a handbag"]),
        ("shoulder_bag", ["a person carrying a shoulder bag"]),
        ("none",         ["a person not carrying any bag"]),
    ],
    "action": [
        ("walking",  ["a person walking"]),
        ("standing", ["a person standing still"]),
        ("running",  ["a person running or jogging"]),
        ("sitting",  ["a person sitting down"]),
        ("cycling",  ["a person riding a bicycle"]),
    ],
}


# ---------------------------------------------------------------------------
# PromptPAR Model (OpenPAR)
# ---------------------------------------------------------------------------

class PromptPARWrapper:
    """
    Wraps the PromptPAR model from OpenPAR.
    Requires the OpenPAR repository to be cloned or pip-installed,
    and pretrained weights to be present.

    Weight download instructions:
        See models/download_models.py  or  docs/MODELS.md
    """

    def __init__(
        self,
        weights_path: str,
        backbone: str = "swin_b",
        device: str = "cuda:0",
        half: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.type == "cuda"
        self._model = None
        self.weights_path = weights_path
        self.backbone = backbone
        self._loaded = False

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load(self) -> bool:
        """
        Attempts to load PromptPAR model.
        Returns True on success, False if weights not available.
        """
        if not Path(self.weights_path).exists():
            return False
        try:
            # OpenPAR needs to be in the Python path
            # Attempt import — will fail gracefully if OpenPAR not installed
            import sys
            openpar_path = str(Path(__file__).parent.parent / "vendor" / "OpenPAR")
            if Path(openpar_path).exists() and openpar_path not in sys.path:
                sys.path.insert(0, openpar_path)

            from models.PromptPAR import PromptPAR
            self._model = PromptPAR(
                num_classes=26,
                backbone=self.backbone,
            )
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            # Handle DataParallel-wrapped checkpoints
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self._model.load_state_dict(state_dict, strict=False)
            self._model.to(self.device)
            self._model.eval()
            if self.half:
                self._model = self._model.half()
            self._loaded = True
            return True
        except ImportError:
            return False
        except Exception as e:
            return False

    @torch.no_grad()
    def predict_batch(self, crops: List[np.ndarray]) -> List[Dict[str, str]]:
        """
        Runs PromptPAR on a batch of BGR numpy crops.
        Returns list of attribute dicts.
        """
        if not self._loaded or self._model is None:
            return [{}] * len(crops)

        imgs = []
        for crop in crops:
            pil = Image.fromarray(crop[..., ::-1])   # BGR→RGB
            t = self.transform(pil)
            imgs.append(t)

        batch = torch.stack(imgs).to(self.device)
        if self.half:
            batch = batch.half()

        logits = self._model(batch)   # (B, 26)
        probs = torch.sigmoid(logits).cpu().float().numpy()  # (B, 26)

        results = []
        for prob in probs:
            results.append(_decode_pa100k(prob))
        return results


def _decode_pa100k(probs: np.ndarray, threshold: float = 0.50) -> Dict[str, str]:
    """Converts PA100K probability vector to attribute dict."""
    attrs: Dict[str, str] = {}

    # Gender
    attrs["gender"] = "female" if probs[0] > threshold else "male"

    # Age (take highest probability)
    age_probs = [("old", probs[1]), ("adult", probs[2]), ("young", probs[3])]
    attrs["age_group"] = max(age_probs, key=lambda x: x[1])[0]

    # Hat & Glasses
    attrs["hat"] = "on" if probs[7] > threshold else "off"
    attrs["glasses"] = "on" if probs[8] > threshold else "off"

    # Bag (take highest)
    bag_probs = [("handbag", probs[9]), ("shoulder_bag", probs[10]), ("backpack", probs[11])]
    best_bag = max(bag_probs, key=lambda x: x[1])
    attrs["bag_type"] = best_bag[0] if best_bag[1] > threshold else "none"

    # Upper type (short/long sleeve, coat)
    if probs[21] > threshold:
        attrs["upper_type"] = "coat"
    elif probs[13] > threshold:
        attrs["upper_type"] = "tshirt"
    elif probs[14] > threshold:
        attrs["upper_type"] = "shirt"
    else:
        attrs["upper_type"] = "other"

    # Lower type
    if probs[23] > threshold:
        attrs["lower_type"] = "shorts"
    elif probs[24] > threshold:
        attrs["lower_type"] = "skirt"
    elif probs[22] > threshold:
        attrs["lower_type"] = "pants"
    else:
        attrs["lower_type"] = "pants"

    return attrs


# ---------------------------------------------------------------------------
# CLIP Fallback
# ---------------------------------------------------------------------------

class CLIPPARWrapper:
    """
    Zero-shot attribute recognition using CLIP ViT-L/14.
    Substantially more accurate than vanilla CLIP for colour queries
    because we use per-attribute prompt ensembles.

    When PromptPAR weights are available, this is NOT used (PromptPAR
    achieves higher accuracy on PA100K). This runs as the reliable fallback.
    """

    def __init__(self, device: str = "cuda:0", half: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.type == "cuda"
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features: Optional[Dict[str, torch.Tensor]] = None

    def load(self) -> bool:
        try:
            import clip
            model_name = "ViT-L/14"
            self._model, self._preprocess = clip.load(model_name, device=self.device)
            if self.half:
                self._model = self._model.half()
            self._model.eval()
            self._preprocess_texts()
            return True
        except ImportError:
            # Try openai-clip from HuggingFace
            try:
                from transformers import CLIPModel, CLIPProcessor
                self._model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
                self._model.to(self.device)
                if self.half:
                    self._model = self._model.half()
                self._model.eval()
                self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
                self._backend = "hf"
                self._preprocess_texts_hf()
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _preprocess_texts(self):
        """Pre-computes CLIP text embeddings for all prompts (one-time cost)."""
        import clip
        self._text_features = {}
        self._label_map = {}   # attr_key → [(label, tensor_idx)]

        idx = 0
        all_texts = []
        for attr_key, label_prompts in CLIP_PROMPTS.items():
            self._label_map[attr_key] = []
            for label, prompts in label_prompts:
                tokens = clip.tokenize(prompts).to(self.device)
                with torch.no_grad():
                    feats = self._model.encode_text(tokens)
                    feats = feats.mean(0, keepdim=True)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                if self.half:
                    feats = feats.half()
                self._label_map[attr_key].append((label, feats))

    def _preprocess_texts_hf(self):
        """HuggingFace CLIP text preprocessing."""
        self._label_map = {}
        for attr_key, label_prompts in CLIP_PROMPTS.items():
            self._label_map[attr_key] = []
            for label, prompts in label_prompts:
                inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    feats = self._model.get_text_features(**inputs)
                    feats = feats.mean(0, keepdim=True)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                if self.half:
                    feats = feats.half()
                self._label_map[attr_key].append((label, feats))
        self._backend = "hf"

    @torch.no_grad()
    def predict_batch(self, crops: List[np.ndarray]) -> List[Dict[str, str]]:
        if self._model is None:
            return [{}] * len(crops)

        results = []
        for crop in crops:
            attrs = self._predict_one(crop)
            results.append(attrs)
        return results

    def _predict_one(self, crop: np.ndarray) -> Dict[str, str]:
        pil = Image.fromarray(crop[..., ::-1])   # BGR→RGB

        if hasattr(self, "_backend") and self._backend == "hf":
            inputs = self._processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            if self.half:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            image_features = self._model.get_image_features(**inputs)
        else:
            img_t = self._preprocess(pil).unsqueeze(0).to(self.device)
            if self.half:
                img_t = img_t.half()
            image_features = self._model.encode_image(img_t)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        attrs = {}
        for attr_key, label_feats in self._label_map.items():
            scores = []
            labels = []
            for label, text_feat in label_feats:
                score = (image_features @ text_feat.T).item()
                scores.append(score)
                labels.append(label)
            best_label = labels[np.argmax(scores)]
            attrs[attr_key] = best_label

        return attrs


# ---------------------------------------------------------------------------
# Unified PAR interface
# ---------------------------------------------------------------------------

class PARModel:
    """
    Unified PAR model: tries PromptPAR first, falls back to CLIP.
    Provides a single predict_batch() interface for the attribute worker.
    """

    def __init__(self, cfg: dict):
        par_cfg = cfg.get("models", {}).get("par", {})
        self.device = par_cfg.get("device", "cuda:0")
        self.half = bool(par_cfg.get("half", True))
        self.weights_path = par_cfg.get("weights_path", "models/promptpar_pa100k.pth")
        self.backbone = par_cfg.get("backbone", "swin_b")
        self.backend_name = "unloaded"

        self._promptpar = PromptPARWrapper(
            self.weights_path, self.backbone, self.device, self.half
        )
        self._clip = CLIPPARWrapper(self.device, self.half)

    def load(self):
        """Loads the best available model."""
        if self._promptpar.load():
            self.backend_name = "PromptPAR (OpenPAR / PA100K)"
        elif self._clip.load():
            self.backend_name = "CLIP ViT-L/14 (zero-shot fallback)"
        else:
            self.backend_name = "NO MODEL LOADED — attributes unavailable"
        return self.backend_name

    def predict_batch(self, crops: List[np.ndarray]) -> List[Dict[str, str]]:
        """
        Runs attribute recognition on a batch of BGR numpy crops.
        Returns list of attribute dicts.
        """
        if self._promptpar._loaded:
            return self._promptpar.predict_batch(crops)
        elif self._clip._model is not None:
            return self._clip.predict_batch(crops)
        else:
            return [{} for _ in crops]
