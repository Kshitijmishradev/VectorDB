"""Shared lazy-loading utilities for the CIFAR-100 + CLIP demo."""
import base64
import io
import json
import pickle
from pathlib import Path

import numpy as np


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def choose_device(torch):
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_ml(data_root, device=None):
    try:
        import clip
        import torch
        from torchvision.datasets import CIFAR100
    except ImportError as exc:
        raise RuntimeError("install requirements-demo.txt first") from exc
    device = device or choose_device(torch)
    model, preprocess = clip.load("ViT-B/32", device=device)
    return torch, clip, CIFAR100, model.eval(), preprocess, device


def image_to_data_url(image, size=144):
    image = image.copy()
    image.thumbnail((size, size))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=86)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def read_coarse_labels(dataset_root, split):
    path = Path(dataset_root) / "cifar-100-python" / split
    with path.open("rb") as source:
        payload = pickle.load(source, encoding="latin1")
    key = "coarse_labels" if "coarse_labels" in payload else b"coarse_labels"
    return np.asarray(payload[key], dtype=np.int64)


def load_manifest(artifact_dir):
    return json.loads((Path(artifact_dir) / "manifest.json").read_text())
