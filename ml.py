"""CLIP via Immich's machine-learning service (the same model Immich itself uses)."""

from __future__ import annotations

import io
import json

import numpy as np
import requests
from PIL import Image

from rules import PROMPT_GROUPS, SCORE_NAMES


class Clip:
    def __init__(self, url: str, model: str, timeout: int = 120) -> None:
        self.url = url.rstrip("/") + "/predict"
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self._groups: dict[str, tuple[np.ndarray, list[int]]] = {}

    def _predict(self, kind: str, files: dict) -> np.ndarray:
        entries = json.dumps({"clip": {kind: {"modelName": self.model}}})
        r = self.session.post(self.url, files={"entries": (None, entries), **files}, timeout=self.timeout)
        r.raise_for_status()
        vec = np.asarray(json.loads(r.json()["clip"]), dtype=np.float32)
        return vec / np.linalg.norm(vec)

    def text(self, s: str) -> np.ndarray:
        return self._predict("textual", {"text": (None, s)})

    def image(self, data: bytes, mime: str = "image/webp") -> np.ndarray:
        return self._predict("visual", {"image": ("thumb", data, mime)})

    def load_prompts(self) -> None:
        """Fetch text embeddings once per run (~140 prompts, a few seconds)."""
        cache: dict[str, np.ndarray] = {}
        for name, (positives, negatives) in PROMPT_GROUPS.items():
            prompts, sizes = [], []
            for group in positives:
                prompts += group
                sizes.append(len(group))
            prompts += negatives
            for p in prompts:
                if p not in cache:
                    cache[p] = self.text(p)
            self._groups[name] = (np.stack([cache[p] for p in prompts]), sizes)

    def scores(self, emb: np.ndarray) -> dict[str, float]:
        """Zero-shot: softmax per group (logit scale 100), summed per positive subgroup."""
        out: dict[str, float] = {}
        for name, (matrix, sizes) in self._groups.items():
            logits = 100.0 * (matrix @ emb)
            p = np.exp(logits - logits.max())
            p /= p.sum()
            start = 0
            for score_name, size in zip(SCORE_NAMES[name], sizes):
                out[score_name] = float(p[start : start + size].sum())
                start += size
        return out


def image_stats(data: bytes) -> dict[str, float]:
    """Brightness (mean), contrast (std) and sharpness (Laplacian variance) of the greyscale thumbnail."""
    g = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float32)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    return {"brightness": float(g.mean()), "contrast": float(g.std()), "sharpness": float(lap.var())}
