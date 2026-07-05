"""ViT emotion classifier — pure onnxruntime, no torch / no transformers.

Model: trpakov/vit-face-expression (ONNX export from the same HuggingFace
repo, 7-class FER2013). Preprocessing per the model's preprocessor_config:
- resize to 224x224
- rescale 1/255
- normalize with mean=0.5 std=0.5
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

from .. import config


def _robust_download(url: str, dst: str | Path, max_attempts: int = 8) -> None:
    """Download ``url`` to ``dst`` with resume + lenient SSL.

    Some networks (corporate proxies, ssl inspection) cause
    ``SSL: UNEXPECTED_EOF_WHILE_READING`` for large files. We:
      - relax SSL verification (cert inspection is the usual culprit)
      - download to a ``.part`` file and resume on Range
      - retry up to ``max_attempts`` times
    """
    import ssl
    import time

    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".part")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    existing = tmp.stat().st_size if tmp.exists() else 0
    for attempt in range(1, max_attempts + 1):
        headers = {"User-Agent": "face-emotion-app/1.0"}
        if existing:
            headers["Range"] = f"bytes={existing}-"
        try:
            req = __import__("urllib.request").request.Request(url, headers=headers)
            with __import__("urllib.request").request.urlopen(
                req, timeout=300, context=ctx
            ) as r:
                total = int(r.headers.get("Content-Length", 0)) + existing
                mode = "ab" if existing else "wb"
                downloaded = existing
                with open(tmp, mode) as f:
                    while True:
                        buf = r.read(256 * 1024)
                        if not buf:
                            break
                        f.write(buf)
                        downloaded += len(buf)
            tmp.replace(dst)
            return
        except Exception:
            if tmp.exists():
                existing = tmp.stat().st_size
            time.sleep(2 + attempt)
    raise RuntimeError(f"Failed to download {url} after {max_attempts} attempts")


class EmotionClassifier:
    """Wraps an ONNX ViT model that emits 7 emotion logits."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
        image_size: int | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else config.EMOTION_MODEL_PATH
        self.providers = providers or config.ORT_PROVIDERS
        self.image_size = image_size or config.EMOTION_IMAGE_SIZE

        self._session: Any | None = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._loaded = False
        self._active_provider: str = ""
        self._labels: list[str] = list(config.EMOTION_LABELS)

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._loaded:
            return
        if not self.model_path.exists():
            self._download_model()

        # Try to also pull config.json for accurate labels (fallback to config defaults)
        try:
            self._sync_labels()
        except Exception:
            pass

        import onnxruntime as ort  # type: ignore

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        providers = [p for p in self.providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        # Some ViT ONNX exports emit named outputs; we only need the first.
        self._output_name = self._session.get_outputs()[0].name
        self._active_provider = self._session.get_providers()[0]
        self._loaded = True

    def _download_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        url = config.EMOTION_MODEL_URL
        try:
            _robust_download(url, self.model_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download emotion model from {url}. "
                f"Place the .onnx file manually in {self.model_path.parent}. "
                f"Underlying error: {exc}"
            ) from exc

    def _sync_labels(self) -> None:
        """Optionally pull config.json to learn id2label. Best effort."""
        cache = self.model_path.with_suffix(".config.json")
        if not cache.exists():
            try:
                urlretrieve(config.EMOTION_CONFIG_URL, cache)  # noqa: S310
            except Exception:
                return
        try:
            cfg = json.loads(cache.read_text(encoding="utf-8"))
            id2label = cfg.get("id2label") or {}
            if id2label:
                sorted_items = sorted(id2label.items(), key=lambda kv: int(kv[0]))
                self._labels = [str(v) for _, v in sorted_items]
        except Exception:
            pass

    # --------------------------------------------------------------- classify
    def classify(self, face_bgr: np.ndarray) -> tuple[str, float, dict[str, float]]:
        if not self._loaded:
            self.load()
        assert self._session is not None

        # BGR -> RGB -> resize 224 -> [0,1] -> (x - 0.5) / 0.5  (== x*2 - 1)
        face_rgb = face_bgr[..., ::-1]
        pil = Image.fromarray(face_rgb)
        if pil.size != (self.image_size, self.image_size):
            pil = pil.resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        # HWC -> CHW -> batch
        arr = arr.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)
        # Cast to float32 explicitly (the model expects float32)
        arr = arr.astype(np.float32)

        outputs = self._session.run(
            [self._output_name], {self._input_name: arr}
        )
        logits = outputs[0]
        if logits.ndim == 2:
            logits = logits[0]
        # softmax
        logits = logits.astype(np.float64)
        probs = _softmax(logits)
        labels = self._labels
        if probs.shape[0] != len(labels):
            # fall back to config defaults
            labels = list(config.EMOTION_LABELS)
        n = min(probs.shape[0], len(labels))
        scores = {labels[i]: float(probs[i]) for i in range(n)}
        top_idx = int(np.argmax(probs[:n]))
        return labels[top_idx], float(probs[top_idx]), scores

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def active_provider(self) -> str:
        return self._active_provider


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)
