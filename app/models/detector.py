"""YOLOv8-face ONNX detector — pure onnxruntime, no torch.

Model: akanametov/yolo-face v1.0.0 (`yolov8n-face.onnx`).

Output layout (this particular export is already NMS'd and sorted by
confidence, padded with zero rows):

    shape == (1, 300, 21)
    row[0..3] = bbox **(x1, y1, x2, y2)** in **absolute pixel** coords
                of the model input (640x640), not normalized
    row[4]    = face confidence, raw logit (apply sigmoid)
    row[5..20] = 16 keypoint values (5 landmarks x 3 + padding) — we
                 don't use them in this app

This is much simpler than the raw YOLOv8 (1, 5, 8400) output — no
decoding, no NMS, no per-class filter, just iterate rows until you hit
the configured confidence threshold.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .. import config


@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float


class FaceDetector:
    """Thin wrapper around an ONNX YOLOv8-face session."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        providers: list[str] | None = None,
        conf: float | None = None,
        iou: float | None = None,
        imgsz: int | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else config.YOLO_FACE_MODEL_PATH
        self.providers = providers or config.ORT_PROVIDERS
        self.conf = conf if conf is not None else config.YOLO_FACE_CONF
        self.iou = iou if iou is not None else config.YOLO_FACE_IOU  # unused but kept for API compat
        self.imgsz = imgsz if imgsz is not None else config.YOLO_FACE_IMGSZ

        self._session: Any | None = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._loaded = False
        self._active_provider: str = ""

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._loaded:
            return
        if not self.model_path.exists():
            self._download_model()

        import onnxruntime as ort  # type: ignore

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 0
        so.inter_op_num_threads = 1

        available = ort.get_available_providers()
        providers = [p for p in self.providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=so,
            providers=providers,
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._active_provider = self._session.get_providers()[0]
        self._loaded = True

    def _download_model(self) -> None:
        import urllib.request

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        url = config.YOLO_FACE_MODEL_URL
        try:
            urllib.request.urlretrieve(url, self.model_path)  # noqa: S310
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download YOLOv8-face from {url}. "
                f"Place the .onnx file manually in {self.model_path.parent}. "
                f"Underlying error: {exc}"
            ) from exc

    # ----------------------------------------------------------------- detect
    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        if not self._loaded:
            self.load()
        assert self._session is not None

        import cv2  # local import

        h0, w0 = frame_bgr.shape[:2]
        imgsz = self.imgsz

        # Letterbox: keep aspect ratio, pad to imgsz x imgsz.
        scale = imgsz / max(h0, w0)
        new_h = int(round(h0 * scale))
        new_w = int(round(w0 * scale))
        pad_y = (imgsz - new_h) // 2
        pad_x = (imgsz - new_w) // 2

        # Preprocess: BGR -> RGB -> resize -> [0,1] -> CHW
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))  # HWC -> CHW

        padded = np.full((3, imgsz, imgsz), 114.0 / 255.0, dtype=np.float32)
        padded[:, pad_y : pad_y + new_h, pad_x : pad_x + new_w] = rgb
        batch = np.expand_dims(padded, 0)

        outputs = self._session.run(
            [self._output_name], {self._input_name: batch}
        )
        pred = outputs[0]
        if pred.ndim != 3 or pred.shape[0] != 1:
            return []
        # Expected shape: (1, 300, 21) for this export.
        if pred.shape[1] < 5 or pred.shape[2] < 5:
            return []
        pred = pred[0]  # (300, 21)

        # Rows are pre-NMS'd and sorted by confidence (descending). Zero
        # padding fills the rest. Iterate until we drop below threshold.
        boxes: list[FaceBox] = []
        for row in pred:
            x1_raw, y1_raw, x2_raw, y2_raw = (
                float(row[0]), float(row[1]), float(row[2]), float(row[3])
            )
            # All-zero rows (padding) have x1=y1=x2=y2=0; skip fast.
            if x2_raw <= 0 or y2_raw <= 0 or x2_raw <= x1_raw or y2_raw <= y1_raw:
                break  # rows are sorted; nothing better after this
            conf = 1.0 / (1.0 + math.exp(-float(row[4])))  # sigmoid
            if conf < self.conf:
                break  # rows are sorted; further rows only get worse

            # Undo letterbox: subtract pad, divide by scale.
            x1 = (x1_raw - pad_x) / scale
            y1 = (y1_raw - pad_y) / scale
            x2 = (x2_raw - pad_x) / scale
            y2 = (y2_raw - pad_y) / scale

            xi1 = max(0, min(w0 - 1, int(round(x1))))
            yi1 = max(0, min(h0 - 1, int(round(y1))))
            xi2 = max(0, min(w0 - 1, int(round(x2))))
            yi2 = max(0, min(h0 - 1, int(round(y2))))
            if xi2 - xi1 < 8 or yi2 - yi1 < 8:
                continue
            boxes.append(FaceBox(xi1, yi1, xi2, yi2, conf))

        return boxes

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def active_provider(self) -> str:
        return self._active_provider
