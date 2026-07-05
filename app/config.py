"""Application configuration.

All settings can be overridden via environment variables or a .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---- Paths ---------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent
MODELS_DIR: Path = BASE_DIR / "models"
OUTPUTS_DIR: Path = BASE_DIR / "outputs"
STATIC_DIR: Path = BASE_DIR / "static"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Server --------------------------------------------------------------
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))

# ---- ONNX Runtime providers ---------------------------------------------
# onnxruntime provider list, tried in order. Examples:
#   "CPUExecutionProvider"                -> CPU only
#   "DmlExecutionProvider", "CPUExecutionProvider" -> DirectML (GPU) then CPU
#   "CUDAExecutionProvider", "CPUExecutionProvider" -> CUDA EP then CPU
ORT_PROVIDERS: list[str] = [
    p.strip()
    for p in os.getenv(
        "ORT_PROVIDERS", "DmlExecutionProvider,CPUExecutionProvider"
    ).split(",")
    if p.strip()
]

# ---- YOLOv8-face ---------------------------------------------------------
# Default URL points to the akanametov/yolo-face 1.0.0 release.
YOLO_FACE_MODEL_NAME: str = os.getenv("YOLO_FACE_MODEL", "yolov8n-face.onnx")
YOLO_FACE_MODEL_PATH: Path = MODELS_DIR / YOLO_FACE_MODEL_NAME
YOLO_FACE_MODEL_URL: str = os.getenv(
    "YOLO_FACE_MODEL_URL",
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.onnx",
)
YOLO_FACE_CONF: float = float(os.getenv("YOLO_FACE_CONF", "0.35"))
YOLO_FACE_IOU: float = float(os.getenv("YOLO_FACE_IOU", "0.45"))
YOLO_FACE_IMGSZ: int = int(os.getenv("YOLO_FACE_IMGSZ", "640"))

# ---- Emotion classifier (trpakov/vit-face-expression, ONNX) -------------
EMOTION_MODEL_NAME: str = os.getenv("EMOTION_MODEL_NAME", "trpakov-vit-face.onnx")
EMOTION_MODEL_PATH: Path = MODELS_DIR / EMOTION_MODEL_NAME
EMOTION_MODEL_URL: str = os.getenv(
    "EMOTION_MODEL_URL",
    "https://huggingface.co/trpakov/vit-face-expression/resolve/main/onnx/model.onnx",
)
EMOTION_CONFIG_URL: str = os.getenv(
    "EMOTION_CONFIG_URL",
    "https://huggingface.co/trpakov/vit-face-expression/resolve/main/onnx/config.json",
)
EMOTION_PREPROC_URL: str = os.getenv(
    "EMOTION_PREPROC_URL",
    "https://huggingface.co/trpakov/vit-face-expression/resolve/main/onnx/preprocessor_config.json",
)
EMOTION_IMAGE_SIZE: int = int(os.getenv("EMOTION_IMAGE_SIZE", "224"))

# FER2013 7-class labels (order matches trpakov/vit-face-expression id2label)
EMOTION_LABELS: tuple[str, ...] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)
EMOTION_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "angry":    ( 56,  56, 255),  # red
    "disgust":  ( 76, 153,   0),  # olive green
    "fear":     (204,  86,  86),  # dusty rose
    "happy":    ( 80, 220, 100),  # bright green
    "sad":      (220, 140,  60),  # muted blue
    "surprise": ( 90, 220, 240),  # cyan
    "neutral":  (200, 200, 200),  # grey
}

# ---- Video / upload limits ----------------------------------------------
# Hard cap enforced by the helpers in app/utils.py. Memory profile:
#   - /api/detect/video   : streamed to a temp file (flat O(1 MiB) RAM).
#   - /api/detect/image   : full bytes held in RAM (peak ≈ MAX_UPLOAD_MB),
#                           because cv2.imdecode needs them. In practice an
#                           image larger than a few hundred MB is not useful
#                           even if the limit allows it.
#   - /api/detect/frame   : same trade-off as /image.
# Keep this in sync with the hint shown in the frontend (see
# static/index.html dropzone text and app.js size pre-check).
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "4096"))
MAX_VIDEO_SIDE: int = int(os.getenv("MAX_VIDEO_SIDE", "1280"))
JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "80"))

# ---- Camera discovery ----------------------------------------------------
CAMERA_PROBE_RANGE: int = int(os.getenv("CAMERA_PROBE_RANGE", "4"))

# ---- Logging -------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
