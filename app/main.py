"""FastAPI application entry point — ONNX-only, no torch.

Run via ``python run.py`` or ``uvicorn app.main:app --reload``.
"""
from __future__ import annotations

import io
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, config
from .models import EmotionClassifier, FaceDetector, Pipeline
from .routers import camera, detect, health, image, video
from .services import CameraManager, VideoProcessor

# ---- Windows console encoding shim --------------------------------------
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("face-emotion-app")
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Face Emotion App v%s starting up (ONNX-only stack)", __version__)
    logger.info("=" * 60)
    logger.info("ORT providers (in priority order): %s", config.ORT_PROVIDERS)
    logger.info("YOLOv8-face model: %s", config.YOLO_FACE_MODEL_PATH)
    logger.info("Emotion model:     %s", config.EMOTION_MODEL_PATH)

    detector = FaceDetector()
    classifier = EmotionClassifier()
    pipeline = Pipeline(detector=detector, classifier=classifier)

    try:
        pipeline.ensure_loaded()
        logger.info(
            "[OK] Models loaded. Detector provider: %s | Classifier provider: %s",
            detector.active_provider or "?",
            classifier.active_provider or "?",
        )
    except Exception as exc:
        logger.error("[FAIL] Failed to load models: %s", exc)
        # Still start the app so the user can see the error in the UI.

    app.state.pipeline = pipeline
    app.state.video_processor = VideoProcessor(pipeline)
    app.state.camera_manager = CameraManager()

    yield

    try:
        app.state.camera_manager.release_all()
    except Exception:
        pass
    logger.info("Face Emotion App stopped.")


app = FastAPI(
    title="Face Emotion Recognition API",
    description=(
        "YOLOv8-face detection + ViT emotion classification, both via ONNX "
        "Runtime. No PyTorch required — works on CPU and DirectML GPU."
    ),
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(image.router)
app.include_router(detect.router)
app.include_router(video.router)
app.include_router(camera.router)

app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(str(config.STATIC_DIR / "index.html"))


@app.get("/favicon.ico")
async def favicon() -> JSONResponse:
    return JSONResponse(status_code=204, content=None)
