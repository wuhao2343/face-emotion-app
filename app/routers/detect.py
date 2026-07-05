"""JSON detection API — image in, detections out. Suitable for other programs
to call programmatically.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from .. import config
from ..models.pipeline import DetectionResult
from ..schemas import BBox, Detection, FrameResult
from ..utils import check_size_preflight, read_upload_with_limit

router = APIRouter(prefix="/api/detect", tags=["detect"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}


def _to_dict(r: DetectionResult) -> Detection:
    return Detection(
        bbox=BBox(x1=r.bbox[0], y1=r.bbox[1], x2=r.bbox[2], y2=r.bbox[3]),
        emotion=r.emotion,
        score=r.score,
        all_scores=r.all_scores,
    )


@router.post("/frame", response_model=FrameResult)
async def detect_frame(request: Request, file: UploadFile = File(...)) -> FrameResult:
    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}",
        )
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    await check_size_preflight(request, max_bytes)
    raw = await read_upload_with_limit(file, max_bytes)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    pipeline = request.app.state.pipeline
    results, elapsed_ms = pipeline.detect_only_json(frame)
    return FrameResult(
        count=len(results),
        detections=[_to_dict(r) for r in results],
        inference_ms=round(elapsed_ms, 2),
        device=pipeline.detector.active_provider or "cpu",
    )


@router.post("/frame_base64", response_model=FrameResult)
async def detect_frame_b64(
    request: Request, payload: dict
) -> FrameResult:
    """Same as ``/frame`` but accepts a base64-encoded image. Useful for
    clients that already hold frames as strings (e.g. browser canvas).
    """
    import base64

    b64 = payload.get("image_base64") or payload.get("image")
    if not b64:
        raise HTTPException(status_code=400, detail="Missing 'image_base64' field")
    if "," in b64:  # data URL: "data:image/jpeg;base64,xxxx"
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {exc}") from exc

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    pipeline = request.app.state.pipeline
    t0 = time.perf_counter()
    results, _ = pipeline.detect_only_json(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return FrameResult(
        count=len(results),
        detections=[_to_dict(r) for r in results],
        inference_ms=round(elapsed_ms, 2),
        device=pipeline.detector.active_provider or "cpu",
    )
