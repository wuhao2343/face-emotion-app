"""Image upload → annotated image response."""
from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import Response

from .. import config
from ..utils import check_size_preflight, read_upload_with_limit

router = APIRouter(prefix="/api/detect", tags=["image"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}


@router.post("/image")
async def detect_image(request: Request, file: UploadFile = File(...)) -> Response:
    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}",
        )

    # NOTE: cv2.imdecode needs the full bytes in memory, so the 4 GB limit
    # implies a 4 GB peak working set on this endpoint. That is a hard
    # trade-off of feeding the decoder from bytes; for an image-only path
    # the realistic upper bound is well under 100 MB, so the limit is set
    # globally by MAX_UPLOAD_MB (shared with /api/detect/video) and the
    # streaming variant is only applied where it can be honoured (video).
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    await check_size_preflight(request, max_bytes)
    raw = await read_upload_with_limit(file, max_bytes)

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image data")

    pipeline = request.app.state.pipeline
    annotated, results = pipeline.detect_and_draw(frame)

    # Encode back as JPEG
    ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode result image")

    # Headers so the client can show a count and a quick summary
    count = len(results)
    summary = ",".join(f"{r.emotion}:{r.score:.2f}" for r in results) if results else ""
    headers = {
        "X-Face-Count": str(count),
        "X-Emotion-Summary": summary,
    }
    return Response(content=buf.tobytes(), media_type="image/jpeg", headers=headers)
