"""Camera listing + MJPEG stream endpoints."""
from __future__ import annotations

import asyncio
import logging
import time

import cv2
from fastapi import APIRouter, HTTPException, Path, Request
from fastapi.responses import StreamingResponse

from .. import config
from ..schemas import CameraInfo
from ..services.camera_service import camera_manager

logger = logging.getLogger("face-emotion-app.camera")

router = APIRouter(prefix="/api/camera", tags=["camera"])


@router.get("/list", response_model=list[CameraInfo])
async def list_cameras() -> list[CameraInfo]:
    infos = camera_manager.list_cameras()
    return [CameraInfo(id=i.id, name=i.name, available=i.available) for i in infos]


@router.get("/{camera_id}/stream")
async def stream_camera(
    request: Request, camera_id: int = Path(ge=0, le=16)
) -> StreamingResponse:
    """MJPEG stream. Each frame is JPEG-encoded and pushed as part of a
    multipart/x-mixed-replace response, so any browser ``<img>`` tag can
    render it without extra JavaScript.
    """
    pipeline = request.app.state.pipeline

    try:
        cap = camera_manager.acquire(camera_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Throw away the first few frames — many webcams return black or
    # wrong-exposure frames in the first ~100 ms.
    for _ in range(5):
        cap.read()

    boundary = b"--frame"

    async def frame_generator():
        frame_idx = 0
        last_log_t = 0.0
        last_fps_log = 0.0
        fps_frames = 0
        fps = 0.0
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("Camera %s: client disconnected", camera_id)
                    break
                ok, frame = await asyncio.get_running_loop().run_in_executor(
                    None, cap.read
                )
                if not ok or frame is None:
                    await asyncio.sleep(0.05)
                    continue

                t0 = time.perf_counter()
                try:
                    annotated, detections = await asyncio.get_running_loop().run_in_executor(
                        None, pipeline.detect_and_draw, frame
                    )
                except Exception as exc:
                    logger.exception(
                        "detect_and_draw failed on frame %d: %s", frame_idx, exc
                    )
                    annotated = frame
                    detections = []
                dt_ms = (time.perf_counter() - t0) * 1000.0

                fps_frames += 1
                now = time.time()
                if now - last_fps_log > 1.0:
                    fps = fps_frames / (now - last_fps_log)
                    fps_frames = 0
                    last_fps_log = now
                    logger.info(
                        "Camera %s: %d faces, %.0f ms/frame, %.1f fps",
                        camera_id, len(detections), dt_ms, fps,
                    )

                ok, buf = cv2.imencode(
                    ".jpg", annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY],
                )
                if not ok:
                    continue
                yield (
                    boundary
                    + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(buf)).encode()
                    + b"\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
                frame_idx += 1
                await asyncio.sleep(0)
        finally:
            # Always release the camera when the loop ends — disconnect,
            # exception, or anything else. This is what turns the LED off.
            try:
                cap.release()
            except Exception:
                pass
            try:
                camera_manager.release(camera_id)
            except Exception:
                pass
            logger.info("Camera %s: capture released (frames=%d)", camera_id, frame_idx)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/{camera_id}/snapshot")
async def camera_snapshot(
    request: Request, camera_id: int = Path(ge=0, le=16)
):
    """Capture a single annotated frame as JPEG. The camera is opened and
    immediately closed so the OS turns the LED off once the response
    is sent.
    """
    from fastapi.responses import Response

    pipeline = request.app.state.pipeline
    try:
        with camera_manager.use_temp(camera_id) as cap:
            # Warm up
            for _ in range(5):
                await asyncio.get_running_loop().run_in_executor(None, cap.read)
            ok, frame = await asyncio.get_running_loop().run_in_executor(
                None, cap.read
            )
            if not ok or frame is None:
                raise HTTPException(
                    status_code=500, detail="Failed to capture frame"
                )
            annotated, detections = await asyncio.get_running_loop().run_in_executor(
                None, pipeline.detect_and_draw, frame
            )
            ok, buf = cv2.imencode(
                ".jpg", annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY],
            )
            if not ok:
                raise HTTPException(
                    status_code=500, detail="Failed to encode JPEG"
                )
            jpeg_bytes = buf.tobytes()
            n_faces = len(detections)
            summary = ",".join(
                f"{r.emotion}:{r.score:.2f}" for r in detections
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "X-Face-Count": str(n_faces),
            "X-Emotion-Summary": summary,
        },
    )


@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: int = Path(ge=0, le=16)) -> dict:
    camera_manager.release(camera_id)
    return {"stopped": camera_id}


@router.post("/release-all")
async def release_all_cameras() -> dict:
    """Force-release every camera the server is holding. Use this if
    the LED is stuck on after closing the browser tab."""
    held = camera_manager.held()
    camera_manager.release_all()
    logger.info("release-all: freed cameras %s", held)
    return {"released": held}
