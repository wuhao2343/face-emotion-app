"""Video upload → annotated MP4 download (with progress tracking)."""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .. import config
from ..utils import check_size_preflight, save_upload_with_limit

router = APIRouter(prefix="/api/detect", tags=["video"])

ALLOWED_VIDEO_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska",
    "video/webm", "application/octet-stream",  # some browsers omit the type
}
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# In-memory task store. For a single-user desktop app this is sufficient.
_TASKS: dict[str, dict] = {}
_TASKS_LOCK = asyncio.Lock()


@router.post("/video")
async def detect_video(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = File(...),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXT and file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type: {file.content_type or suffix}",
        )

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    await check_size_preflight(request, max_bytes)

    task_id = uuid.uuid4().hex
    work_dir = Path(tempfile.mkdtemp(prefix=f"fevid_{task_id}_"))
    src_path = work_dir / f"input{suffix or '.mp4'}"

    try:
        await save_upload_with_limit(file, src_path, max_bytes)
    except HTTPException:
        # 413 / client disconnect: drop the temp dir we just created so we
        # don't leave orphan work_dirs behind for rejected uploads.
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    pipeline = request.app.state.pipeline
    output_path = config.OUTPUTS_DIR / f"{task_id}.mp4"

    async with _TASKS_LOCK:
        _TASKS[task_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "Queued",
            "download_url": None,
            "error": None,
            "meta": {},
        }

    async def _run() -> None:
        async with _TASKS_LOCK:
            _TASKS[task_id].update(status="processing", message="Starting...")
        try:
            processor = request.app.state.video_processor
            loop = asyncio.get_running_loop()

            def on_progress(frame_idx: int, frac: float) -> None:
                # Called from the worker thread; schedule the dict update on
                # the event loop to avoid races.
                async def _update() -> None:
                    async with _TASKS_LOCK:
                        _TASKS[task_id]["progress"] = float(frac)
                        _TASKS[task_id]["message"] = (
                            f"Processed {frame_idx} frames ({int(frac * 100)}%)"
                        )

                # schedule on the loop
                try:
                    loop.call_soon_threadsafe(asyncio.ensure_future, _update())
                except RuntimeError:
                    pass

            meta = await loop.run_in_executor(
                None,
                lambda: processor.process(
                    src_path, output_path, on_progress=on_progress
                ),
            )
            async with _TASKS_LOCK:
                _TASKS[task_id].update(
                    status="done",
                    progress=1.0,
                    message="Done",
                    download_url=f"/api/detect/video/{task_id}/download",
                    meta={
                        "fps": meta.fps,
                        "width": meta.width,
                        "height": meta.height,
                        "total_frames": meta.total_frames,
                        "duration_s": meta.duration_s,
                    },
                )
        except Exception as exc:  # pragma: no cover
            async with _TASKS_LOCK:
                _TASKS[task_id].update(status="error", error=str(exc), message="Error")
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    background.add_task(_run)
    return JSONResponse({"task_id": task_id, "status": "queued"})


@router.get("/video/{task_id}/status")
async def video_status(task_id: str) -> JSONResponse:
    async with _TASKS_LOCK:
        info = _TASKS.get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Unknown task_id")
    return JSONResponse({"task_id": task_id, **info})


@router.get("/video/{task_id}/download")
async def video_download(task_id: str) -> FileResponse:
    output_path = config.OUTPUTS_DIR / f"{task_id}.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output not ready")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=f"annotated_{task_id}.mp4",
    )
