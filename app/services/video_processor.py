"""Frame-by-frame video processing with progress reporting.

Used by the ``POST /api/detect/video`` endpoint. The pipeline is two-stage:

1. ``cv2.VideoWriter`` reads the source, runs face detection + emotion
   classification on every frame, and writes an annotated MP4 to a
   temp file. OpenCV's writer is video-only — it has no audio support.
2. An ffmpeg subprocess muxes that annotated video stream with the
   original audio stream into the final output MP4 (H.264 video + AAC
   audio, ``+faststart`` for browser-friendly streaming). ffmpeg comes
   from the ``imageio-ffmpeg`` package so no system install is needed.

If ffmpeg is unavailable, the source has no audio, or the mux fails for
any reason, the function falls back to the cv2-only output (silent MP4,
same as before) and logs a warning — so a missing dependency never
breaks the endpoint, it just loses the audio track.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .. import config
from ..models.pipeline import Pipeline

logger = logging.getLogger("face-emotion-app.video")


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    total_frames: int

    @property
    def duration_s(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.total_frames / self.fps


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

# Module-level cache. Semantics:
#   None  -> not probed yet (initial state)
#   False -> probed and not found
#   str   -> absolute path to a working ffmpeg binary
_FFMPEG_EXE_CACHE: "str | None | bool" = None


def _ffmpeg_exe() -> str | None:
    """Locate the bundled ffmpeg binary, or ``None`` if unavailable.

    Cached after the first call — imageio-ffmpeg is slow to import.
    """
    global _FFMPEG_EXE_CACHE
    if _FFMPEG_EXE_CACHE is None:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = shutil.which("ffmpeg")
        _FFMPEG_EXE_CACHE = exe or False
        if _FFMPEG_EXE_CACHE:
            logger.info("ffmpeg available at %s", _FFMPEG_EXE_CACHE)
        else:
            logger.warning(
                "ffmpeg not found — processed videos will have no audio. "
                "Install imageio-ffmpeg to enable audio muxing."
            )
    return _FFMPEG_EXE_CACHE or None


def _source_has_audio(ffmpeg: str, source: Path) -> bool:
    """Probe ``source`` for an audio stream by parsing ffmpeg's stderr."""
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(source)],
            capture_output=True,
            timeout=15,
        )
    except Exception as exc:
        logger.debug("audio probe failed: %s", exc)
        return False
    return b"Audio:" in proc.stderr


def _mux_audio(
    ffmpeg: str,
    annotated: Path,
    source: Path,
    output: Path,
) -> bool:
    """Mux the original audio into the annotated video. Returns True on success.

    Strategy: probe the source for an audio stream first; if none, write the
    annotated video straight through (re-encoded to H.264 + faststart) and
    return True. If muxing fails for any other reason, return False so the
    caller can fall back to the cv2-only output.
    """
    has_audio = _source_has_audio(ffmpeg, source)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(annotated)]
    if has_audio:
        cmd += ["-i", str(source)]

    cmd += ["-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "128k"]
    cmd += [
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",      # max player compatibility
        "-movflags", "+faststart",  # browser-friendly streaming
        "-shortest",
        str(output),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
    except Exception as exc:
        logger.warning("ffmpeg mux failed (%s) — falling back to silent output", exc)
        return False
    if proc.returncode != 0:
        logger.warning(
            "ffmpeg mux rc=%s stderr=%s — falling back to silent output",
            proc.returncode,
            proc.stderr.decode("utf-8", errors="replace")[:500],
        )
        return False
    return output.exists() and output.stat().st_size > 0


# ---------------------------------------------------------------------------
# VideoProcessor
# ---------------------------------------------------------------------------


class VideoProcessor:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def probe(self, source: str | Path) -> VideoMeta:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if w == 0 or h == 0:
                # Fallback: decode first frame
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Cannot decode first frame of video")
                h, w = frame.shape[:2]
                total = max(total, 1)
            return VideoMeta(fps=fps, width=w, height=h, total_frames=total)
        finally:
            cap.release()

    def process(
        self,
        source: str | Path,
        output: str | Path,
        on_progress: Callable[[int, float], None] | None = None,
        max_side: int | None = None,
    ) -> VideoMeta:
        """Read ``source`` and write annotated MP4 to ``output``.

        ``on_progress`` is called with ``(frame_idx, fraction)`` after each
        frame. ``max_side`` rescales the longer edge for memory/performance.
        Audio from ``source`` is muxed in when ffmpeg is available.
        """
        meta = self.probe(source)
        max_side = max_side or config.MAX_VIDEO_SIDE
        scale = self._scale_for(meta.width, meta.height, max_side)
        out_w = max(1, int(round(meta.width * scale)))
        out_h = max(1, int(round(meta.height * scale)))

        ffmpeg = _ffmpeg_exe()
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Keep the intermediate annotated file inside a tempdir so it is
        # always cleaned up, even on exception.
        with tempfile.TemporaryDirectory(prefix="fevid_proc_") as td:
            annotated_path = Path(td) / "annotated.mp4"

            # ---- Stage 1: cv2 writes annotated video (no audio) -----------
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {source}")
            try:
                # mp4v -> broadly playable; we'll re-mux to H.264 below.
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(annotated_path), fourcc, meta.fps, (out_w, out_h)
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Failed to open VideoWriter for {annotated_path}"
                    )
                try:
                    idx = 0
                    t_start = time.perf_counter()
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if scale != 1.0:
                            frame = cv2.resize(
                                frame, (out_w, out_h), interpolation=cv2.INTER_AREA
                            )
                        annotated, _ = self.pipeline.detect_and_draw(frame)
                        writer.write(annotated)
                        idx += 1
                        if on_progress and meta.total_frames:
                            frac = min(1.0, idx / meta.total_frames)
                            on_progress(idx, frac)
                    if on_progress:
                        on_progress(idx, 1.0)
                    _ = t_start  # available for future ETA calculation
                finally:
                    writer.release()
            finally:
                cap.release()

            # ---- Stage 2: mux original audio into annotated video ---------
            if ffmpeg and _mux_audio(ffmpeg, annotated_path, source, output_path):
                return meta
            # Fallback: ffmpeg missing or mux failed — ship the cv2 output
            # verbatim (silent, but at least the user gets a result).
            shutil.copy(annotated_path, output_path)
            return meta

    @staticmethod
    def _scale_for(w: int, h: int, max_side: int) -> float:
        if max_side <= 0:
            return 1.0
        longer = max(w, h)
        if longer <= max_side:
            return 1.0
        return max_side / float(longer)
