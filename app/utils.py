"""Shared HTTP/upload helpers.

The routers under :mod:`app.routers` all need to:

1. Reject oversize uploads with HTTP 413 (Payload Too Large).
2. Avoid loading multi-gigabyte bodies into memory before the size check.

This module provides three small async helpers that take a FastAPI
``Request``/``UploadFile`` and a byte limit and return either the bytes
(:func:`read_upload_with_limit`), a streamed file on disk
(:func:`save_upload_with_limit`), or a fast 413 if the
``Content-Length`` header already exceeds the limit
(:func:`check_size_preflight`).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Final

from fastapi import HTTPException, Request, UploadFile, status

# 1 MiB read chunk. Tuned so memory stays bounded regardless of upload
# size while keeping the syscall count low (a 4 GB upload = 4096 reads).
_CHUNK_SIZE: Final[int] = 1024 * 1024


def _limit_mb(max_bytes: int) -> int:
    return max_bytes // (1024 * 1024)


def _too_large(max_bytes: int) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        detail=f"File too large; limit is {_limit_mb(max_bytes)} MB",
    )


async def check_size_preflight(request: Request, max_bytes: int) -> None:
    """Reject 413 up-front if ``Content-Length`` already exceeds the limit.

    Saves the wire cost of receiving a 5 GB upload only to reject it after
    4 GB. Falls through silently if the header is missing or unparseable —
    the streaming check below is still authoritative.
    """
    cl = request.headers.get("content-length")
    if cl is None:
        return
    try:
        n = int(cl)
    except ValueError:
        return
    if n > max_bytes:
        raise _too_large(max_bytes)


async def read_upload_with_limit(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload into memory in 1 MiB chunks, aborting with 413 on overflow.

    Memory footprint at peak: ~max_bytes (the returned ``bytes`` object).
    Suitable when the caller actually needs the bytes in memory (e.g. feeding
    ``cv2.imdecode``). For multi-GB uploads that are going straight to disk
    use :func:`save_upload_with_limit` instead.
    """
    buf = io.BytesIO()
    total = 0
    while True:
        chunk = await upload.read(_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise _too_large(max_bytes)
        buf.write(chunk)
    return buf.getvalue()


async def save_upload_with_limit(
    upload: UploadFile, dest: Path, max_bytes: int
) -> int:
    """Stream an upload to ``dest`` on disk, aborting with 413 on overflow.

    Memory footprint: O(``_CHUNK_SIZE``) regardless of upload size — the
    body never lives fully in RAM. On any failure (413, client disconnect,
    OS error) the partial file at ``dest`` is removed before the exception
    propagates.

    Returns the number of bytes written.
    """
    total = 0
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await upload.read(_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise _too_large(max_bytes)
                f.write(chunk)
    except Exception:
        # Best-effort cleanup of the partial file. The caller is still
        # responsible for cleaning up any wrapping directory (e.g. work_dir).
        try:
            dest.unlink()
        except FileNotFoundError:
            pass
        raise
    return total
