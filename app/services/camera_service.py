"""OpenCV camera capture manager.

Holds one ``VideoCapture`` per camera id and lets callers open / release
them safely. The camera stream router uses this to share capture objects
between concurrent requests.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import cv2

from .. import config


@dataclass
class CameraInfo:
    id: int
    name: str
    available: bool


class CameraManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._captures: dict[int, cv2.VideoCapture] = {}

    # ---------------------------------------------------------------- helpers
    def list_cameras(self, probe_range: int | None = None) -> list[CameraInfo]:
        """Probe a range of camera indices; report availability + name."""
        rng = probe_range or config.CAMERA_PROBE_RANGE
        results: list[CameraInfo] = []
        for i in range(rng):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    name = f"Camera {i}"
                    try:
                        backend = cap.getBackendName()  # type: ignore[attr-defined]
                        if backend:
                            name = f"{backend} #{i}"
                    except Exception:
                        pass
                    results.append(CameraInfo(id=i, name=name, available=True))
                    cap.release()
                else:
                    cap.release()
                    results.append(CameraInfo(id=i, name=f"Camera {i}", available=False))
            else:
                results.append(CameraInfo(id=i, name=f"Camera {i}", available=False))
        return results

    def acquire(self, camera_id: int) -> cv2.VideoCapture:
        """Long-lived acquire — caller MUST pair with :meth:`release`."""
        with self._lock:
            cap = self._captures.get(camera_id)
            if cap is None or not cap.isOpened():
                cap = self._open(camera_id)
                self._captures[camera_id] = cap
            return cap

    @contextmanager
    def use_temp(self, camera_id: int) -> Iterator[cv2.VideoCapture]:
        """Open the camera, yield a capture, then ALWAYS close it on exit.

        Use this for one-shot operations (snapshot, probe) so the OS
        releases the device and the LED turns off when the request ends.
        """
        cap = self._open(camera_id)
        try:
            yield cap
        finally:
            try:
                cap.release()
            except Exception:
                pass

    @staticmethod
    def _open(camera_id: int) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def release(self, camera_id: int) -> None:
        with self._lock:
            cap = self._captures.pop(camera_id, None)
            if cap is not None:
                cap.release()

    def release_all(self) -> None:
        with self._lock:
            for cap in self._captures.values():
                cap.release()
            self._captures.clear()

    def held(self) -> list[int]:
        with self._lock:
            return [cid for cid, cap in self._captures.items() if cap.isOpened()]


camera_manager = CameraManager()
