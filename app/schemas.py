"""Pydantic response schemas for the API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Detection(BaseModel):
    bbox: BBox
    emotion: str
    score: float = Field(ge=0.0, le=1.0)
    all_scores: dict[str, float]


class FrameResult(BaseModel):
    count: int
    detections: list[Detection]
    inference_ms: float
    device: str


class CameraInfo(BaseModel):
    id: int
    name: str
    available: bool


class HealthInfo(BaseModel):
    status: str
    service: str = "face-emotion-app"
    version: str
    device: str
    cuda_available: bool
    gpu_name: str | None = None
    models_loaded: dict[str, bool]


class VideoInfo(BaseModel):
    filename: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_s: float


class VideoTaskStatus(BaseModel):
    task_id: str
    status: str  # queued | processing | done | error
    progress: float = Field(ge=0.0, le=1.0)
    message: str = ""
    download_url: str | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
