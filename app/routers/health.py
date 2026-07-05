"""Health check endpoint — torch-free."""
from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from ..schemas import HealthInfo

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthInfo)
async def health(request: Request) -> HealthInfo:
    pipeline = request.app.state.pipeline
    cuda_available = False  # ONNX-only stack; CUDA via ORT is a separate concern
    gpu_name: str | None = None
    try:
        import onnxruntime as ort  # type: ignore

        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            gpu_name = "DirectML adapter (Windows GPU)"
        elif "CUDAExecutionProvider" in providers:
            gpu_name = "CUDA ExecutionProvider"
        else:
            gpu_name = None
        cuda_available = bool(gpu_name)
    except Exception:
        pass

    # Get the active provider that the pipeline is actually using
    active = ""
    try:
        active = pipeline.detector.active_provider or pipeline.classifier.active_provider
    except Exception:
        pass

    return HealthInfo(
        status="ok",
        version=__version__,
        device=active or "cpu",
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        models_loaded={
            "detector": pipeline.detector.is_loaded,
            "classifier": pipeline.classifier.is_loaded,
        },
    )
