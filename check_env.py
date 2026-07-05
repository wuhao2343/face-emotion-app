"""Environment check — verifies the ONNX-only stack is installed.

Run ``python check_env.py`` before starting the server to surface any
missing pieces early.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path


REQUIRED_PY = (3, 10)
RECOMMENDED_PY = (3, 12)

CHECKS: list[tuple[str, str]] = [
    ("onnxruntime", "onnxruntime"),
    ("Pillow",      "PIL"),
    ("cv2",         "cv2"),       # opencv-python
    ("numpy",       "numpy"),
    ("fastapi",     "fastapi"),
    ("uvicorn",     "uvicorn"),
    ("huggingface_hub", "huggingface_hub"),
]


def line(status: str, msg: str) -> None:
    icon = {"ok": "[OK]", "warn": "[!]", "fail": "[X]", "info": "[i]"}.get(status, "[?]")
    print(f"  {icon} {msg}")


def main() -> int:
    print("=" * 60)
    print("Face Emotion App — environment check (ONNX-only stack)")
    print("=" * 60)

    py = sys.version_info
    py_ok = py >= REQUIRED_PY
    py_rec = py >= RECOMMENDED_PY and py < (3, 13)
    line("ok" if py_ok else "fail",
         f"Python {py.major}.{py.minor}.{py.micro} "
         f"({'recommended' if py_rec else 'minimum OK' if py_ok else 'too old, need 3.10+'})")

    overall_ok = py_ok

    # ---- onnxruntime + available providers ------------------------------
    try:
        ort = importlib.import_module("onnxruntime")
        line("ok", f"onnxruntime {ort.__version__}")
        providers = ort.get_available_providers()
        line("info", f"Available ORT providers: {providers}")
        if "DmlExecutionProvider" in providers:
            line("ok", "DirectML available — GPU acceleration enabled by default")
        elif "CUDAExecutionProvider" in providers:
            line("ok", "CUDA EP available")
        else:
            line("info", "Running on CPU only (no GPU provider detected)")
    except ImportError:
        line("fail", "onnxruntime is NOT installed. Run: pip install -r requirements.txt")
        overall_ok = False

    # ---- Other packages --------------------------------------------------
    missing: list[str] = []
    for nice, mod in CHECKS:
        if mod == "onnxruntime":
            continue
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            line("ok", f"{nice} {ver}")
        except ImportError:
            line("fail", f"{nice} is NOT installed")
            missing.append(nice)
            overall_ok = False

    # ---- Models directory ------------------------------------------------
    models_dir = Path(__file__).resolve().parent / "models"
    if not models_dir.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        line("info", "Created models/ directory")
    for name in ("yolov8n-face.onnx", "trpakov-vit-face.onnx"):
        p = models_dir / name
        if p.exists():
            line("ok", f"{name} present ({p.stat().st_size // 1024 // 1024} MB)")
        else:
            line("warn", f"{name} not found; will be downloaded on first run")

    # ---- Summary ---------------------------------------------------------
    print("-" * 60)
    if overall_ok and not missing:
        print("[OK] Environment ready.")
        return 0
    if missing:
        print(f"[X] Missing: {', '.join(missing)}")
        print("    Run: pip install -r requirements.txt")
    if not overall_ok:
        print("[X] Environment not ready — see messages above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
