"""One-shot launcher.

Workflow:
  1. Locate Python 3.10+ (prefer 3.12)
  2. Ensure a project-local virtualenv exists at ``.venv/``
  3. Install requirements into the venv
  4. Run ``check_env.py`` to surface any remaining issues
  5. Launch uvicorn from inside the venv

Usage:
    python run.py
    python run.py --port 9000
    python run.py --reload         # dev hot-reload
    python run.py --recreate-venv  # nuke and rebuild the venv
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
PY_MIN = (3, 10)
PY_MAX_EXCLUSIVE = (3, 14)  # torch-less stack, so 3.14 is OK too; onnxruntime has wheels


def find_system_python() -> str:
    """Find a usable Python interpreter on the system."""
    candidates: list[str] = []

    # 1. Well-known Windows locations for 3.12 / 3.11 / 3.10
    if platform.system() == "Windows":
        home = Path.home()
        candidates += [
            str(home / "AppData/Local/Programs/Python/Python312/python.exe"),
            str(home / "AppData/Local/Programs/Python/Python311/python.exe"),
            str(home / "AppData/Local/Programs/Python/Python310/python.exe"),
            r"C:\Python312\python.exe",
            r"C:\Python311\python.exe",
            r"C:\Python310\python.exe",
        ]

    # 2. PATH
    from shutil import which
    for name in ("python3.12", "python3.11", "python3.10", "python3", "python"):
        w = which(name)
        if w:
            candidates.append(w)

    # 3. Current interpreter as last resort
    candidates.append(sys.executable)

    for c in candidates:
        if Path(c).exists() and _is_acceptable(c):
            return c
    return sys.executable  # last resort


def _is_acceptable(p: str) -> bool:
    try:
        out = subprocess.check_output(
            [p, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            stderr=subprocess.DEVNULL,
        )
        major, minor = (int(x) for x in out.decode().strip().split())
        return (major, minor) >= PY_MIN
    except Exception:
        return False


def _is_port_busy(host: str, port: int) -> bool:
    """Return True if anything is listening on ``host:port``."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _find_free_port(host: str, start: int, end: int) -> int | None:
    import socket
    for p in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return None


def venv_python() -> str:
    """Path to the python executable inside the project venv."""
    if platform.system() == "Windows":
        return str(VENV_DIR / "Scripts" / "python.exe")
    return str(VENV_DIR / "bin" / "python")


def ensure_venv(system_python: str, recreate: bool = False) -> str:
    """Create or reuse ``.venv``. Returns the venv's python path."""
    if recreate and VENV_DIR.exists():
        print(f"[i] Removing existing venv at {VENV_DIR}")
        import shutil
        shutil.rmtree(VENV_DIR, ignore_errors=True)

    if not VENV_DIR.exists():
        print(f"[i] Creating venv at {VENV_DIR} using {system_python}")
        subprocess.run(
            [system_python, "-m", "venv", str(VENV_DIR)], check=True
        )
    py = venv_python()
    if not Path(py).exists():
        raise RuntimeError(f"venv was created but {py} is missing")
    return py


def ensure_dependencies(venv_py: str) -> None:
    req = ROOT / "requirements.txt"
    if not req.exists():
        return
    # Probe: can we import all required packages?
    probe = subprocess.run(
        [venv_py, "-c",
         "import onnxruntime, cv2, fastapi, PIL, numpy, huggingface_hub"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode == 0:
        print(f"[i] Dependencies already installed in venv")
        return
    print(f"[i] Installing requirements into venv...")
    res = subprocess.run(
        [venv_py, "-m", "pip", "install", "--upgrade", "pip"],
    )
    if res.returncode != 0:
        print("[!] pip self-upgrade failed (continuing anyway)")
    res = subprocess.run(
        [venv_py, "-m", "pip", "install", "-r", str(req)],
    )
    if res.returncode != 0:
        print("[X] pip install failed. Run manually:")
        print(f"    {venv_py} -m pip install -r {req}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--recreate-venv", action="store_true",
                        help="Delete and rebuild the .venv before starting")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't open the browser automatically")
    args = parser.parse_args()

    system_py = find_system_python()
    print(f"[i] System Python: {system_py}")
    venv_py = ensure_venv(system_py, recreate=args.recreate_venv)
    print(f"[i] Venv Python:   {venv_py}")

    ensure_dependencies(venv_py)

    check = ROOT / "check_env.py"
    if check.exists():
        print()
        subprocess.run([venv_py, str(check)])

    if not args.no_browser and not args.reload:
        import threading, time, webbrowser
        def _open() -> None:
            time.sleep(2.5)
            url = f"http://127.0.0.1:{args.port}/"
            print(f"[i] Opening {url}")
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=_open, daemon=True).start()

    # Find a free port if the requested one is already in use.
    chosen_port = args.port
    if _is_port_busy(args.host, args.port):
        print(f"[!] Port {args.port} is busy, scanning for a free one...")
        free = _find_free_port(args.host, start=args.port + 1, end=args.port + 50)
        if free is None:
            print(f"[X] No free port found in [{args.port}..{args.port + 50}]")
            sys.exit(1)
        print(f"[i] Using port {free} instead")
        chosen_port = free

    cmd = [
        venv_py, "-m", "uvicorn", "app.main:app",
        "--host", args.host, "--port", str(chosen_port),
    ]
    if args.reload:
        cmd.append("--reload")

    print()
    print("=" * 60)
    print(f"Starting Face Emotion App at http://{args.host}:{args.port}")
    print("=" * 60)
    try:
        subprocess.run(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n[i] Shutting down.")


if __name__ == "__main__":
    main()
