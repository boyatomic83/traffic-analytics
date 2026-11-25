#!/usr/bin/env python
"""Check and record the runtime environment (Codespace, Kaggle, Space).

Prints Python/CPU/package info, checks that core imports work, reports whether
ffmpeg and a CUDA GPU are available, and optionally writes everything to JSON.

Examples:
    python scripts/env_check.py --quick
    python scripts/env_check.py --out assets/env_kaggle.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from traffic_analytics.utils import get_logger, run_metadata

log = get_logger("env_check")

# import name -> required for the CPU pipeline?
IMPORTS: dict[str, bool] = {
    "numpy": True,
    "cv2": True,
    "onnxruntime": True,
    "supervision": True,
    "yaml": True,
    "torch": False,
    "ultralytics": False,
}


def check_imports(names: dict[str, bool]) -> dict[str, str]:
    """Try importing each module; return ``{name: "ok" | "missing" | "error: ..."}``."""
    status: dict[str, str] = {}
    for name in names:
        try:
            importlib.import_module(name)
            status[name] = "ok"
        except ImportError:
            status[name] = "missing"
        except Exception as exc:  # broken install, e.g. ABI mismatch
            status[name] = f"error: {type(exc).__name__}: {exc}"
    return status


def gpu_info() -> dict[str, object]:
    """Return CUDA availability and device names (empty if torch is absent)."""
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "devices": []}
    ok = torch.cuda.is_available()
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if ok else []
    return {"cuda_available": ok, "devices": devices}


def onnxruntime_providers() -> list[str]:
    """Return the ONNX Runtime execution providers available here."""
    try:
        import onnxruntime as ort
    except ImportError:
        return []
    return list(ort.get_available_providers())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None, help="Write the report to this JSON file.")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smoke mode: skip the GPU probe (no CUDA init) and exit 0 even if required imports are missing.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    imports = check_imports(IMPORTS)
    report = run_metadata(
        {
            "imports": imports,
            "ffmpeg": shutil.which("ffmpeg"),
            "onnxruntime_providers": onnxruntime_providers(),
            "gpu": None if args.quick else gpu_info(),
        }
    )
    print(json.dumps(report, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        log.info("Wrote %s", args.out)

    missing = [n for n, required in IMPORTS.items() if required and imports[n] != "ok"]
    if missing:
        log.warning("Required imports not OK: %s", ", ".join(missing))
    if report["ffmpeg"] is None:
        log.warning("ffmpeg not found on PATH (needed from M5 for H.264 re-encode).")
    return 0 if (args.quick or not missing) else 1


if __name__ == "__main__":
    sys.exit(main())
