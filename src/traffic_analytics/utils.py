"""Shared helpers: logging, seeding and run metadata for artifact JSONs (SPEC §11)."""

from __future__ import annotations

import logging
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import numpy as np

#: Packages whose versions are recorded in every artifact JSON.
TRACKED_PACKAGES: tuple[str, ...] = (
    "traffic_analytics",
    "numpy",
    "opencv-python-headless",
    "opencv-python",
    "onnxruntime",
    "onnx",
    "supervision",
    "ultralytics",
    "torch",
    "gradio",
    "huggingface_hub",
    "pyyaml",
)

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str, level: int | str | None = None) -> logging.Logger:
    """Return a logger with a single stream handler.

    The level comes from ``level`` or the ``TA_LOG_LEVEL`` env var (default INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level or os.environ.get("TA_LOG_LEVEL", "INFO").upper())
    return logger


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Return installed versions of ``packages`` (``None`` if not installed)."""
    versions: dict[str, str | None] = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def cpu_model() -> str:
    """Return the CPU model name (from ``lscpu`` on Linux, else ``platform``)."""
    if shutil.which("lscpu"):
        try:
            out = subprocess.run(
                ["lscpu"], capture_output=True, text=True, timeout=10, check=False
            ).stdout
            for line in out.splitlines():
                if line.strip().startswith("Model name:"):
                    return line.split(":", 1)[1].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or platform.machine() or "unknown"


def run_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the metadata block that every artifact JSON must include.

    Contains a UTC timestamp, Python/platform info, CPU model and logical core
    count, and the versions of :data:`TRACKED_PACKAGES`.
    """
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }
    if extra:
        meta.update(extra)
    return meta
