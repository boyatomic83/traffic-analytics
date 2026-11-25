"""M0 smoke tests: the package installs, imports and its helpers work on CPU."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import traffic_analytics
from traffic_analytics.utils import package_versions, run_metadata, set_seed

REPO = Path(__file__).resolve().parents[1]


def test_package_version() -> None:
    assert traffic_analytics.__version__ == "0.1.0"


def test_run_metadata_keys() -> None:
    meta = run_metadata({"run": "test"})
    for key in ("timestamp_utc", "python", "platform", "cpu_model", "cpu_count", "packages", "run"):
        assert key in meta
    assert meta["packages"]["traffic_analytics"] == "0.1.0"
    json.dumps(meta)  # must be JSON-serialisable


def test_package_versions_missing_is_none() -> None:
    assert package_versions(("definitely-not-a-real-package-xyz",)) == {
        "definitely-not-a-real-package-xyz": None
    }


def test_set_seed_is_deterministic() -> None:
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_core_imports() -> None:
    """The CPU pipeline deps must import (catches e.g. opencv-python needing libGL)."""
    import cv2  # noqa: F401
    import onnxruntime  # noqa: F401
    import supervision  # noqa: F401
    import yaml  # noqa: F401


def test_env_check_quick(tmp_path: Path) -> None:
    out = tmp_path / "env.json"
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "env_check.py"), "--quick", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["imports"]["numpy"] == "ok"


def test_env_check_help() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "env_check.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--quick" in proc.stdout
