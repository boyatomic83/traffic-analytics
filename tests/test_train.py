"""M2 tests: training helpers (pure) and a CPU end-to-end --quick run of scripts/train.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from traffic_analytics.train_utils import (
    collect_artifacts,
    find_dataset_root,
    read_results_csv,
    training_summary,
)

REPO = Path(__file__).resolve().parents[1]


def _make_yolo_dataset(root: Path, n: int = 8) -> Path:
    """Tiny YOLO dataset: grey images with one bright rectangle labelled as class 0..3."""
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.full((128, 160, 3), 90, np.uint8)
            x, y = int(rng.integers(10, 100)), int(rng.integers(10, 70))
            cv2.rectangle(img, (x, y), (x + 40, y + 30), (230, 230, 230), -1)
            cv2.imwrite(str(root / "images" / split / f"{i}.jpg"), img)
            cls = i % 4
            (root / "labels" / split / f"{i}.txt").write_text(f"{cls} {(x + 20) / 160:.6f} {(y + 15) / 128:.6f} {40 / 160:.6f} {30 / 128:.6f}\n")
    (root / "data.yaml").write_text("path: /nonexistent\ntrain: images/train\nval: images/val\nnames: {0: car, 1: van, 2: bus, 3: truck_other}\n")
    return root


# ------------------------------------------------------------------ helpers


def test_find_dataset_root_nested(tmp_path: Path) -> None:
    ds = _make_yolo_dataset(tmp_path / "notebooks" / "user" / "traffic-m1-data-prep" / "detrac_yolo", n=1)
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "data.yaml").write_text("x: 1\n")  # no images/train -> ignored
    assert find_dataset_root(tmp_path) == ds


def test_find_dataset_root_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_dataset_root(tmp_path)
    _make_yolo_dataset(tmp_path / "a", n=1)
    _make_yolo_dataset(tmp_path / "b", n=1)
    with pytest.raises(FileNotFoundError, match="Several"):
        find_dataset_root(tmp_path)


def test_training_summary_picks_best_epoch(tmp_path: Path) -> None:
    csv = tmp_path / "results.csv"
    csv.write_text("epoch,time,  metrics/mAP50-95(B)\n1,10.0,0.20\n2,20.0,0.35\n3,30.0,0.30\n")
    s = training_summary(read_results_csv(csv))
    assert s == {"epochs_run": 3, "best_epoch": 2, "best_metrics/mAP50-95(B)": 0.35, "csv_total_time_s": 30.0}


def test_collect_artifacts_renames_and_reports_missing(tmp_path: Path) -> None:
    train, val, out = tmp_path / "train", tmp_path / "val", tmp_path / "out"
    (train / "weights").mkdir(parents=True)
    val.mkdir()
    (train / "weights" / "best.pt").write_bytes(b"x")
    (train / "results.csv").write_text("epoch\n1\n")
    (val / "BoxPR_curve.png").write_bytes(b"png")
    missing = collect_artifacts(train, val, out)
    assert (out / "best.pt").exists() and (out / "PR_curve.png").exists()
    assert any(m.endswith("results.png") for m in missing)
    assert any(m.endswith("confusion_matrix_normalized.png") for m in missing)


# ------------------------------------------------------------------ scripts


def _run(script: str, *args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / script), *args], capture_output=True, text=True, timeout=timeout, cwd=REPO)


def test_train_help() -> None:
    proc = _run("train.py", "--help")
    assert proc.returncode == 0 and "--quick" in proc.stdout


def test_push_to_hf_dry_run(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text("{}")
    proc = _run("push_to_hf.py", "--repo-id", "someone/some-repo", "--folder", str(tmp_path), "--quick")
    assert proc.returncode == 0, proc.stderr
    assert "Dry run" in proc.stderr


def test_train_quick_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("ultralytics")
    ds = _make_yolo_dataset(tmp_path / "ds")
    out = tmp_path / "run"
    proc = _run("train.py", "--data-root", str(ds), "--out", str(out), "--quick", "--assets-dir", str(tmp_path / "assets"))
    assert proc.returncode == 0, proc.stderr[-3000:]
    art = out / "artifacts"
    for name in ("best.pt", "results.csv", "PR_curve.png", "confusion_matrix_normalized.png", "metrics.json"):
        assert (art / name).exists(), name
    m = json.loads((art / "metrics.json").read_text())
    assert m["split"] == "val" and m["epochs_run"] == 1
    assert set(m["overall"]) == {"precision", "recall", "mAP50", "mAP50-95"}
    assert {r["Class"] for r in m["per_class"]} <= {"car", "van", "bus", "truck_other"}
    assert "packages" in m and "timestamp_utc" in m
    assert (tmp_path / "assets" / "metrics.json").exists()
