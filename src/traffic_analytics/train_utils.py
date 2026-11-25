"""Helpers for scripts/train.py: dataset discovery, artifact collection, metric extraction.

Kept free of training itself so they can be unit-tested on CPU without a GPU.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from traffic_analytics.utils import get_logger

log = get_logger(__name__)

#: Ultralytics output file (relative to its run dir) → name we publish (SPEC §6).
ARTIFACT_MAP: dict[str, str] = {
    "weights/best.pt": "best.pt",
    "results.png": "results.png",
    "results.csv": "results.csv",
    "args.yaml": "args.yaml",
}
#: Plots taken from the post-training validation of best.pt.
VAL_PLOT_MAP: dict[str, str] = {
    "BoxPR_curve.png": "PR_curve.png",  # Ultralytics 8.4 prefixes plot names with "Box"
    "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
}


def find_dataset_root(search_root: Path, max_depth: int = 5) -> Path:
    """Find the YOLO dataset dir (has ``data.yaml`` and ``images/train``) under ``search_root``.

    Searches breadth-first up to ``max_depth`` levels so a large attached raw dataset
    is not walked file by file. Raises ``FileNotFoundError`` if none or several match.
    """
    search_root = Path(search_root)
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth + ["data.yaml"])
        hits = sorted(p.parent for p in search_root.glob(pattern) if (p.parent / "images" / "train").is_dir())
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError(f"Several YOLO datasets under {search_root}: {hits}; pass --data-root.")
    raise FileNotFoundError(f"No YOLO dataset (data.yaml + images/train) under {search_root}.")


def collect_artifacts(train_dir: Path, val_dir: Path | None, out_dir: Path) -> list[str]:
    """Copy published artifacts into ``out_dir`` under their SPEC names; return missing sources."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    sources = [(Path(train_dir), ARTIFACT_MAP)] + ([(Path(val_dir), VAL_PLOT_MAP)] if val_dir else [])
    for base, mapping in sources:
        for src, dst in mapping.items():
            if (base / src).exists():
                shutil.copy2(base / src, out_dir / dst)
            else:
                missing.append(str(base / src))
    if missing:
        log.warning("Missing artifacts: %s", missing)
    return missing


def read_results_csv(path: Path) -> list[dict[str, float]]:
    """Read Ultralytics ``results.csv`` (column names are stripped of padding)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): float(v) for k, v in r.items() if v not in (None, "")} for r in rows]


def training_summary(rows: list[dict[str, float]], key: str = "metrics/mAP50-95(B)") -> dict[str, Any]:
    """Epochs run, best epoch by ``key`` (1-based), and total time if the CSV logs it."""
    if not rows:
        return {"epochs_run": 0, "best_epoch": None}
    best_i = max(range(len(rows)), key=lambda i: rows[i].get(key, float("-inf")))
    out: dict[str, Any] = {
        "epochs_run": len(rows),
        "best_epoch": int(rows[best_i].get("epoch", best_i + 1)),
        f"best_{key}": rows[best_i].get(key),
    }
    if "time" in rows[-1]:
        out["csv_total_time_s"] = rows[-1]["time"]
    return out


def metrics_to_dict(metrics: Any) -> dict[str, Any]:
    """Overall and per-class numbers from an Ultralytics ``DetMetrics`` (``model.val()`` result)."""
    rd = metrics.results_dict
    return {
        "overall": {
            "precision": float(rd["metrics/precision(B)"]),
            "recall": float(rd["metrics/recall(B)"]),
            "mAP50": float(rd["metrics/mAP50(B)"]),
            "mAP50-95": float(rd["metrics/mAP50-95(B)"]),
        },
        "per_class": metrics.summary(),
    }
