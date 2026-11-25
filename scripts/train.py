#!/usr/bin/env python
"""Fine-tune YOLO11n on the M1 YOLO dataset and collect the M2 artifacts.

Steps: find the dataset → write a data.yaml pointing at it → train (Ultralytics)
→ re-validate best.pt on the val split → write metrics.json and copy best.pt and
charts into <out>/artifacts (the folder pushed to the Hugging Face model repo).

Metrics here are on the VAL split. The official test split is scored in M4.

Examples:
    python scripts/train.py --search /kaggle/input --out /kaggle/working/m2
    python scripts/train.py --search /kaggle/input --out /kaggle/working/m2_timing --epochs 1
    python scripts/train.py --data-root /tmp/tiny_yolo --out /tmp/m2q --quick
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import yaml

from traffic_analytics.detrac import write_data_yaml
from traffic_analytics.train_utils import (
    collect_artifacts,
    find_dataset_root,
    metrics_to_dict,
    read_results_csv,
    training_summary,
)
from traffic_analytics.utils import get_logger, run_metadata, set_seed

log = get_logger("train")

QUICK = {"epochs": 1, "imgsz": 160, "batch": 4, "device": "cpu", "workers": 0, "plots": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--data-root", type=Path, help="YOLO dataset dir (has data.yaml and images/train).")
    src.add_argument("--search", type=Path, default=Path("/kaggle/input"), help="Search here for the dataset dir.")
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--out", type=Path, default=Path("outputs/m2"), help="Run dir; artifacts go to <out>/artifacts.")
    p.add_argument("--assets-dir", type=Path, default=None, help="Also copy metrics.json + charts here (e.g. assets/).")
    for name, typ in (("epochs", int), ("batch", int), ("imgsz", int), ("workers", int), ("patience", int), ("fraction", float)):
        p.add_argument(f"--{name}", type=typ, default=None, help=f"Override '{name}' from the config.")
    p.add_argument("--device", default=None, help='Override device, e.g. "0,1", "0" or "cpu".')
    p.add_argument("--model", default=None, help="Override the starting weights (default from config).")
    p.add_argument("--quick", action="store_true", help="Smoke mode: 1 epoch, 160 px, batch 4, CPU.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO  # heavy import after --help

    cfg: dict = yaml.safe_load(args.config.read_text())
    overrides = {k: getattr(args, k) for k in ("epochs", "batch", "imgsz", "workers", "patience", "fraction", "device", "model")}
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if args.quick:
        cfg.update(QUICK)
    cfg.update(overrides)
    set_seed(int(cfg.get("seed", 0)))

    data_root = args.data_root or find_dataset_root(args.search)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_yaml = out / "data.yaml"
    write_data_yaml(data_yaml, data_root)  # M1's data.yaml points at /kaggle/working of the prep run
    log.info("Dataset: %s  config: %s", data_root, cfg)

    model_name = cfg.pop("model")
    model = YOLO(model_name)
    t0 = time.perf_counter()
    # Absolute project + exist_ok keeps the DDP worker writing to the same run dir.
    model.train(data=str(data_yaml), project=str(out), name="train", exist_ok=True, **cfg)
    train_seconds = time.perf_counter() - t0
    train_dir = out / "train"
    best = train_dir / "weights" / "best.pt"
    if not best.exists():
        log.error("No best.pt in %s", train_dir)
        return 1

    # Re-validate best.pt on one device so the per-class numbers come from a single, clean run.
    val_device = "cpu" if str(cfg.get("device")) == "cpu" else str(cfg.get("device", "0")).split(",")[0]
    vmetrics = YOLO(str(best)).val(
        data=str(data_yaml), split="val", imgsz=cfg["imgsz"], batch=cfg["batch"], device=val_device,
        plots=True, project=str(out), name="val_best", exist_ok=True,
    )

    art = out / "artifacts"
    missing = collect_artifacts(train_dir, out / "val_best", art)
    summary = training_summary(read_results_csv(train_dir / "results.csv"))
    try:
        import torch

        gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001
        gpus = []
    metrics = run_metadata(
        {
            "split": "val",
            "note": "Validation split (location-grouped, from official train). Official test is scored in M4.",
            "model_start": model_name,
            "data_root": str(data_root),
            "config": cfg,
            "gpus": gpus,
            "train_wall_time_s": round(train_seconds, 1),
            **summary,
            **metrics_to_dict(vmetrics),
            "missing_artifacts": missing,
        }
    )
    (art / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    log.info("Wrote %s", art / "metrics.json")
    log.info("VAL overall: %s", metrics["overall"])
    for row in metrics["per_class"]:
        log.info("VAL %-12s mAP50=%.4f mAP50-95=%.4f (instances=%d)", row["Class"], row["mAP50"], row["mAP50-95"], row["Instances"])

    if args.assets_dir:
        args.assets_dir.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "results.png", "PR_curve.png", "confusion_matrix_normalized.png"):
            if (art / name).exists():
                shutil.copy2(art / name, args.assets_dir / name)
        log.info("Copied metrics + charts to %s", args.assets_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
