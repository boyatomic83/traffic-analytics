#!/usr/bin/env python
"""Convert UA-DETRAC (original XML release) into a YOLO dataset.

Steps: inspect → parse → group sequences by camera location → split by sequence
→ subsample frames → black out ignore regions → write images/labels, data.yaml,
splits.json, a summary JSON and visual checks.

Split strategy:
  * official train AND test annotations found: test = official test sequences,
    val = ~val-frac of official train sequences, chosen by whole location groups.
  * otherwise: group-aware ~80/10/10 split of all sequences.

Examples:
    python scripts/prepare_data.py --src /kaggle/input/ua-detrac-orig --inspect-only
    python scripts/prepare_data.py --src /kaggle/input/ua-detrac-orig --out /kaggle/working/detrac_yolo
    python scripts/prepare_data.py --src /path/to/detrac --out /tmp/dq --quick
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from traffic_analytics import detrac
from traffic_analytics.utils import get_logger, run_metadata, set_seed

log = get_logger("prepare_data")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="Root of the attached UA-DETRAC dataset.")
    p.add_argument("--out", type=Path, default=Path("datasets/detrac_yolo"), help="Output YOLO dataset dir.")
    p.add_argument("--data-yaml", type=Path, default=None, help="Default: configs/data.yaml (quick: <out>/data.yaml).")
    p.add_argument("--splits", type=Path, default=None, help="Default: configs/splits.json (quick: <out>/splits.json).")
    p.add_argument("--summary", type=Path, default=None, help="Default: assets/data_summary.json (quick: <out>/data_summary.json).")
    p.add_argument("--train-stride", type=int, default=10, help="Keep every Nth annotated frame in train.")
    p.add_argument("--eval-stride", type=int, default=10, help="Keep every Nth annotated frame in val/test.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1, help="Only used without an official test set.")
    p.add_argument("--loc-threshold", type=float, default=0.75, help="Background NCC to merge sequences into one location.")
    p.add_argument("--location-links", type=Path, default=Path("configs/location_links.yaml"),
                   help="YAML of curated same-location sequence lists (missing file = no links).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4, help="Threads for image I/O.")
    p.add_argument("--viz", type=int, default=5, help="Number of label-overlay images to draw.")
    p.add_argument("--viz-dir", type=Path, default=None, help="Default: <out>/viz.")
    p.add_argument("--inspect-only", action="store_true", help="Print the dataset summary and exit.")
    p.add_argument("--overwrite", action="store_true", help="Replace existing <out>/images and <out>/labels.")
    p.add_argument("--quick", action="store_true", help="Smoke mode: ≤4 sequences, ≤10 frames each; outputs stay in <out>.")
    return p.parse_args(argv)


def pick_quick(seqs: list[detrac.SequenceInfo]) -> list[detrac.SequenceInfo]:
    """≤3 train + ≤1 test sequences if official subsets exist, else the first 4."""
    train = [s for s in seqs if s.subset == "train"][:3]
    test = [s for s in seqs if s.subset == "test"][:1]
    return train + test if train and test else seqs[:4]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    out: Path = args.out
    in_out = lambda name: out / name  # noqa: E731
    data_yaml = args.data_yaml or (in_out("data.yaml") if args.quick else Path("configs/data.yaml"))
    splits_path = args.splits or (in_out("splits.json") if args.quick else Path("configs/splits.json"))
    summary_path = args.summary or (in_out("data_summary.json") if args.quick else Path("assets/data_summary.json"))
    viz_dir = args.viz_dir or in_out("viz")

    # 1. Inspect ------------------------------------------------------------
    log.info("Inspecting %s ...", args.src)
    info = detrac.inspect_dataset(args.src)
    print(json.dumps(info, indent=2))
    if info["format"] != "detrac_xml":
        log.error(
            "Format %r is not the original UA-DETRAC XML release. A sequence-level split needs "
            "sequence IDs per frame; pick a mirror with the XML annotations.", info["format"]
        )
        return 2
    if args.inspect_only:
        out.mkdir(parents=True, exist_ok=True)
        (out / "inspect.json").write_text(json.dumps(run_metadata({"inspect": info}), indent=2) + "\n")
        log.info("Wrote %s", out / "inspect.json")
        return 0
    if info["unmapped_vehicle_types"]:
        log.error("Unmapped vehicle types %s; extend VEHICLE_TYPE_TO_ID.", info["unmapped_vehicle_types"])
        return 2

    # 2. Parse --------------------------------------------------------------
    seqs, problems = detrac.discover(args.src)
    if args.quick:
        seqs = pick_quick(seqs)
    anns = {s.name: detrac.parse_sequence_xml(s.xml_path) for s in seqs}
    seqs = [s for s in seqs if anns[s.name].frames]
    log.info("Using %d sequences (%s).", len(seqs), dict(Counter(s.subset for s in seqs)))

    # 3. Location groups ----------------------------------------------------
    sigs = {s.name: detrac.location_signature(s.image_dir, list(anns[s.name].frames)) for s in seqs}
    links = detrac.load_location_links(args.location_links)
    log.info("Curated location links: %d lists from %s", len(links), args.location_links)
    groups = detrac.group_by_location(sigs, args.loc_threshold, links)
    group_sizes = Counter(groups.values())
    log.info("Location groups: %d (sizes: %s)", len(group_sizes), sorted(group_sizes.values(), reverse=True))

    # 4. Split --------------------------------------------------------------
    counts = {n: detrac.sequence_class_counts(a) for n, a in anns.items()}
    split = detrac.make_splits(seqs, counts, groups, args.val_frac, args.test_frac, args.seed)
    log.info(
        "Split %s: train=%d val=%d test=%d sequences; leakage=%s",
        split["strategy"], len(split["train"]), len(split["val"]), len(split["test"]), split["location_leakage"],
    )

    # 5. Convert ------------------------------------------------------------
    for sub in ("images", "labels"):
        d = out / sub
        if d.exists() and any(d.iterdir()):
            if not args.overwrite:
                log.error("%s is not empty; pass --overwrite to replace it.", d)
                return 1
            shutil.rmtree(d)
    by_name = {s.name: s for s in seqs}
    max_frames = 10 if args.quick else None
    split_stats: dict[str, dict] = {}
    for split_name in ("train", "val", "test"):
        stride = args.train_stride if split_name == "train" else args.eval_stride
        agg = {"sequences": len(split[split_name]), "images": 0, "boxes": np.zeros(len(detrac.CLASS_NAMES), np.int64),
               "dropped_ignored": 0, "dropped_small": 0, "missing_images": 0}
        for n in split[split_name]:
            st = detrac.convert_sequence(by_name[n], anns[n], split_name, out, stride, max_frames, args.workers)
            agg["images"] += st["images"]
            agg["boxes"] += np.asarray(st["boxes"])
            for k in ("dropped_ignored", "dropped_small", "missing_images"):
                agg[k] += st[k]
        agg["boxes"] = dict(zip(detrac.CLASS_NAMES, agg["boxes"].tolist()))
        split_stats[split_name] = agg
        log.info("%-5s stride=%d images=%d boxes=%s", split_name, stride, agg["images"], agg["boxes"])

    # 6. Outputs ------------------------------------------------------------
    detrac.write_data_yaml(data_yaml, out)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    splits_doc = run_metadata(
        {
            "strategy": split["strategy"],
            "params": params,
            "train": split["train"], "val": split["val"], "test": split["test"],
            "location_groups": groups,
            "location_leakage": split["location_leakage"],
        }
    )
    summary_doc = run_metadata(
        {
            "class_names": list(detrac.CLASS_NAMES),
            "note": "UA-DETRAC has no motorcycle class; 'others' is mapped to truck_other.",
            "inspect": info,
            "discovery_problems": problems,
            "strategy": split["strategy"],
            "location_groups": len(group_sizes),
            "location_leakage": split["location_leakage"],
            "splits": split_stats,
            "params": params,
        }
    )
    for path, doc in ((splits_path, splits_doc), (summary_path, summary_doc)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2) + "\n")
        log.info("Wrote %s", path)
    log.info("Wrote %s", data_yaml)

    # 7. Visual checks: draw from the label files we wrote (round trip) ------
    viz_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted((out / "images").rglob("*.jpg"))
    rng = random.Random(args.seed)
    for img_path in rng.sample(imgs, min(args.viz, len(imgs))):
        lbl = out / "labels" / img_path.parent.name / f"{img_path.stem}.txt"
        cv2.imwrite(str(viz_dir / f"{img_path.parent.name}_{img_path.name}"), detrac.draw_yolo_labels(img_path, lbl))
    first = {n: min(a.frames) for n, a in anns.items()}
    cv2.imwrite(str(viz_dir / "location_groups.jpg"), detrac.location_contact_sheet(seqs, groups, first))
    log.info("Wrote %d overlays and location_groups.jpg to %s", min(args.viz, len(imgs)), viz_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
