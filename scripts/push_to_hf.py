#!/usr/bin/env python
"""Upload a folder of artifacts to a Hugging Face model repo.

The token is read from the HF_TOKEN environment variable (on Kaggle: a Secret),
never from code or the command line.

Examples:
    python scripts/push_to_hf.py --repo-id andromedaboy/traffic-analytics-yolo11n \\
        --folder /kaggle/working/m2/artifacts --path-in-repo m2
    python scripts/push_to_hf.py --repo-id user/repo --folder outputs/m2/artifacts --quick   # dry run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from traffic_analytics.utils import get_logger

log = get_logger("push_to_hf")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True, help="e.g. andromedaboy/traffic-analytics-yolo11n")
    p.add_argument("--folder", type=Path, required=True, help="Local folder to upload.")
    p.add_argument("--path-in-repo", default="", help="Sub-folder in the repo (e.g. m2).")
    p.add_argument("--message", default=None, help="Commit message on the Hub.")
    p.add_argument("--quick", action="store_true", help="Dry run: list files and sizes, no network.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted(p for p in args.folder.rglob("*") if p.is_file())
    if not files:
        log.error("Nothing to upload in %s", args.folder)
        return 1
    for f in files:
        log.info("%10.2f MB  %s", f.stat().st_size / 1e6, f.relative_to(args.folder))
    target = f"{args.repo_id}/{args.path_in_repo}".rstrip("/")
    if args.quick:
        log.info("Dry run: would upload %d files to %s", len(files), target)
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.error("HF_TOKEN is not set.")
        return 1
    from huggingface_hub import HfApi

    info = HfApi(token=token).upload_folder(
        repo_id=args.repo_id,
        folder_path=str(args.folder),
        path_in_repo=args.path_in_repo or None,
        repo_type="model",
        commit_message=args.message or f"Upload {args.folder.name} to {args.path_in_repo or '/'}",
    )
    log.info("Uploaded %d files to %s: %s", len(files), target, getattr(info, "commit_url", info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
