"""Tests for UA-DETRAC → YOLO conversion on a tiny synthetic dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from traffic_analytics import detrac
from traffic_analytics.detrac import Box

REPO = Path(__file__).resolve().parents[1]
W, H = 192, 108
N_FRAMES = 12

# (sequence, official subset, location id)
SEQS = [
    ("MVI_10001", "Train", 0), ("MVI_10002", "Train", 0),
    ("MVI_20001", "Train", 1), ("MVI_20002", "Train", 1),
    ("MVI_30001", "Train", 2), ("MVI_30002", "Train", 2),
    ("MVI_40001", "Test", 3), ("MVI_40002", "Test", 3),
]
IGNORED = (150.0, 0.0, 40.0, 30.0)  # left, top, width, height (top-right corner)
PARTIAL = (130, 10, 30, 20)  # x 130-160 overlaps the ignore region; centre (145, 20) is outside


def _background(loc: int) -> np.ndarray:
    rng = np.random.default_rng(1000 + loc)
    small = rng.integers(0, 255, size=(9, 16), dtype=np.uint8)
    return cv2.cvtColor(cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)


def _xml(name: str, types: list[str]) -> str:
    frames = []
    for num in range(1, N_FRAMES + 1):
        x = 10 + 5 * num
        targets = [
            f'<target id="{i + 1}"><box left="{x + 40 * i}" top="50" width="30" height="20"/>'
            f'<attribute orientation="0" speed="1" trajectory_length="1" truncation_ratio="0" vehicle_type="{t}"/></target>'
            for i, t in enumerate(types)
        ]
        # kept box that overlaps the ignore region (centre outside) -> must stay visible
        targets.append(
            f'<target id="98"><box left="{PARTIAL[0]}" top="{PARTIAL[1]}" width="{PARTIAL[2]}" height="{PARTIAL[3]}"/>'
            f'<attribute vehicle_type="car"/></target>'
        )
        # one box whose centre is inside the ignore region -> must be dropped
        targets.append(
            f'<target id="99"><box left="160" top="5" width="10" height="10"/>'
            f'<attribute vehicle_type="car"/></target>'
        )
        frames.append(f'<frame density="{len(targets)}" num="{num}"><target_list>{"".join(targets)}</target_list></frame>')
    l, t, w, h = IGNORED
    return (
        f'<?xml version="1.0" encoding="utf-8"?><sequence name="{name}">'
        f'<sequence_attribute camera_state="stable" sence_weather="sunny"/>'
        f'<ignored_region><box left="{l}" top="{t}" width="{w}" height="{h}"/></ignored_region>'
        f'{"".join(frames)}</sequence>'
    )


@pytest.fixture(scope="module")
def fake_detrac(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("detrac")
    rng = np.random.default_rng(0)
    types_cycle = [["car", "van"], ["bus", "others"], ["car", "bus"], ["van", "others"]]
    for i, (name, subset, loc) in enumerate(SEQS):
        xml_dir = root / f"DETRAC-{subset}-Annotations-XML"
        img_dir = root / f"Insight-MVT_Annotation_{subset}" / name
        xml_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        (xml_dir / f"{name}.xml").write_text(_xml(name, types_cycle[i % 4]))
        bg = _background(loc)
        for num in range(1, N_FRAMES + 1):
            noise = rng.integers(-8, 8, size=bg.shape)
            frame = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(img_dir / f"img{num:05d}.jpg"), frame)
    return root


# ------------------------------------------------------------------ unit tests


def test_to_yolo_line_normalises() -> None:
    line = detrac.to_yolo_line(2, Box(10, 20, 50, 60), 100, 200)
    assert line == "2 0.300000 0.200000 0.400000 0.200000"


def test_to_yolo_line_clips_and_rejects_tiny() -> None:
    clipped = detrac.to_yolo_line(0, Box(-10, -10, 10, 10), 100, 100)
    assert clipped == "0 0.050000 0.050000 0.100000 0.100000"
    assert detrac.to_yolo_line(0, Box(99.5, 10, 120, 20), 100, 100) is None


def test_center_in_any() -> None:
    region = [Box(0, 0, 10, 10)]
    assert detrac.center_in_any(Box(2, 2, 6, 6), region)
    assert not detrac.center_in_any(Box(8, 8, 20, 20), region)  # overlaps but centre outside


def test_select_frames_stride() -> None:
    assert detrac.select_frames([5, 1, 3, 2, 4], stride=2) == [1, 3, 5]
    assert detrac.select_frames(list(range(1, 101)), stride=10, max_frames=3) == [1, 11, 21]


def test_parse_and_discover(fake_detrac: Path) -> None:
    seqs, problems = detrac.discover(fake_detrac)
    assert len(seqs) == len(SEQS)
    assert not any(problems.values())
    subsets = {s.name: s.subset for s in seqs}
    assert subsets["MVI_10001"] == "train" and subsets["MVI_40001"] == "test"
    ann = detrac.parse_sequence_xml(seqs[0].xml_path)
    assert len(ann.frames) == N_FRAMES
    assert ann.ignored == [Box.from_ltwh(*IGNORED)]
    assert ann.attrs["sence_weather"] == "sunny"


def test_inspect_summary(fake_detrac: Path) -> None:
    info = detrac.inspect_dataset(fake_detrac)
    assert info["format"] == "detrac_xml"
    assert info["per_subset"]["train"]["sequences"] == 6
    assert info["per_subset"]["test"]["sequences"] == 2
    assert info["unmapped_vehicle_types"] == []
    assert info["image_sizes"] == {f"{W}x{H}": len(SEQS)}


def test_location_grouping_recovers_cameras(fake_detrac: Path) -> None:
    seqs, _ = detrac.discover(fake_detrac)
    sigs = {s.name: detrac.location_signature(s.image_dir, list(range(1, N_FRAMES + 1))) for s in seqs}
    groups = detrac.group_by_location(sigs, threshold=0.75)
    expected = {name: loc for name, _, loc in SEQS}
    for a in expected:
        for b in expected:
            assert (groups[a] == groups[b]) == (expected[a] == expected[b]), (a, b)


def test_average_linkage_does_not_chain() -> None:
    # a~b = 0.71, b~c = 0.71, a~c = 0  -> single linkage would chain all three
    r2 = np.sqrt(2.0)
    sigs = {"MVI_11111": np.array([r2, 0.0]), "MVI_22222": np.array([1.0, 1.0]), "MVI_33333": np.array([0.0, r2])}
    groups = detrac.group_by_location(sigs, threshold=0.7)
    assert len(set(groups.values())) == 2
    assert groups["MVI_11111"] != groups["MVI_33333"]


def test_same_session_is_always_one_group() -> None:
    sigs = {"MVI_40901": np.array([1.0, -1.0]), "MVI_40902": np.array([-1.0, 1.0]), "MVI_50001": None}
    groups = detrac.group_by_location(sigs, threshold=0.9)
    assert groups["MVI_40901"] == groups["MVI_40902"] != groups["MVI_50001"]
    assert detrac.session_key("MVI_40901") == "MVI_4090"


def test_curated_links_merge_groups(tmp_path: Path) -> None:
    f = tmp_path / "links.yaml"
    f.write_text("links:\n  - [MVI_11111, MVI_33333, MVI_99999]\n")  # unknown name is ignored
    links = detrac.load_location_links(f)
    sigs = {"MVI_11111": np.array([1.0, -1.0]), "MVI_22222": np.array([1.0, -1.0]), "MVI_33333": np.array([-1.0, 1.0])}
    groups = detrac.group_by_location(sigs, threshold=0.99, links=links)
    assert groups["MVI_11111"] == groups["MVI_33333"]
    assert detrac.load_location_links(tmp_path / "missing.yaml") == []


def _split_inputs(root: Path):
    seqs, _ = detrac.discover(root)
    anns = {s.name: detrac.parse_sequence_xml(s.xml_path) for s in seqs}
    counts = {n: detrac.sequence_class_counts(a) for n, a in anns.items()}
    groups = {name: loc for name, _, loc in SEQS}
    return seqs, counts, groups


def test_official_split_and_group_val(fake_detrac: Path) -> None:
    seqs, counts, groups = _split_inputs(fake_detrac)
    sp = detrac.make_splits(seqs, counts, groups, val_frac=0.3, seed=0)
    assert sp["strategy"].startswith("official_test")
    assert sp["test"] == ["MVI_40001", "MVI_40002"]
    assert sp["val"] and set(sp["val"]).isdisjoint(sp["train"])
    assert set(sp["train"]) | set(sp["val"]) == {n for n, s, _ in SEQS if s == "Train"}
    assert sp["location_leakage"]["train_val_shared_groups"] == 0
    assert detrac.make_splits(seqs, counts, groups, val_frac=0.3, seed=0) == sp  # deterministic


def test_fallback_split_without_official_subsets(fake_detrac: Path) -> None:
    seqs, counts, groups = _split_inputs(fake_detrac)
    for s in seqs:
        s.subset = "unknown"
    sp = detrac.make_splits(seqs, counts, groups, val_frac=0.25, test_frac=0.25, seed=0)
    assert sp["strategy"] == "grouped_random_80_10_10"
    parts = [set(sp["train"]), set(sp["val"]), set(sp["test"])]
    assert all(parts) and sum(map(len, parts)) == len(SEQS) and len(set.union(*parts)) == len(SEQS)
    assert sp["location_leakage"] == {
        "train_val_shared_groups": 0, "train_test_shared_groups": 0, "test_sequences_sharing_location_with_train": 0
    }


def test_convert_blacks_out_ignore_and_drops_boxes(fake_detrac: Path, tmp_path: Path) -> None:
    seqs, _ = detrac.discover(fake_detrac)
    seq = seqs[0]
    ann = detrac.parse_sequence_xml(seq.xml_path)
    st = detrac.convert_sequence(seq, ann, "train", tmp_path, stride=4)
    assert st["images"] == 3  # frames 1, 5, 9
    assert st["dropped_ignored"] == 3  # one per frame
    assert sum(st["boxes"]) == 9  # 2 typed + 1 partial-overlap car, per frame
    img = cv2.imread(str(tmp_path / "images" / "train" / f"{seq.name}_img00001.jpg")).astype(int)
    src = cv2.imread(str(seq.image_dir / "img00001.jpg")).astype(int)
    l, t, w, h = map(int, IGNORED)
    # ignore region is black except where it overlaps the kept box (x 150-160, y 10-30)
    assert img[t + 2 : t + h - 2, 162 : l + w - 2].max() <= 10
    assert img[0:8, l + 2 : l + w - 2].max() <= 10
    overlap = (slice(12, 28), slice(151, 159))
    assert np.abs(img[overlap] - src[overlap]).mean() < 8  # restored, not black
    lines = (tmp_path / "labels" / "train" / f"{seq.name}_img00001.txt").read_text().splitlines()
    assert len(lines) == 3 and all(len(x.split()) == 5 for x in lines)


def test_convert_rejects_unmapped_type(fake_detrac: Path, tmp_path: Path) -> None:
    seqs, _ = detrac.discover(fake_detrac)
    ann = detrac.parse_sequence_xml(seqs[0].xml_path)
    ann.frames[1].append(("motorcycle", Box(0, 0, 10, 10)))
    with pytest.raises(ValueError, match="motorcycle"):
        detrac.convert_sequence(seqs[0], ann, "train", tmp_path, stride=1)


# ------------------------------------------------------------------ script tests


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / "prepare_data.py"), *args],
        capture_output=True, text=True, timeout=300,
    )


def test_script_help() -> None:
    proc = _run("--help")
    assert proc.returncode == 0 and "--quick" in proc.stdout and "--inspect-only" in proc.stdout


def test_script_inspect_only(fake_detrac: Path, tmp_path: Path) -> None:
    proc = _run("--src", str(fake_detrac), "--out", str(tmp_path), "--inspect-only")
    assert proc.returncode == 0, proc.stderr
    assert json.loads((tmp_path / "inspect.json").read_text())["inspect"]["sequences_total"] == len(SEQS)
    assert not (tmp_path / "images").exists()


def test_script_quick_end_to_end(fake_detrac: Path, tmp_path: Path) -> None:
    out = tmp_path / "yolo"
    proc = _run("--src", str(fake_detrac), "--out", str(out), "--quick", "--train-stride", "1", "--eval-stride", "1")
    assert proc.returncode == 0, proc.stderr
    cfg = yaml.safe_load((out / "data.yaml").read_text())
    assert cfg["names"] == {0: "car", 1: "van", 2: "bus", 3: "truck_other"}
    for sub in ("train", "val", "test"):
        imgs = sorted((out / cfg[sub]).glob("*.jpg"))
        assert imgs, sub
        assert all((out / "labels" / sub / f"{p.stem}.txt").exists() for p in imgs)
    splits = json.loads((out / "splits.json").read_text())
    assert set(splits["train"]).isdisjoint(splits["val"]) and set(splits["train"]).isdisjoint(splits["test"])
    assert "packages" in splits and "timestamp_utc" in splits
    assert len(list((out / "viz").glob("*.jpg"))) == 6  # 5 overlays + location sheet
    # second run refuses to clobber without --overwrite
    assert _run("--src", str(fake_detrac), "--out", str(out), "--quick").returncode == 1
