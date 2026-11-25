"""UA-DETRAC → YOLO conversion: discovery, XML parsing, location grouping, splits.

The original UA-DETRAC release ships one XML file per sequence::

    <sequence name="MVI_20011">
      <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
      <ignored_region><box left=".." top=".." width=".." height=".."/></ignored_region>
      <frame density="7" num="1">
        <target_list>
          <target id="1">
            <box left=".." top=".." width=".." height=".."/>
            <attribute vehicle_type="car" .../>
          </target>
        </target_list>
      </frame>
    </sequence>

and images in ``.../MVI_20011/img00001.jpg``. Nothing here hard-codes a Kaggle path:
every function takes the dataset root as an argument.
"""

from __future__ import annotations

import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from traffic_analytics.utils import get_logger

log = get_logger(__name__)

CLASS_NAMES: tuple[str, ...] = ("car", "van", "bus", "truck_other")
#: UA-DETRAC ``vehicle_type`` → our class id. UA-DETRAC has no motorcycles.
VEHICLE_TYPE_TO_ID: dict[str, int] = {"car": 0, "van": 1, "bus": 2, "others": 3}

_SEQ_RE = re.compile(r"^MVI_\d+$")
_IMG_EXTS = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_ltwh(cls, left: float, top: float, width: float, height: float) -> Box:
        return cls(left, top, left + width, top + height)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass
class SequenceAnn:
    """Parsed annotations of one sequence."""

    name: str
    frames: dict[int, list[tuple[str, Box]]]  # frame num → [(vehicle_type, box)]
    ignored: list[Box] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class SequenceInfo:
    """Where a sequence's annotation and images live, and its official subset."""

    name: str
    subset: str  # "train", "test" or "unknown"
    xml_path: Path
    image_dir: Path


# ---------------------------------------------------------------------- discovery


def _box_from_elem(elem: ET.Element) -> Box:
    a = elem.attrib
    return Box.from_ltwh(float(a["left"]), float(a["top"]), float(a["width"]), float(a["height"]))


def _is_sequence_xml(path: Path) -> bool:
    try:
        for _, elem in ET.iterparse(path, events=("start",)):
            return elem.tag == "sequence"
    except ET.ParseError:
        return False
    return False


def find_annotation_xmls(root: Path) -> list[Path]:
    """Return all UA-DETRAC sequence XML files under ``root`` (root tag ``<sequence>``)."""
    return sorted(p for p in Path(root).rglob("*.xml") if _is_sequence_xml(p))


def find_image_dirs(root: Path) -> dict[str, list[Path]]:
    """Map sequence name (``MVI_12345``) → directories under ``root`` holding its frames."""
    found: dict[str, list[Path]] = {}
    for d in sorted(Path(root).rglob("MVI_*")):
        if d.is_dir() and _SEQ_RE.match(d.name) and any(
            f.suffix.lower() in _IMG_EXTS for f in d.iterdir()
        ):
            found.setdefault(d.name, []).append(d)
    return found


def infer_subset(path: Path, root: Path) -> str:
    """Infer the official subset from folder names (``...Train...`` / ``...Test...``)."""
    rel = str(Path(path).relative_to(root)).lower()
    has_train, has_test = "train" in rel, "test" in rel
    if has_train and not has_test:
        return "train"
    if has_test and not has_train:
        return "test"
    return "unknown"


def detect_format(root: Path) -> str:
    """Return ``"detrac_xml"``, ``"yolo"`` or ``"unknown"`` for the dataset at ``root``."""
    root = Path(root)
    if find_annotation_xmls(root):
        return "detrac_xml"
    if any(p.is_dir() and p.name == "labels" for p in root.rglob("labels")) and any(
        root.rglob("*.txt")
    ):
        return "yolo"
    return "unknown"


def parse_sequence_xml(path: Path) -> SequenceAnn:
    """Parse one UA-DETRAC sequence XML file."""
    tree = ET.parse(path)
    seq = tree.getroot()
    name = seq.attrib.get("name", Path(path).stem)
    attrs_elem = seq.find("sequence_attribute")
    attrs = dict(attrs_elem.attrib) if attrs_elem is not None else {}
    ignored = [_box_from_elem(b) for b in seq.findall("./ignored_region/box")]
    frames: dict[int, list[tuple[str, Box]]] = {}
    for fr in seq.findall("frame"):
        num = int(fr.attrib["num"])
        targets = []
        for t in fr.findall("./target_list/target"):
            box_e, attr_e = t.find("box"), t.find("attribute")
            if box_e is None:
                continue
            vtype = attr_e.attrib.get("vehicle_type", "unknown") if attr_e is not None else "unknown"
            targets.append((vtype, _box_from_elem(box_e)))
        frames[num] = targets
    return SequenceAnn(name=name, frames=frames, ignored=ignored, attrs=attrs)


def discover(root: Path) -> tuple[list[SequenceInfo], dict[str, list[str]]]:
    """Pair every annotation XML with its image directory.

    Returns ``(sequences, problems)`` where ``problems`` lists sequence names with
    annotations but no images, images but no annotations, and duplicate image dirs.
    """
    root = Path(root)
    xmls = find_annotation_xmls(root)
    img_dirs = find_image_dirs(root)
    problems: dict[str, list[str]] = {"xml_without_images": [], "images_without_xml": [], "duplicate_image_dirs": []}
    seqs: list[SequenceInfo] = []
    xml_names: set[str] = set()
    for x in xmls:
        name = x.stem.split(".")[0]
        if name in xml_names:
            log.warning("Duplicate annotation for %s at %s; keeping the first.", name, x)
            continue
        xml_names.add(name)
        dirs = img_dirs.get(name)
        if not dirs:
            problems["xml_without_images"].append(name)
            continue
        if len(dirs) > 1:
            problems["duplicate_image_dirs"].append(name)
        seqs.append(SequenceInfo(name, infer_subset(x, root), x, dirs[0]))
    problems["images_without_xml"] = sorted(set(img_dirs) - xml_names)
    return seqs, problems


def frame_path(image_dir: Path, num: int) -> Path:
    """Return the image path for frame ``num`` (UA-DETRAC naming: ``img00001.jpg``)."""
    return Path(image_dir) / f"img{num:05d}.jpg"


# ------------------------------------------------------------------- inspection


def inspect_dataset(root: Path) -> dict:
    """Summarise the dataset at ``root`` without converting anything."""
    root = Path(root)
    fmt = detect_format(root)
    summary: dict = {"root": str(root), "format": fmt}
    if fmt != "detrac_xml":
        return summary
    seqs, problems = discover(root)
    per_subset: dict[str, dict] = {}
    vtypes: Counter = Counter()
    weather: Counter = Counter()
    camera: Counter = Counter()
    sizes: Counter = Counter()
    for s in seqs:
        ann = parse_sequence_xml(s.xml_path)
        sub = per_subset.setdefault(
            s.subset,
            {"sequences": 0, "annotated_frames": 0, "image_files": 0, "boxes": Counter(), "ignored_regions": 0},
        )
        sub["sequences"] += 1
        sub["annotated_frames"] += len(ann.frames)
        sub["image_files"] += sum(1 for f in s.image_dir.iterdir() if f.suffix.lower() in _IMG_EXTS)
        sub["ignored_regions"] += len(ann.ignored)
        for targets in ann.frames.values():
            for vt, _ in targets:
                sub["boxes"][vt] += 1
                vtypes[vt] += 1
        weather[ann.attrs.get("sence_weather", "n/a")] += 1
        camera[ann.attrs.get("camera_state", "n/a")] += 1
        first = min(ann.frames) if ann.frames else 1
        img = cv2.imread(str(frame_path(s.image_dir, first)))
        sizes[f"{img.shape[1]}x{img.shape[0]}" if img is not None else "unreadable"] += 1
    for sub in per_subset.values():
        sub["boxes"] = dict(sub["boxes"])
    summary.update(
        {
            "sequences_total": len(seqs),
            "per_subset": per_subset,
            "vehicle_types": dict(vtypes),
            "unmapped_vehicle_types": sorted(set(vtypes) - set(VEHICLE_TYPE_TO_ID)),
            "weather": dict(weather),
            "camera_state": dict(camera),
            "image_sizes": dict(sizes),
            "problems": {k: v for k, v in problems.items()},
        }
    )
    return summary


# ------------------------------------------------------------- location grouping


def location_signature(image_dir: Path, frame_nums: list[int], n_frames: int = 5, size: tuple[int, int] = (96, 54)) -> np.ndarray | None:
    """Background "fingerprint" of a sequence's camera view.

    Takes the pixel-wise median of ``n_frames`` evenly spaced frames (removes moving
    vehicles), equalises and blurs it, and returns it standardised (zero mean, unit
    variance) so a dot product / N is the normalised cross-correlation.
    """
    if not frame_nums:
        return None
    nums = sorted(frame_nums)
    idx = np.linspace(0, len(nums) - 1, num=min(n_frames, len(nums))).round().astype(int)
    stack = []
    for i in idx:
        img = cv2.imread(str(frame_path(image_dir, nums[i])), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            stack.append(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
    if not stack:
        return None
    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    bg = cv2.GaussianBlur(cv2.equalizeHist(bg), (5, 5), 0).astype(np.float32)
    return (bg - bg.mean()) / (bg.std() + 1e-6)


def session_key(name: str) -> str:
    """Recording-session key: the sequence ID without its last digit (``MVI_40901`` → ``MVI_4090``).

    In UA-DETRAC, IDs that differ only in the last digit were recorded back to
    back at the same place, so they are always kept in one location group.
    """
    return name[:-1]


def group_by_location(signatures: dict[str, np.ndarray | None], threshold: float = 0.75) -> dict[str, int]:
    """Group sequences filmed at the same camera location.

    1. Must-link: sequences with the same :func:`session_key` start in one cluster.
    2. Average-linkage agglomeration: repeatedly merge the two clusters whose mean
       pairwise background NCC is highest, while it is ≥ ``threshold``. Average
       (not single) linkage prevents chaining A~B~C into one group when A and C
       look nothing alike.

    Sequences without a signature only join via their session. Group ids are
    0..G-1 in order of each group's first (sorted) sequence name.
    """
    names = sorted(signatures)
    clusters: dict[str, list[str]] = {}
    for n in names:
        clusters.setdefault(session_key(n), []).append(n)
    cl = list(clusters.values())

    def sim(a: str, b: str) -> float | None:
        sa, sb = signatures[a], signatures[b]
        return None if sa is None or sb is None else float((sa * sb).mean())

    pair = {(a, b): sim(a, b) for i, a in enumerate(names) for b in names[i + 1 :]}

    def link(c1: list[str], c2: list[str]) -> float:
        vals = [pair[(a, b) if a < b else (b, a)] for a in c1 for b in c2]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else -1.0

    while len(cl) > 1:
        best, bi, bj = -2.0, -1, -1
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                s = link(cl[i], cl[j])
                if s > best:
                    best, bi, bj = s, i, j
        if best < threshold:
            break
        cl[bi] = cl[bi] + cl[bj]
        del cl[bj]

    cl.sort(key=lambda c: min(c))
    return {n: gid for gid, c in enumerate(cl) for n in c}


# ------------------------------------------------------------------------ splits


def sequence_class_counts(ann: SequenceAnn) -> np.ndarray:
    """Box counts per our class id over all annotated frames (unmapped types skipped)."""
    c = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for targets in ann.frames.values():
        for vt, _ in targets:
            if vt in VEHICLE_TYPE_TO_ID:
                c[VEHICLE_TYPE_TO_ID[vt]] += 1
    return c


def choose_holdout(
    candidates: list[str],
    groups: dict[str, int],
    class_counts: dict[str, np.ndarray],
    frac: float,
    seed: int = 0,
    n_trials: int = 200,
) -> list[str]:
    """Pick ≈ ``frac`` of ``candidates`` as a hold-out set, by whole location groups.

    Tries ``n_trials`` seeded shuffles of the groups, fills the hold-out greedily
    (skipping a group that would overshoot the target by > 50%), and keeps the
    trial whose class proportions are closest to the candidate pool's (L1), with
    penalties for missing classes and for missing the size target. If the
    candidates form fewer than 2 groups, falls back to one group per sequence.
    """
    candidates = sorted(candidates)
    target = max(1, round(frac * len(candidates)))
    cand_groups: dict[int, list[str]] = {}
    for n in candidates:
        cand_groups.setdefault(groups.get(n, -1), []).append(n)
    if len(cand_groups) < 2:
        log.warning("Fewer than 2 location groups among %d candidates; splitting by sequence.", len(candidates))
        cand_groups = {i: [n] for i, n in enumerate(candidates)}

    pool = sum(class_counts[n] for n in candidates)
    pool_p = pool / max(pool.sum(), 1)
    rng = random.Random(seed)
    gids = sorted(cand_groups)
    best: tuple[float, list[str]] | None = None
    for _ in range(n_trials):
        rng.shuffle(gids)
        hold: list[str] = []
        for g in gids:
            members = cand_groups[g]
            if len(hold) >= target:
                break
            if hold and len(hold) + len(members) > 1.5 * target:
                continue
            if len(hold) + len(members) >= len(candidates):  # keep something for train
                continue
            hold.extend(members)
        if not hold:
            continue
        h = sum(class_counts[n] for n in hold)
        h_p = h / max(h.sum(), 1)
        score = float(np.abs(h_p - pool_p).sum())
        score += float(((pool > 0) & (h == 0)).sum())  # a class missing from the hold-out
        score += abs(len(hold) - target) / target
        if best is None or score < best[0]:
            best = (score, sorted(hold))
    if best is None:
        raise ValueError("Could not build a hold-out set; check location grouping threshold.")
    return best[1]


def make_splits(
    seqs: list[SequenceInfo],
    class_counts: dict[str, np.ndarray],
    groups: dict[str, int],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> dict:
    """Build train/val/test sequence lists.

    If both official subsets are present: test = official test, val carved from
    official train by location group. Otherwise an 80/10/10-style group-aware split
    of all sequences.
    """
    subsets = {s.subset for s in seqs}
    by_subset = {sub: sorted(s.name for s in seqs if s.subset == sub) for sub in subsets}
    if {"train", "test"} <= subsets and "unknown" not in subsets:
        strategy = "official_test__val_from_official_train_by_location"
        test = by_subset["test"]
        pool = by_subset["train"]
        val = choose_holdout(pool, groups, class_counts, val_frac, seed)
    else:
        strategy = "grouped_random_80_10_10"
        pool = sorted(s.name for s in seqs)
        test = choose_holdout(pool, groups, class_counts, test_frac, seed)
        rest = [n for n in pool if n not in set(test)]
        val = choose_holdout(rest, groups, class_counts, val_frac / (1 - test_frac), seed + 1)
    train = sorted(n for n in pool if n not in set(val) | set(test))

    def gset(names: list[str]) -> set[int]:
        return {groups[n] for n in names}

    leakage = {
        "train_val_shared_groups": len(gset(train) & gset(val)),
        "train_test_shared_groups": len(gset(train) & gset(test)),
        "test_sequences_sharing_location_with_train": sum(groups[n] in gset(train) for n in test),
    }
    return {"strategy": strategy, "train": train, "val": val, "test": test, "location_leakage": leakage}


# -------------------------------------------------------------------- conversion


def center_in_any(box: Box, regions: list[Box]) -> bool:
    """True if the centre of ``box`` lies inside any of ``regions``."""
    cx, cy = box.center
    return any(r.x1 <= cx <= r.x2 and r.y1 <= cy <= r.y2 for r in regions)


def to_yolo_line(class_id: int, box: Box, img_w: int, img_h: int, min_size: float = 1.0) -> str | None:
    """Clip ``box`` to the image and format a YOLO label line (normalised cx cy w h).

    Returns ``None`` if the clipped box is smaller than ``min_size`` pixels.
    """
    x1, y1 = max(0.0, box.x1), max(0.0, box.y1)
    x2, y2 = min(float(img_w), box.x2), min(float(img_h), box.y2)
    w, h = x2 - x1, y2 - y1
    if w < min_size or h < min_size:
        return None
    return f"{class_id} {(x1 + w / 2) / img_w:.6f} {(y1 + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def select_frames(frame_nums: list[int], stride: int, max_frames: int | None = None) -> list[int]:
    """Every ``stride``-th annotated frame (only frames present in the XML are used)."""
    picked = sorted(frame_nums)[:: max(1, stride)]
    return picked[:max_frames] if max_frames else picked


def convert_sequence(
    seq: SequenceInfo,
    ann: SequenceAnn,
    split: str,
    out_root: Path,
    stride: int,
    max_frames: int | None = None,
    workers: int = 4,
) -> dict:
    """Write images and YOLO labels of one sequence into ``out_root/{images,labels}/split``.

    Ignore regions are blacked out in the image and boxes whose centre is inside
    one are dropped; pixels inside kept boxes are restored so labelled vehicles
    are never partly black. Raises ``ValueError`` on an unmapped vehicle type.
    """
    img_out = Path(out_root) / "images" / split
    lbl_out = Path(out_root) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    frames = select_frames(list(ann.frames), stride, max_frames)
    if not frames:
        return {"images": 0, "boxes": [0] * len(CLASS_NAMES), "dropped_ignored": 0, "dropped_small": 0, "missing_images": 0}
    probe = cv2.imread(str(frame_path(seq.image_dir, frames[0])))
    if probe is None:
        raise FileNotFoundError(frame_path(seq.image_dir, frames[0]))
    img_h, img_w = probe.shape[:2]

    stats = {"images": 0, "boxes": np.zeros(len(CLASS_NAMES), dtype=np.int64), "dropped_ignored": 0, "dropped_small": 0, "missing_images": 0}

    def one(num: int) -> tuple[int, np.ndarray, int, int, bool]:
        src = frame_path(seq.image_dir, num)
        stem = f"{seq.name}_img{num:05d}"
        if not src.exists():
            return 0, np.zeros(len(CLASS_NAMES), dtype=np.int64), 0, 0, False
        lines, counts, n_ign, n_small = [], np.zeros(len(CLASS_NAMES), dtype=np.int64), 0, 0
        kept: list[Box] = []
        for vt, box in ann.frames[num]:
            if vt not in VEHICLE_TYPE_TO_ID:
                raise ValueError(f"Unmapped vehicle_type {vt!r} in {seq.name} frame {num}")
            if center_in_any(box, ann.ignored):
                n_ign += 1
                continue
            cid = VEHICLE_TYPE_TO_ID[vt]
            line = to_yolo_line(cid, box, img_w, img_h)
            if line is None:
                n_small += 1
                continue
            lines.append(line)
            counts[cid] += 1
            kept.append(box)
        dst = img_out / f"{stem}.jpg"
        if ann.ignored:
            orig = cv2.imread(str(src))
            img = orig.copy()
            for r in ann.ignored:
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(np.ceil(r.x2)), int(np.ceil(r.y2))), (0, 0, 0), -1)
            # A labelled vehicle that overlaps an ignore region must stay fully visible,
            # otherwise the model learns boxes around black pixels.
            for b in kept:
                x1, y1 = max(0, int(b.x1)), max(0, int(b.y1))
                x2, y2 = min(img_w, int(np.ceil(b.x2))), min(img_h, int(np.ceil(b.y2)))
                img[y1:y2, x1:x2] = orig[y1:y2, x1:x2]
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            shutil.copyfile(src, dst)
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        return 1, counts, n_ign, n_small, True

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n_img, counts, n_ign, n_small, ok in ex.map(one, frames):
            stats["images"] += n_img
            stats["boxes"] += counts
            stats["dropped_ignored"] += n_ign
            stats["dropped_small"] += n_small
            stats["missing_images"] += 0 if ok else 1
    stats["boxes"] = stats["boxes"].tolist()
    return stats


def write_data_yaml(path: Path, dataset_root: Path) -> None:
    """Write an Ultralytics dataset YAML pointing at ``dataset_root``."""
    import yaml

    cfg = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(CLASS_NAMES)),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False))


# ------------------------------------------------------------------ visualisation

_COLORS = ((0, 200, 0), (255, 160, 0), (0, 0, 255), (200, 0, 200))  # BGR per class


def draw_yolo_labels(image_path: Path, label_path: Path) -> np.ndarray:
    """Draw the boxes of a YOLO label file onto its image (round-trip check)."""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    text = Path(label_path).read_text().strip()
    for line in text.splitlines() if text else []:
        cid, cx, cy, bw, bh = line.split()
        c, cx, cy, bw, bh = int(cid), float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p1, p2 = (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2))
        cv2.rectangle(img, p1, p2, _COLORS[c], 2)
        cv2.putText(img, CLASS_NAMES[c], (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COLORS[c], 1, cv2.LINE_AA)
    return img


def location_contact_sheet(
    seqs: list[SequenceInfo], groups: dict[str, int], first_frames: dict[str, int], thumb: tuple[int, int] = (160, 90), max_per_row: int = 12
) -> np.ndarray:
    """One row per location group (up to ``max_per_row`` thumbnails) for eyeballing."""
    by_group: dict[int, list[SequenceInfo]] = {}
    for s in sorted(seqs, key=lambda s: s.name):
        by_group.setdefault(groups[s.name], []).append(s)
    rows = []
    for gid in sorted(by_group):
        tiles = []
        for s in by_group[gid][:max_per_row]:
            img = cv2.imread(str(frame_path(s.image_dir, first_frames[s.name])))
            tile = cv2.resize(img, thumb) if img is not None else np.zeros((thumb[1], thumb[0], 3), np.uint8)
            label = f"g{gid} {s.name} {s.subset[:2]}"
            cv2.putText(tile, label, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(tile)
        tiles += [np.zeros((thumb[1], thumb[0], 3), np.uint8)] * (max_per_row - len(tiles))
        rows.append(np.hstack(tiles))
    return np.vstack(rows) if rows else np.zeros((thumb[1], thumb[0], 3), np.uint8)
