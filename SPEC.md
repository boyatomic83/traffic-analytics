# SPEC — Traffic & Vehicle Analytics System

Detection + tracking + line counting + speed estimation on fixed-camera traffic video,
optimized for CPU with ONNX Runtime, served as a free web demo.

---

## 1. Goals

1. Fine-tune YOLO11n on traffic-camera data and show it beats COCO weights on the same test set.
2. Export to ONNX and **measure** the CPU speed-up against PyTorch, plus FP32 vs INT8.
3. Turn detections into metrics: persistent IDs, per-class in/out counts, km/h speed, overspeed alerts.
4. Ship a web demo where a visitor clicks a sample clip and gets an annotated video plus a summary.
5. A README that proves everything with real numbers, charts and a GIF.

## 2. Non-goals

- No live RTSP or webcam streams in v1.
- No speed estimation on moving cameras (dashcams). The homography assumes a fixed camera.
- No licence-plate or face recognition.
- No made-up numbers. Every metric in the README comes from a logged run.

## 3. Development environment (no local machine)

| Where | Used for |
|---|---|
| **GitHub repo** | Single source of truth for all code. Notebooks never contain core logic. |
| **Kaggle Notebook (GPU T4 x2, Internet ON)** | Data prep, training, export, CPU benchmark, evaluation, rendering demo videos |
| **Google Colab (free)** | Backup only, when the Kaggle quota runs out |
| **Hugging Face Space (Gradio, ZeroGPU)** | Main deployment |
| **Streamlit Community Cloud** | Backup deployment (about 1 GB RAM, CPU only) |

Rules:
- Each notebook is thin: `git clone` → `pip install -e .` → call scripts in `scripts/` → save artifacts.
- Kaggle needs a phone-verified account for Internet access. Store `GITHUB_TOKEN` and `HF_TOKEN` as Kaggle Secrets, never in code.
- Kaggle quota counts wall-clock time, so stop sessions when idle. Use "Save & Run All (Commit)" for long training runs.
- Artifacts (`best.pt`, `best.onnx`, `best_int8.onnx`, metrics, charts) are pushed to a **Hugging Face model repo**. Weights are never committed to GitHub. Only small PNGs and JSONs go into `assets/`.
- All non-training code must run on CPU, so it can be tested in a Kaggle CPU session without using GPU quota.

## 4. Repo layout

```
traffic-analytics/
├── README.md
├── SPEC.md
├── pyproject.toml            # package: traffic_analytics
├── requirements.txt          # runtime deps for the app (pinned after first green run)
├── configs/
│   ├── data.yaml             # YOLO dataset config (generated)
│   ├── train.yaml            # epochs, imgsz, batch, etc.
│   └── scenes/
│       ├── sample_1.yaml     # per-video calibration (see §7)
│       └── sample_2.yaml
├── src/traffic_analytics/
│   ├── __init__.py
│   ├── detector.py           # ONNX Runtime YOLO wrapper (letterbox, decode, NMS)
│   ├── tracker.py            # thin wrapper over sv.ByteTrack
│   ├── counter.py            # line crossing, per-class in/out
│   ├── speed.py              # homography + smoothed speed per track
│   ├── annotate.py           # boxes, IDs, trails, speed labels, counter panel
│   ├── pipeline.py           # process_video(video_path, scene_cfg) -> (out_mp4, report)
│   ├── report.py             # summary JSON schema
│   ├── video_io.py           # reading, writing, ffmpeg H.264 re-encode
│   └── scene.py              # load and validate scene YAML
├── scripts/
│   ├── prepare_data.py       # UA-DETRAC → YOLO format, class mapping, sequence-level split
│   ├── train.py
│   ├── export.py             # ONNX FP32 + INT8 (static quantization with calibration images)
│   ├── benchmark.py          # PyTorch vs ONNX FP32 vs ONNX INT8 on CPU
│   ├── evaluate.py           # mAP of COCO vs fine-tuned vs INT8 on the same test split
│   ├── run_video.py          # CLI: process one video with a scene config
│   └── make_gif.py           # 5-second README GIF from an annotated video
├── notebooks/
│   ├── 01_train_kaggle.ipynb
│   ├── 02_export_benchmark_kaggle.ipynb
│   └── 03_render_demos_kaggle.ipynb
├── app/
│   ├── app.py                # Gradio app (HF Space)
│   ├── streamlit_app.py      # backup app
│   ├── requirements.txt
│   └── samples/              # 2 short sample clips (≤ 15 s each)
├── assets/                   # README images: GIF, PR curve, confusion matrix, benchmark chart
└── tests/
    ├── test_speed.py
    ├── test_counter.py
    └── test_detector.py
```

## 5. Data

**Primary dataset: UA-DETRAC** (fixed traffic cameras, about 960×540, 25 FPS), attached from a Kaggle mirror.
VisDrone is optional and only for a later small-object experiment.

`scripts/prepare_data.py` must:
1. **Inspect the attached dataset first** and detect its format (XML or already YOLO). Print a summary
   (number of sequences, images, annotations per class) before converting anything.
2. Map classes to: `0 car`, `1 van`, `2 bus`, `3 truck_other` (UA-DETRAC "others" → truck_other).
   Note: UA-DETRAC has no motorcycles. Say this in the README. Do not pretend otherwise.
3. **Split by sequence, not by frame** (about 80/10/10), because neighbouring frames are near-duplicates
   and a random split leaks test data into training. Save the split lists to `configs/splits.json`.
4. Subsample training frames (every 5th–10th frame, configurable) to keep each epoch under about 10 minutes on a T4.
5. Respect "ignore regions" if the annotations have them (black them out or drop the boxes inside).
6. Write `configs/data.yaml`.

## 6. Training, export, benchmark, evaluation

**Training** (`scripts/train.py`, Ultralytics):
- Model `yolo11n.pt`, `imgsz=640`, 30–50 epochs, `device=0,1` (T4 x2), early stopping (patience 10).
- Save `results.png`, `PR_curve.png`, `confusion_matrix_normalized.png` and `best.pt`.
- Log the final mAP50 and mAP50-95 per class to `assets/metrics.json`.

**Export** (`scripts/export.py`):
- ONNX FP32: `imgsz=640, simplify=True, opset` set explicitly. Ultralytics handles the export.
- ONNX INT8: `onnxruntime.quantization.quantize_static` with about 200 calibration images from the train split.
- Check that each ONNX model's outputs match `best.pt` on 20 images (box IoU > 0.9 for matched boxes).

**Benchmark** (`scripts/benchmark.py`, CPU only, `CUDA_VISIBLE_DEVICES=""`):
- Run 200 frames from a sample video and discard the first 20 as warm-up.
- Compare: PyTorch `best.pt`, ONNX FP32, ONNX INT8. Also test `intra_op_num_threads` = 2 and 4.
- Report the mean and p95 ms/frame for preprocessing, inference and postprocessing separately, plus end-to-end FPS.
- Record the CPU model (`lscpu`) and thread count in the output. **The README must state the hardware.**
  Run the benchmark with 2 threads as well, because the deployment target has only 2 vCPUs.
- Output `assets/benchmark.json` and `assets/benchmark.png`.

**Evaluation** (`scripts/evaluate.py`):
- On the same test split, score COCO-pretrained `yolo11n.pt` (with COCO classes mapped to ours), fine-tuned FP32
  and fine-tuned INT8, all on mAP50 and mAP50-95.
- Output `assets/eval.json` and a comparison table that the README uses.

## 7. Analytics pipeline

### 7.1 Detector (`detector.py`)
- Our own ONNX Runtime wrapper, not `YOLO("best.onnx")`: letterbox → inference → decode → class-aware NMS
  (`cv2.dnn.NMSBoxes` or our own) → rescale boxes to the original frame.
- Returns `sv.Detections` with `xyxy`, `confidence`, `class_id`.
- Configurable `conf` (default 0.3), `iou` (0.5), `providers=["CPUExecutionProvider"]`, `num_threads`.
- `test_detector.py`: on a fixed image, our wrapper's detections match Ultralytics' own ONNX predictions.

### 7.2 Tracker (`tracker.py`)
- `sv.ByteTrack(frame_rate=video_fps, ...)`, with parameters exposed in the scene config.
- Keep a `track_id → class_id` map using a majority vote over the track's life, so labels don't flicker.

### 7.3 Counter (`counter.py`)
- `sv.LineZone(start, end)` with the bottom-centre anchor.
- Per-class in/out counts. Use supervision's per-class counts if the installed version has them.
  Otherwise count crossings ourselves using the `track_id → class` map. Each track ID counts at most once per direction.
- `test_counter.py`: synthetic tracks that cross, don't cross, and wobble on the line → expected counts.

### 7.4 Speed (`speed.py`)
- Each scene config defines `source_polygon` (4 pixel points on the road, clockwise from top-left) and
  `target_size_m: [width, length]`. `cv2.getPerspectiveTransform` maps pixels to metres.
- For each track, transform the bottom-centre point and keep a deque of (frame_idx, x_m, y_m) covering the last 1 s.
- Speed = displacement over the window ÷ time from frame indices and **video FPS** (never wall-clock time) × 3.6.
- Only report speed once a track has ≥ 0.5 s of history **inside the polygon**. Smooth with a median of the last N estimates.
- Clamp or ignore physically impossible values (> 200 km/h) and count them as `speed_rejected` in the report.
- `test_speed.py`: a synthetic point moving at a known metres/frame through a known homography → error within ±2%.

### 7.5 Scene config (`configs/scenes/*.yaml`)
```yaml
video: app/samples/sample_1.mp4
fps_override: null            # use the container FPS unless set
line: {start: [100, 400], end: [860, 400]}
source_polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
target_size_m: [10.5, 30.0]   # width, length — estimated from lane width / dash spacing
speed_limit_kmh: 60
calibration_note: "Lane width assumed 3.5 m; dashed-line period assumed 9 m. Speeds are approximate."
tracker: {track_activation_threshold: 0.25, lost_track_buffer: 30, minimum_matching_threshold: 0.8}
detector: {conf: 0.3, iou: 0.5, num_threads: 2}
max_seconds: 20
```

### 7.6 Annotation (`annotate.py`)
Boxes coloured by class, `#ID class speed km/h` labels, short motion trails, the counting line, the speed polygon
(semi-transparent) and a counter panel (per-class in/out, live FPS). Overspeeding vehicles get a red box.

### 7.7 Video I/O (`video_io.py`)
- Read with OpenCV. Write frames with `mp4v`, then re-encode with
  `ffmpeg -y -i tmp.mp4 -c:v libx264 -preset veryfast -pix_fmt yuv420p -movflags +faststart out.mp4`
  so the video plays in browsers.
- Downscale input so its longest side is ≤ 960 px, and cut to `max_seconds`.

### 7.8 Report (`report.py`) — output JSON schema
```json
{
  "video": "sample_1.mp4",
  "duration_s": 15.0, "fps": 25.0, "frames_processed": 375,
  "processing_fps": 0.0,
  "counts": {"car": {"in": 0, "out": 0}, "van": {...}, "bus": {...}, "truck_other": {...}},
  "total_vehicles": 0,
  "unique_tracks": 0,
  "speed": {"avg_kmh": 0.0, "median_kmh": 0.0, "p85_kmh": 0.0, "max_kmh": 0.0, "speed_rejected": 0},
  "speed_limit_kmh": 60,
  "overspeeding_alerts": [{"track_id": 12, "class": "car", "speed_kmh": 74.2, "time_s": 6.4}],
  "calibration_note": "..."
}
```

## 8. Web app

### 8.1 Gradio (`app/app.py`) — Hugging Face Space on ZeroGPU
- Inputs: a video upload, a sample dropdown (2 bundled clips, one click to run) and a speed-limit slider.
- Outputs: the annotated video, a summary table (counts per class), alert rows and a JSON download.
- Uploaded videos without a scene config: counting and tracking run with a default horizontal line at 60% height.
  **Speed is turned off** and the UI explains why (no calibration).
- v1 runs ONNX on CPU only. Optional v2: wrap a PyTorch path in `@spaces.GPU(duration=60)` and fall back to
  ONNX CPU if the GPU quota is unavailable.
- Hard limits: 20 s of video, ≤ 960 px. Show a progress bar (`gr.Progress`).
- Model weights are downloaded from the HF model repo at startup with `huggingface_hub.hf_hub_download`.

### 8.2 Streamlit backup (`app/streamlit_app.py`)
Same pipeline and the same limits, but INT8 model, 640 px max and 15 s max because of about 1 GB RAM.
Use `packages.txt` with `ffmpeg` and `opencv-python-headless`.

## 9. README contents (in this order)
1. Title, one-line pitch, **live demo link**, badges.
2. A 5-second looping GIF (boxes, IDs, trails, speed labels, counter going up).
3. Architecture diagram (Mermaid).
4. Results: a COCO vs fine-tuned vs INT8 mAP table, the PR curve and the confusion matrix.
5. CPU benchmark table, with the CPU model and thread count stated.
6. How speed is calibrated, and its limits (approximate, fixed camera only).
7. How to run it (Kaggle notebook links, CLI usage).
8. Limitations and future work. Licence notes: Ultralytics is AGPL-3.0; follow the dataset's licence.

## 10. Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| M0 | Repo skeleton, package, tests run on CPU | `pytest` passes in a Kaggle CPU session |
| M1 | Data prep | Class counts printed, sequence-level split saved, `data.yaml` valid, 5 sample images with drawn labels look right |
| M2 | Training | `best.pt`, charts and `metrics.json` saved and pushed to the HF model repo |
| M3 | Export + benchmark | FP32 and INT8 ONNX match PyTorch (§6), `benchmark.json` and chart exist |
| M4 | Evaluation | `eval.json` has the COCO vs fine-tuned vs INT8 comparison |
| M5 | Pipeline | `run_video.py` produces a browser-playable MP4 and a valid report JSON for both samples. Counts were checked by hand on one clip (within ±10%) |
| M6 | Gradio app on HF | Public Space runs a sample in under 3 minutes |
| M7 | Streamlit backup | Deployed and runs one sample |
| M8 | README | All sections in §9 filled with real numbers |

Stretch goals (after M8): click-to-calibrate in the UI, wrong-way detection, lane-level `PolygonZone` stats,
a ByteTrack vs BoT-SORT comparison with TrackEval (HOTA), SAHI on VisDrone.

## 11. Coding conventions
- Python 3.10+, type hints, docstrings on public functions, `logging` instead of print in `src/`.
- No hard-coded Kaggle paths in `src/`. Paths come from CLI arguments or environment variables.
- Every script has `--help` and a `--quick` flag (a tiny subset or a few frames) for smoke tests.
- Fixed random seeds. Every artifact JSON includes package versions and a timestamp.
- Pin `requirements.txt` only after the first working run, using the versions that actually ran.