# Real-Time Visual Analytics with Ultralytics YOLO and OpenCV

A small, educational proof of concept for a real-time visual analytics
architecture (Smart Retail: foot traffic, queue wait time, shelf stockouts).
OpenCV reads a video file, a webcam or an RTSP stream. A single-stage, anchor-free
YOLO detector finds people in each frame, and ByteTrack gives each person an ID that
stays the same from frame to frame. Polygon regions of interest (ROIs) turn those
tracks into occupancy counts and dwell times (how long each person stays in a zone).
The results are sent as lightweight JSON telemetry to a FastAPI + SQLite backend,
aggregated into hourly KPIs, shown in a Streamlit dashboard, benchmarked across two
inference backends (PyTorch and ONNX Runtime), and checked against threshold rules
that fire an n8n webhook with the annotated frame attached. Each step is one short
script you can read from top to bottom. The goal is to show the concepts clearly;
it is not production code.

## Directory layout

```
vision-analytics/
├── requirements.txt          Python dependencies (CPU only)  - or Pipfile for pipenv
├── README.md / RUN.md        Overview + concepts / exact run instructions
├── ARCHITECTURE.md           How the pieces connect (diagram + roles)
├── CHANGEMENTS.md            What changed in this version and why (in French)
├── check_environnement.py    Checks libraries, models and the test video
├── vision_common.py          EDGE helpers shared by all steps: zones, ByteTrack, dwell time, frame buffer
├── telemetry_db.py           CLOUD storage: SQLite tables + hourly KPI aggregation + trajectories
├── api.py                    CLOUD API: FastAPI + Pydantic validation
├── step1_tracking.py         Step 1: YOLO (+ --seg masks) + ByteTrack IDs + ROI counting + dwell → annotated.mp4
├── step2_telemetry.py        Step 2: JSON events (zone_count, dwell, stockout) → API → SQLite, hourly KPIs, trajectories
├── step3_benchmark.py        Step 3: PyTorch vs ONNX FPS benchmark + spatial occupancy heatmap
├── step4_alerts.py           Step 4: sustained-occupancy rule → n8n webhook with the annotated frame
├── step5_dashboard.py        Step 5: Streamlit BI dashboard
├── n8n/vision_alert_workflow.json   n8n workflow to import (webhook + verification gate)
├── data/                     Input video (data/people.mp4); events.db is created here
└── output/                   Generated results: annotated video, heatmap, benchmark, alert frames
```

## Files at a glance

| File | What it does |
|---|---|
| `step1_tracking.py` | Detects people (boxes, or masks with `--seg`), tracks them with persistent IDs, counts them in two polygon zones, shows each person's time in the zone, saves an annotated video. |
| `step2_telemetry.py` | Every second of video, POSTs `zone_count` events `{zone_name, object_count, track_ids}`; when a person leaves a zone, POSTs a `dwell` event with the time spent there. At the end it rebuilds the `kpi_hourly` table and prints KPIs and trajectories. With `--shelf`, it also counts product-like COCO objects in a shelf zone; below `--min-stock` it saves the evidence image and POSTs a `stockout` event (and optionally the n8n webhook). |
| `step3_benchmark.py` | Measures FPS and latency for the `.pt` and `.onnx` models, writes `output/benchmark.txt`, and builds `output/heatmap.jpg` from tracked floor positions. |
| `step4_alerts.py` | If a zone has at least *N* people for 1 s in a row, it saves the triggering frame and POSTs it (base64 + SHA-256 + track IDs) to the n8n webhook, and stores the alert in the API. |
| `step5_dashboard.py` | Streamlit dashboard: KPIs per zone, hourly KPI table, wait times, trajectories, alerts, evidence images, heatmap. |
| `api.py` | `POST /events`, `GET /events`, `GET /kpi/hourly`, `GET /trajectories` (docs at `/docs`). |

## Key Concepts Demonstrated

| Theory term | Where | What to look at |
|---|---|---|
| **Single-stage, anchor-free detection** | Step 1 (all steps) | `model(frame)` predicts classes and boxes in one forward pass. YOLOv8 is anchor-free: each cell predicts the distances to the 4 sides of the box directly, no predefined anchor boxes. |
| **NMS / IoU** | Step 1 (inside Ultralytics) | NMS keeps the highest-confidence box and drops any other box whose IoU with it is above a threshold. ByteTrack also uses IoU to match new boxes to existing tracks. |
| **mAP@50-95** | Theory | The detector's quality score: mean of the AP computed at IoU thresholds 0.50, 0.55 … 0.95 (yolov8n: 37.3 on COCO). |
| **Instance segmentation** | Step 1 `--seg` | `yolov8n-seg.pt` adds one pixel mask per person (`sv.MaskAnnotator`). |
| **ByteTrack / Kalman filtering** | `vision_common.make_tracker` | Kalman prediction + IoU matching in two rounds (high- then low-confidence boxes). The real video fps is passed so a lost track is kept 1 s. |
| **ROI counting + dwell time** | `make_zones`, `DwellTracker` | A person is in a zone when the bottom-center of the box (the feet) is inside the polygon. The dwell tracker measures how long each ID stays: the queue wait time. |
| **Decoupled frame buffer** | `vision_common.FrameBuffer` (Step 2) | Capture runs in its own thread and fills a bounded queue; the inference worker reads from it. On a live stream the oldest frames are dropped. |
| **Edge-to-cloud telemetry** | Step 2, 4 | The cameras (edge) never send video: they send small validated JSON events to the API (cloud). |
| **Hourly BI KPIs** | `telemetry_db.rebuild_hourly_kpis`, Step 5 | Events are aggregated per hour and zone: average/peak occupancy, unique visitors, average/max wait, alerts, stockouts. |
| **Model optimisation** | Step 3 | `model.export(format="onnx")`, then PyTorch vs ONNX Runtime by FPS and latency. |
| **Spatial analytics (heatmap)** | Step 3 | Tracked floor positions → 2D histogram → Gaussian blur → colour map → blended over a frame. |
| **Workflow automation + verification gate** | Step 4 + `n8n/` | A sustained rule fires one alert per episode; n8n checks `count == len(track_ids)`, the threshold and the image before accepting it. |

See **RUN.md** to run it, **ARCHITECTURE.md** for the data flow and **CHANGEMENTS.md** for what changed.
