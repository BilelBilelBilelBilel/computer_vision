# Architecture

## Reference architecture (assignment) and what the prototype uses

| Reference block (assignment) | In this prototype | In production |
|---|---|---|
| RTSP / video stream | `data/people.mp4`, webcam, or `--source rtsp://...` | IP cameras (RTSP) |
| OpenCV frame buffer | `vision_common.FrameBuffer`: capture thread + bounded queue | same, one per camera |
| YOLO + ByteTrack inference worker | `step2_telemetry.py` (and steps 1, 4) | one worker per camera, ONNX/TensorRT |
| Structured JSON event bus | HTTP `POST /events`, validated by Pydantic | Kafka / Redis Streams / MQTT |
| PostgreSQL time-series | SQLite `data/events.db` (`events`, `kpi_hourly`) | PostgreSQL + TimescaleDB |
| BI dashboard | Streamlit `step5_dashboard.py` | Streamlit / Power BI / Grafana |
| Alert automation | n8n webhook + verification gate | n8n → Slack / e-mail / Telegram |

```
  EDGE (camera side)                                  only JSON leaves the edge
  +-------------------------------+
  | Video file / webcam / RTSP    |
  +---------------+---------------+
                  | frames
                  v
  +-------------------------------+
  | OpenCV FrameBuffer            |  capture thread -> bounded queue
  | (vision_common.FrameBuffer)   |  live stream: drop oldest frames
  +---------------+---------------+
                  | BGR numpy arrays
                  v
  +-------------------------------+     Step 1: output/annotated.mp4 (--seg: masks)
  | Inference worker              |     Step 3: FPS benchmark (.pt vs .onnx)
  |  YOLOv8n detect (+NMS/IoU)    |---> Step 3: floor-position heatmap
  |  ByteTrack -> track IDs       |
  |  PolygonZone -> counts        |
  |  DwellTracker -> wait times   |
  +-----+-----------------+-------+      +-----------------------------+
        |                 |              | Shelf camera (step2 --shelf)|
        | Step 2          | Step 4       | YOLO products in shelf zone |
        | zone_count /1 s | count >= T   | count < min -> stockout     |
        | dwell on exit   | during 1 s   +--------------+--------------+
        |                 | + frame JPEG                |
  ======|=================|=============================|====  network boundary  ====
        | POST /events    | POST webhook                | POST /events (+ webhook)
        v                 v                             v
  +--------------------+  +------------------------------+
  | FastAPI (api.py)   |  | n8n workflow                 |
  | Pydantic: 422 if   |  |  Webhook -> Verification gate|
  | the event is wrong |  |  -> IF ok -> save image -> 200|
  +---------+----------+  |  -> else -> 422 (reasons)    |
            |             +------------------------------+
            v
  +--------------------+     +---------------------------+
  | SQLite events      |---->| kpi_hourly (per hour/zone)|
  | zone_count, dwell, |     | avg/peak occupancy,       |
  | stockout, alert    |     | unique visitors, wait,    |
  +---------+----------+     | alerts, stockouts         |
            |                +-------------+-------------+
            v                              v
  +------------------------------------------------------+
  | Streamlit dashboard (step5): KPIs, hourly table,     |
  | wait times, trajectories, alerts, images, heatmap    |
  +------------------------------------------------------+
  CLOUD / SERVICES
```

## Components

**Video source + OpenCV FrameBuffer**
- `cv2.VideoCapture` hides where frames come from: a file, a webcam (index `0`) or an RTSP URL.
- In Step 2 a capture thread fills a bounded queue (64 frames) and the worker reads from it:
  reading the video never waits for YOLO. For a file nothing is lost; for a live stream
  the oldest frame is dropped when the worker is too slow (always process recent frames).

**YOLO detector (Ultralytics)**
- Single-stage, anchor-free CNN: one forward pass gives boxes, classes and confidences.
- NMS (based on IoU) removes duplicate boxes. It runs inside `model(frame)`.
- `yolov8n` is the "nano" model (3.2 M parameters, mAP@50-95 = 37.3 on COCO), so it runs on a CPU.
- `yolov8n-seg` adds instance masks (`step1_tracking.py --seg`).

**ByteTrack tracker (supervision)**
- Turns per-frame detections into tracks that keep the same `tracker_id`.
- Kalman prediction + IoU matching, first with high-confidence boxes (> 0.25), then with
  low-confidence ones (0.1–0.25) for tracks still unmatched; a new ID needs a score ≥ 0.35.
- `make_tracker(fps)` passes the real frame rate, so a lost track is kept exactly 1 s.

**ROI zones + dwell time (`vision_common`)**
- Polygons defined once, as fractions of the frame (`ZONES`): the queue zone is a trapezoid
  that follows the floor perspective.
- `zone.trigger(detections)` tests the bottom-center of each box (the feet).
- `DwellTracker` stores, per (zone, ID), the first and last frame inside. A visit ends after
  0.5 s of absence; its duration (video time, not CPU time) is the wait time.

**Telemetry API (FastAPI + Pydantic + SQLite)**
- Four event types: `zone_count` (count + IDs), `dwell`, `stockout`, `alert`.
- Pydantic rejects inconsistent events with HTTP 422 (e.g. `object_count != len(track_ids)`).
- `run_id` identifies each worker launch, because ByteTrack restarts at #1 every time.
- `GET /kpi/hourly` rebuilds the `kpi_hourly` table; `GET /trajectories` lists each person's
  path through the zones (built from the dwell events).

**Alerting (webhook → n8n verification gate)**
- Rule with persistence: count ≥ T for 1 s, one alert per episode, re-armed when it drops.
- The payload carries the annotated JPEG (base64), its SHA-256 and the exact ByteTrack IDs.
- n8n accepts it only if `count == len(track_ids)`, `count >= threshold`, the rule was
  sustained and the image is present; otherwise it answers 422 with the reasons.

**Shelf stockout (step2 `--shelf`)**
- A second camera looks at a shelf; YOLO counts COCO product-like objects in the shelf zone.
- Below `--min-stock` (3 checks in a row for a video) → evidence image + `stockout` event.
- Limit: COCO only knows a few products; a real store needs a YOLO fine-tuned on its SKUs.

**Benchmark + heatmap (offline analysis)**
- The same frames run through PyTorch and ONNX Runtime; only the model call is timed.
- Heatmap: 2D histogram of floor positions → Gaussian blur → `COLORMAP_JET`, blended over a frame.
