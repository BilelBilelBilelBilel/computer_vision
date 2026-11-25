# How to Run (from a fresh machine)

These instructions assume no prior setup. Everything runs on the **CPU**. Nothing
needs to be configured by hand: the YOLO weights (`yolov8n.pt`, about 6 MB) download
automatically the first time you run a script.

## 0. Prerequisites

- **Python 3.10 – 3.13** ([python.org](https://www.python.org/downloads/)). On Windows,
  tick *"Add Python to PATH"* during installation.
- An internet connection for the first install (PyTorch is a large download, a few hundred MB).

Open a terminal **inside the `vision-analytics/` folder**. Every command below is
run from there, because the scripts use relative paths (`data/`, `output/`).

```bash
cd vision-analytics
```

## 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
> If PowerShell blocks the script, run this once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> (or use Command Prompt: `.venv\Scripts\activate.bat`)

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

When the environment is active, your prompt starts with `(.venv)`.

## 2. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
The three core libraries alone are `pip install ultralytics opencv-python supervision`;
`requirements.txt` adds the API, the dashboard and ONNX Runtime.

**Or with pipenv** (uses `Pipfile`): `pipenv install` then `pipenv shell`.

## 3. Get a sample video

Any video with people walking in it will work. To use the free Intel sample clip:

**Windows (PowerShell)** — type `curl.exe`, because plain `curl` is a PowerShell alias for a different command:
```powershell
curl.exe -L -o data/people.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
```

**Linux / macOS:**
```bash
curl -L -o data/people.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
```

Or copy your own `.mp4` into `data/` and use its name in the commands below.

> **Webcam fallback:** leave out `--source` and the scripts open webcam index `0`.
> A webcam stream never ends, so stop it with **Ctrl+C** (or **q** in the Step 1 preview window).

## 4. Check the environment

```bash
python check_environnement.py
```
It lists every library with its version, checks the models and the video, and ends with
`OK` or with what to fix.

> Start each demo with a clean database so the KPIs only contain this run:
> delete `data/events.db` (it is recreated automatically).

## 5. Step 1 — Detection, tracking, zone counting and dwell time

```bash
python step1_tracking.py --source data/people.mp4
python step1_tracking.py --source data/people.mp4 --seg     # instance segmentation masks
```
Add `--show` to watch it live. Each label shows the ID and, inside a zone, the time spent
there (e.g. `#3 2.4s`). An RTSP camera works the same way: `--source rtsp://user:pass@ip:554/stream`.
**Result:** `output/annotated.mp4`.

## 6. Step 2 — Telemetry API (FastAPI + SQLite) + hourly KPIs

```bash
python step2_telemetry.py --source data/people.mp4
```
This starts the API at `http://127.0.0.1:8000`, processes the video and sends:
- one `zone_count` event per zone **every second of video** (count + ByteTrack IDs);
- one `dwell` event each time a person **leaves** a zone (time spent = wait time).

At the end it prints the hourly KPIs and the trajectories. **The API keeps running**, open:
- http://127.0.0.1:8000/docs — interactive Swagger UI
- http://127.0.0.1:8000/events — last events (`?event_type=dwell`)
- http://127.0.0.1:8000/kpi/hourly — hourly KPIs per zone (table `kpi_hourly`)
- http://127.0.0.1:8000/trajectories — path of each person through the zones

**Add an event by hand** (PowerShell):
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/events -ContentType "application/json" -Body '{"timestamp":"2026-01-01T12:00:00Z","zone_name":"queue zone","object_count":3,"track_ids":[1,4,7]}'
```
Send `"object_count":2` with three IDs and the API answers **422**: the count must equal the
number of IDs (Pydantic validation).

The API can also run alone: `uvicorn api:app --port 8000`, then
`python step2_telemetry.py --source data/people.mp4 --api-url http://127.0.0.1:8000`.

## 7. Step 2 — Shelf stockout (same script, `--shelf`)

Needs a photo or video of a shelf with bottles, cups or fruit (COCO classes), e.g. `data/shelf.jpg`.
```bash
python step2_telemetry.py --source data/people.mp4 --shelf data/shelf.jpg --min-stock 5
python step2_telemetry.py --shelf data/shelf.jpg --min-stock 5      # shelf only
```
The API is started by the same command (or add `--api-url http://127.0.0.1:8000` if it is already running).
If fewer than 5 products are found in the shelf zone, it saves `output/stockout_<ts>.jpg`
and stores a `stockout` event. Add `--webhook http://localhost:5678/webhook-test/vision-alert`
to also send it to n8n. Tip: take two photos, one full shelf and one almost empty.

## 8. Step 3 — Benchmark (PyTorch vs ONNX) + heatmap

```bash
python step3_benchmark.py --source data/people.mp4
```
The script runs up to 300 frames through each backend. The first run exports
`yolov8n.onnx` with `model.export(format="onnx")`. If `onnxruntime` is missing, the ONNX
part is skipped with a note.
**Results:** `output/benchmark.txt` and `output/heatmap.jpg`.

## 9. Step 4 — Threshold alerts to n8n (with the image)

**Set up n8n once:**
1. Run `npx n8n` (requires Node.js), then open http://localhost:5678.
2. *Workflows → Import from file* → choose `n8n/vision_alert_workflow.json`.
3. Click **Execute workflow** (test mode) — the test URL is `/webhook-test/vision-alert`.
   Or **activate** the workflow — the production URL is `/webhook/vision-alert`.

**Run:**
```bash
python step4_alerts.py --source data/people.mp4 --threshold 2 --webhook http://localhost:5678/webhook-test/vision-alert
```
The rule: a zone holds at least 2 people for 1 s (12 frames at 12 fps). The payload contains
`count`, `track_ids`, `image_base64`, `image_sha256`, `frame_index`… The n8n
**Verification gate** checks it, saves the image (`Save image` node) and answers 200,
or answers 422 with the reasons. If the Step 2 API is running, the alert is also stored
(`/events?event_type=alert`).

Without n8n, the payload is printed in the console and the script keeps running.
You can also use https://webhook.site to see what is received.

## 10. Step 5 — Dashboard

```bash
streamlit run step5_dashboard.py
```
Opens http://localhost:8501: KPIs per zone, hourly KPI table, wait times, trajectories,
alerts and stockouts, last alert images and the heatmap. Click **Rafraîchir** after new runs.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not read from source` | Check the video path, or that no other app is using the webcam. |
| `Port 8000 is already in use` (Step 2) | Run with `--port 8001` and use `8001` in the curl URLs. |
| `FutureWarning: ByteTrack was deprecated` | Harmless. `requirements.txt` pins supervision 0.30.x, the last version that still includes `sv.ByteTrack`. |
| Step 4 says `Webhook unreachable or alert rejected` | n8n answered an error: the workflow is not active (use the `/webhook-test/` URL after clicking *Execute workflow*), or the verification gate rejected the alert (422, see the execution in n8n). |
| n8n `Save image` node fails | n8n may restrict file access to its own folder. The workflow continues anyway (the image is still visible in the execution); change the file path or set `N8N_RESTRICT_FILE_ACCESS_TO`. |
| Slow (< 5 FPS) | This is normal on an older CPU. Use a shorter or lower-resolution video. |
| `--show` window doesn't open on Linux | Run without `--show`; the output video is still saved. |
