"""
Step 2 — Edge-to-cloud telemetry (vision workers -> JSON events -> FastAPI -> SQLite).

What it does : starts the API (or uses one already running) and runs up to two "edge"
               cameras that send small JSON events to it:
               1) PEOPLE camera (--source): the Step 1 pipeline through a FrameBuffer
                 - zone_count : people per zone + their ByteTrack IDs, once per second
                 - dwell      : when a person LEAVES a zone, how long they stayed
                                (= queue wait time for the queue zone)
               2) SHELF camera (--shelf, optional): counts the products visible in a
                  SHELF zone with YOLO. If the count stays below --min-stock it emits
                  ONE "stockout" event and saves the annotated shelf image as evidence
                  (optionally also sends it to the n8n webhook, like Step 4).
               At the end it rebuilds the hourly KPI table and prints the KPIs and
               the trajectories. The API then stays up so you can query it.
Assignment   : Step 2 (structured telemetry aggregation + stockout alerts)
               + example app (low-shelf-stock alerts).
How to run   : python step2_telemetry.py --source data/people.mp4
               python step2_telemetry.py --source data/people.mp4 --shelf data/shelf.jpg --min-stock 5
               python step2_telemetry.py --shelf data/shelf.jpg --min-stock 5        (shelf only)
               add --webhook http://localhost:5678/webhook-test/vision-alert to send stockouts to n8n
               then:  http://127.0.0.1:8000/docs   /events   /kpi/hourly   /trajectories
               (Ctrl+C to stop the API)
               With an API already running elsewhere:
               python step2_telemetry.py --source data/people.mp4 --api-url http://server:8000

DESIGN CHOICE: the vision loop talks to the API over HTTP even when both run on the
same PC. That mirrors a real deployment where the cameras ("edge") and the server
("cloud") are different machines: only the URL changes.

HONEST LIMIT (shelf): the YOLO model is trained on COCO, which only knows a few
product-like classes (bottle, cup, bowl, banana, apple, orange...). For a real store you
would fine-tune YOLO on your own product images (e.g. the SKU-110K dataset). The pipeline
(count in a zone -> threshold -> event + image) stays exactly the same.
"""
import argparse
import os
import threading
import time
from datetime import datetime, timezone

import cv2
import httpx
import numpy as np
import supervision as sv
import uvicorn
from ultralytics import YOLO

from vision_common import (MODEL_PATH, PERSON_CLASS, DwellTracker, FrameBuffer, encode_jpeg,
                           make_tracker, make_zones, post_json, tracker_ids)

# ---- People camera ----
EMIT_EVERY_SECONDS = 1.0       # one zone_count event per zone per second of video

# ---- Shelf camera ----
# COCO classes that look like products on a shelf
PRODUCT_CLASSES = {39: "bottle", 41: "cup", 45: "bowl", 46: "banana", 47: "apple",
                   49: "orange", 51: "carrot"}
SHELF_ZONE = [(0.02, 0.02), (0.98, 0.02), (0.98, 0.98), (0.02, 0.98)]   # fractions of the frame
CONFIDENCE = 0.15            # small objects on a shelf get low scores with the nano model
CHECK_EVERY_SECONDS = 1.0
SUSTAIN_CHECKS = 3           # video: the shelf must look empty on 3 checks in a row
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def start_api_in_background(port):
    from api import app
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:      # wait until the server is ready to accept requests
        if not thread.is_alive():  # uvicorn exits immediately if the port is taken
            raise SystemExit(f"Port {port} is already in use. Try: --port 8001")
        time.sleep(0.1)
    print(f"API running at http://127.0.0.1:{port}  (docs: http://127.0.0.1:{port}/docs)")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# =====================================================================================
# 1) People camera: zone_count + dwell events
# =====================================================================================
def run_people_camera(source, events_url, run_id):
    frames = FrameBuffer(source)                              # capture thread + queue
    fps = frames.fps
    emit_every = max(1, int(round(fps * EMIT_EVERY_SECONDS)))
    print(f"Source: {frames.width}x{frames.height} @ {fps:.1f} fps -> "
          f"one zone_count event every {emit_every} frames")

    model = YOLO(MODEL_PATH)
    tracker = make_tracker(fps)
    zones = make_zones(frames.width, frames.height)
    dwell = DwellTracker(fps)

    def send_dwell(visits):
        for v in visits:
            event = {
                "event_type": "dwell", "timestamp": now_iso(), "run_id": run_id,
                "zone_name": v["zone"], "track_id": v["track_id"],
                "dwell_seconds": v["dwell_seconds"],
                "details": {"entered_s": v["entered_s"], "exited_s": v["exited_s"]},
            }
            post_json(events_url, event)
            print(f"dwell: #{v['track_id']} stayed {v['dwell_seconds']:.1f}s in {v['zone']}")
        return len(visits)

    frame_idx, sent = 0, 0
    try:
        while (frame := frames.read()) is not None:
            result = model(frame, classes=[PERSON_CLASS], verbose=False)[0]
            detections = tracker.update_with_detections(sv.Detections.from_ultralytics(result))
            ids = tracker_ids(detections)

            for name, zone in zones.items():
                inside = zone.trigger(detections)                  # boolean mask
                ids_inside = [int(t) for t in ids[inside]] if len(ids) else []
                sent += send_dwell(dwell.update(frame_idx, name, ids_inside))

                if frame_idx % emit_every == 0:
                    event = {
                        "event_type": "zone_count", "timestamp": now_iso(), "run_id": run_id,
                        "zone_name": name,
                        "object_count": len(ids_inside),
                        "track_ids": ids_inside,
                        "details": {"frame_index": frame_idx,
                                    "video_time_s": round(frame_idx / fps, 2)},
                    }
                    post_json(events_url, event)                   # edge -> cloud
                    sent += 1
                    print("event:", event)
            frame_idx += 1
    except KeyboardInterrupt:
        print("Video loop stopped by user.")
    finally:
        frames.stop()

    sent += send_dwell(dwell.flush())          # people still inside at the end
    print(f"\nVideo done: {frame_idx} frames, {sent} events sent "
          f"({frames.dropped} frames dropped by the buffer).")


# =====================================================================================
# 2) Shelf camera: stockout events
# =====================================================================================
def is_image(source):
    return os.path.splitext(source)[1].lower() in IMAGE_EXTENSIONS


def read_shelf_frames(source):
    """Yields (frame_index, frame, fps). Works for one image or for a video."""
    if is_image(source):
        img = cv2.imread(source)
        if img is None:
            raise SystemExit(f"Could not read image: {source}")
        yield 0, img, 1.0
        return
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(round(fps * CHECK_EVERY_SECONDS)))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx, frame, fps
        idx += 1
    cap.release()


def run_shelf_camera(args, events_url, run_id):
    print(f"\nShelf camera: {args.shelf}  (alert below {args.min_stock} products)")
    model = YOLO(MODEL_PATH)
    box_annotator, label_annotator = sv.BoxAnnotator(), sv.LabelAnnotator()
    zone = None
    low_checks, alerted = 0, False
    needed = 1 if is_image(args.shelf) else SUSTAIN_CHECKS   # one photo = one check

    for idx, frame, fps in read_shelf_frames(args.shelf):
        if zone is None:
            h, w = frame.shape[:2]
            zone = sv.PolygonZone(polygon=(np.array(SHELF_ZONE) * [w, h]).astype(int),
                                  triggering_anchors=(sv.Position.CENTER,))
            zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.from_hex("#ffaa00"),
                                                     display_in_zone_count=False)

        result = model(frame, classes=list(PRODUCT_CLASSES), conf=CONFIDENCE, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[zone.trigger(detections)] if len(detections) else detections
        count = len(detections)
        per_class = {}
        for c in (detections.class_id if count else []):
            per_class[PRODUCT_CLASSES[int(c)]] = per_class.get(PRODUCT_CLASSES[int(c)], 0) + 1
        print(f"t={idx / fps:5.1f}s  products in {args.shelf_zone}: {count} {per_class}")

        low_checks = low_checks + 1 if count < args.min_stock else 0
        if count >= args.min_stock:
            alerted = False                                   # shelf refilled: re-arm
        if low_checks >= needed and not alerted:
            alerted = True
            annotated = zone_annotator.annotate(box_annotator.annotate(frame.copy(), detections))
            annotated = label_annotator.annotate(
                annotated, detections,
                labels=[PRODUCT_CLASSES[int(c)] for c in detections.class_id] if count else [])
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (0, 140, 255), -1)
            cv2.putText(annotated, f"STOCKOUT {args.shelf_zone}: {count} < {args.min_stock}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            send_stockout(args, events_url, run_id, annotated, count, per_class, idx, fps)


def send_stockout(args, events_url, run_id, annotated, count, per_class, idx, fps):
    now = datetime.now(timezone.utc)
    os.makedirs("output", exist_ok=True)
    path = f"output/stockout_{now.strftime('%Y%m%dT%H%M%S%fZ')}.jpg"
    cv2.imwrite(path, annotated)
    image_b64, sha = encode_jpeg(annotated)
    details = {"min_stock": args.min_stock, "per_class": per_class, "frame_path": path,
               "image_sha256": sha, "video_time_s": round(idx / fps, 2)}
    print(f"STOCKOUT on {args.shelf_zone}: {count} < {args.min_stock} -> {path}")

    post_json(events_url, {
        "event_type": "stockout", "timestamp": now.isoformat(), "run_id": run_id,
        "zone_name": args.shelf_zone, "object_count": count, "details": details})
    if args.webhook:
        post_json(args.webhook, {
            "alert_type": "stockout", "zone": args.shelf_zone, "count": count,
            "track_ids": [], "threshold": args.min_stock, "timestamp": now.isoformat(),
            "model": MODEL_PATH, "frame_path": path, "image_mime": "image/jpeg",
            "image_sha256": sha, "image_base64": image_b64, "details": details})


# =====================================================================================
# KPIs + main
# =====================================================================================
def print_kpis(base_url, run_id):
    """Structured hourly KPIs + trajectories (computed by the API from the stored events)."""
    try:
        kpis = httpx.get(f"{base_url}/kpi/hourly", timeout=10).json()
        print("\nHourly KPIs (table kpi_hourly):")
        for k in kpis:
            wait = "-" if k["avg_dwell_s"] is None else f"{k['avg_dwell_s']}s"
            print(f"  {k['hour']}  {k['zone']:<14} avg={k['avg_count']} peak={k['peak_count']} "
                  f"visitors={k['unique_visitors']} avg_wait={wait} stockouts={k['stockouts']}")
        paths = [t for t in httpx.get(f"{base_url}/trajectories", timeout=10).json()
                 if t["run_id"] == run_id]
        print(f"\nTrajectories of this run ({len(paths)} people):")
        for t in paths[:15]:
            print(f"  #{t['track_id']}: {' -> '.join(t['path'])}  ({t['total_dwell_s']}s)")
    except httpx.HTTPError as e:
        print(f"Could not read KPIs ({type(e).__name__})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None,
                        help="people camera: video file / RTSP URL (default: webcam 0, "
                             "unless only --shelf is given)")
    parser.add_argument("--shelf", default=None, help="shelf camera: image or video (optional)")
    parser.add_argument("--min-stock", type=int, default=5, help="stockout below this many products")
    parser.add_argument("--shelf-zone", default="shelf A", help="name of the shelf zone")
    parser.add_argument("--webhook", default=None, help="optional n8n webhook URL for stockouts")
    parser.add_argument("--port", type=int, default=8000, help="port of the local API")
    parser.add_argument("--api-url", default=None,
                        help="use an API that is already running (e.g. http://127.0.0.1:8000)")
    args = parser.parse_args()

    if args.api_url:
        base_url = args.api_url.rstrip("/")
    else:
        start_api_in_background(args.port)
        base_url = f"http://127.0.0.1:{args.port}"
    events_url = f"{base_url}/events"
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")   # IDs restart at #1 each run

    if args.source is not None or args.shelf is None:        # people camera
        run_people_camera(args.source if args.source else 0, events_url, run_id)
    if args.shelf:                                            # shelf camera
        run_shelf_camera(args, events_url, run_id)

    print_kpis(base_url, run_id)

    if args.api_url:
        return
    print(f"\nAPI still running. Try:  {base_url}/kpi/hourly   (Ctrl+C to quit)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Bye.")


if __name__ == "__main__":
    main()
