"""
Step 4 — Rule-based alerts + n8n webhook + verification gate.

What it does : runs the tracking pipeline; if a zone holds >= THRESHOLD people for
               SUSTAIN_SECONDS in a row, it fires ONE alert for that episode:
                 1. saves the exact annotated frame (output/alert_frame_<ts>.jpg)
                 2. POSTs a JSON payload to the n8n webhook WITH the image inside
                    (base64) + its SHA-256 + the ByteTrack IDs that were counted
                 3. stores the alert in the telemetry API (if it is running)
Assignment   : Step 4 (workflow automation & verification gate).
How to run   : python step4_alerts.py --source data/people.mp4 --threshold 2
               python step4_alerts.py --source data/people.mp4 --threshold 2 \
                      --webhook http://localhost:5678/webhook-test/vision-alert
               Import n8n/vision_alert_workflow.json in n8n first (see RUN.md).

VERIFICATION GATE (two levels)
- In Python: the alert is sent only if the rule holds for SUSTAIN_SECONDS (not for a
  single noisy frame), and only once per episode (re-armed when the zone empties).
- In n8n   : the "Verification gate" node accepts the alert only if
  count == len(track_ids), count >= threshold, the rule was sustained long enough and
  the image is present. Otherwise n8n answers HTTP 422 and the alert is rejected.
The count is deterministic: it is exactly the list of ByteTrack IDs inside the zone
on the saved frame, so anyone can recount the boxes on the image.
"""
import argparse
from datetime import datetime, timezone

import cv2
import supervision as sv
from ultralytics import YOLO

from vision_common import (MODEL_PATH, PERSON_CLASS, ZONE_COLORS, encode_jpeg,
                           make_tracker, make_zones, post_json, tracker_ids)

SUSTAIN_SECONDS = 1.0   # occupancy must stay high this long (12 frames at 12 fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video file / RTSP URL (default: webcam 0)")
    parser.add_argument("--webhook", default="http://localhost:5678/webhook/vision-alert")
    parser.add_argument("--threshold", type=int, default=5, help="people per zone to alert on")
    parser.add_argument("--sustain-seconds", type=float, default=SUSTAIN_SECONDS)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000",
                        help="telemetry API where alerts are also stored ('' to disable)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source if args.source else 0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from source: {args.source or 'webcam 0'}")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sustain_frames = max(1, int(round(args.sustain_seconds * fps)))
    print(f"Rule: zone >= {args.threshold} people during {args.sustain_seconds}s "
          f"= {sustain_frames} consecutive frames at {fps:.0f} fps")

    model = YOLO(MODEL_PATH)
    tracker = make_tracker(fps)
    zones = make_zones(w, h)
    zone_annotators = {name: sv.PolygonZoneAnnotator(zone=z, color=c, thickness=2,
                                                     display_in_zone_count=False)
                       for (name, z), c in zip(zones.items(), ZONE_COLORS)}
    box_annotator, label_annotator = sv.BoxAnnotator(), sv.LabelAnnotator()

    # Per-zone state: how many consecutive frames above threshold, and whether we
    # already alerted for this "episode" (re-armed once occupancy drops again).
    streak = {name: 0 for name in zones}
    alerted = {name: False for name in zones}
    n_alerts = 0

    frame_idx = 0
    try:
        while ok:
            result = model(frame, classes=[PERSON_CLASS], verbose=False)[0]
            detections = tracker.update_with_detections(sv.Detections.from_ultralytics(result))
            ids = tracker_ids(detections)

            annotated = box_annotator.annotate(frame.copy(), detections)
            annotated = label_annotator.annotate(
                annotated, detections, labels=[f"#{t}" for t in ids])

            inside_ids = {}
            for name, zone in zones.items():
                inside = zone.trigger(detections)
                inside_ids[name] = [int(t) for t in ids[inside]] if len(ids) else []
                annotated = zone_annotators[name].annotate(annotated)
                x, y = zone.polygon[0]
                cv2.putText(annotated, f"{name}: {len(inside_ids[name])}", (int(x) + 5, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- The threshold rule ---
            for name, track_list in inside_ids.items():
                count = len(track_list)
                if count >= args.threshold:
                    streak[name] += 1
                else:
                    streak[name], alerted[name] = 0, False     # re-arm
                if streak[name] >= sustain_frames and not alerted[name]:
                    alerted[name] = True                        # fire ONE alert
                    n_alerts += 1
                    fire_alert(args, annotated, name, track_list, streak[name],
                               sustain_frames, frame_idx, fps)

            if frame_idx % max(1, int(fps)) == 0:
                print(f"frame {frame_idx}: { {n: len(v) for n, v in inside_ids.items()} }")
            frame_idx += 1
            ok, frame = cap.read()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
    print(f"Done. {frame_idx} frames processed, {n_alerts} alert(s).")


def fire_alert(args, annotated, zone, track_list, streak, sustain_frames, frame_idx, fps):
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S%fZ")
    count = len(track_list)

    # Write the reason on the image itself: the picture is the evidence
    evidence = annotated.copy()
    banner = f"ALERT {zone}: {count} >= {args.threshold}  IDs {track_list}"
    cv2.rectangle(evidence, (0, 0), (evidence.shape[1], 34), (0, 0, 200), -1)
    cv2.putText(evidence, banner, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    frame_path = f"output/alert_frame_{ts}.jpg"
    cv2.imwrite(frame_path, evidence)                   # the exact triggering frame
    image_b64, image_sha256 = encode_jpeg(evidence)

    payload = {
        "alert_type": "zone_occupancy",
        "zone": zone,
        "count": count,                                 # deterministic ByteTrack count ...
        "track_ids": track_list,                        # ... = exactly these IDs
        "threshold": args.threshold,
        "sustained_frames": streak,
        "sustain_required_frames": sustain_frames,
        "frame_index": frame_idx,
        "video_time_s": round(frame_idx / fps, 2),
        "timestamp": now.isoformat(),
        "model": MODEL_PATH,
        "tracker": "ByteTrack (supervision)",
        "frame_path": frame_path,
        "image_mime": "image/jpeg",
        "image_sha256": image_sha256,
        "image_base64": image_b64,                      # the image itself, not just a path
    }
    status = post_json(args.webhook, payload)
    short = {k: v for k, v in payload.items() if k != "image_base64"}
    if status:
        print(f"ALERT sent to webhook (HTTP {status}): {short}")
    else:
        print(f"Webhook unreachable or alert rejected -> fallback, payload was:\n  {short}")

    if args.api_url:                                    # keep a trace in the BI database
        post_json(f"{args.api_url.rstrip('/')}/events", {
            "event_type": "alert", "timestamp": now.isoformat(), "zone_name": zone,
            "object_count": count, "track_ids": track_list,
            "details": {k: short[k] for k in ("threshold", "sustained_frames", "frame_index",
                                              "video_time_s", "frame_path", "image_sha256")}
                       | {"webhook_status": status},
        })


if __name__ == "__main__":
    main()
