"""
Step 1 — Detection (+ optional segmentation) + Tracking + ROI counting + dwell time.

What it does : reads a video (file, webcam or RTSP URL), detects people with YOLO,
               gives each person a persistent ID with ByteTrack, counts how many
               people are inside each polygonal zone and shows, next to each ID,
               how long that person has been in the zone. Saves output/annotated.mp4.
Assignment   : Step 1 (core vision pipeline).
How to run   : python step1_tracking.py --source data/people.mp4
               python step1_tracking.py --source data/people.mp4 --seg   (instance masks)
               python step1_tracking.py --source rtsp://user:pass@ip:554/stream
               python step1_tracking.py              (webcam index 0)
               add --show to open a live preview window (press q to quit)

THEORY IN 5 LINES
- Single-stage, anchor-free detection: YOLOv8 predicts, for every cell of the feature
  map, a class score and the distances to the 4 box sides in ONE forward pass (no
  region proposals, no anchor boxes), then NMS/IoU removes duplicate boxes.
- Instance segmentation (--seg): the -seg model also predicts one pixel mask per object.
- ByteTrack: a Kalman filter predicts where each track moves next; new boxes are
  matched to tracks by IoU (high- AND low-confidence boxes) -> stable IDs across frames.
- ROI counting: a polygon zone checks whether each box's bottom-center point lies inside it.
"""
import argparse

import cv2
import supervision as sv
from ultralytics import YOLO

from vision_common import (MODEL_PATH, PERSON_CLASS, SEG_MODEL_PATH, ZONE_COLORS,
                           DwellTracker, make_tracker, make_zones, tracker_ids)

OUTPUT_PATH = "output/annotated.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, help="video file / RTSP URL (default: webcam 0)")
    parser.add_argument("--show", action="store_true", help="show a live preview window")
    parser.add_argument("--seg", action="store_true", help="use yolov8n-seg and draw instance masks")
    args = parser.parse_args()

    # 1) OpenCV video capture: a file path, an RTSP URL, or webcam index 0
    cap = cv2.VideoCapture(args.source if args.source else 0)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from source: {args.source or 'webcam 0'}")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # webcams sometimes report 0
    print(f"Source: {w}x{h} @ {fps:.1f} fps")

    # 2) Model, tracker (with the real fps), zones, annotators
    model = YOLO(SEG_MODEL_PATH if args.seg else MODEL_PATH)
    tracker = make_tracker(fps)
    zones = make_zones(w, h)
    dwell = DwellTracker(fps)
    zone_annotators = [
        sv.PolygonZoneAnnotator(zone=z, color=c, thickness=2, display_in_zone_count=False)
        for z, c in zip(zones.values(), ZONE_COLORS)
    ]
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(opacity=0.45)
    label_annotator = sv.LabelAnnotator()

    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame_idx = 0
    try:
        while ok:
            # 3) Detect (single forward pass; NMS is applied inside Ultralytics)
            result = model(frame, classes=[PERSON_CLASS], verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # 4) Track: assign persistent IDs (detections.tracker_id)
            detections = tracker.update_with_detections(detections)
            ids = tracker_ids(detections)

            # 5) Count per zone + dwell time of each ID inside the zone
            counts, seconds = {}, {}
            for name, z in zones.items():
                inside = z.trigger(detections)                 # True/False per detection
                ids_inside = [int(t) for t in ids[inside]] if len(ids) else []
                counts[name] = len(ids_inside)
                dwell.update(frame_idx, name, ids_inside)
                for t in ids_inside:
                    seconds[t] = dwell.current_seconds(name, t)

            # 6) Draw everything
            labels = [f"#{t} {seconds[t]:.1f}s" if t in seconds else f"#{t}" for t in ids]
            annotated = frame.copy()
            if args.seg and detections.mask is not None:
                annotated = mask_annotator.annotate(annotated, detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels=labels)
            for (name, z), za in zip(zones.items(), zone_annotators):
                annotated = za.annotate(annotated)
                x, y = z.polygon[0]
                cv2.putText(annotated, f"{name}: {counts[name]}", (int(x) + 5, int(y) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(annotated)
            if args.show:
                cv2.imshow("Step 1 - tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frame_idx % max(1, int(fps)) == 0:              # about once per second
                print(f"frame {frame_idx}: {counts}")
            frame_idx += 1
            ok, frame = cap.read()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    print(f"Done. {frame_idx} frames -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
