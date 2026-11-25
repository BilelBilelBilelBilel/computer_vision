"""
Shared "edge" helpers used by every step (one place instead of four copies).

- ZONES / make_zones : the polygonal ROI zones, defined once as fractions of the frame
- make_tracker       : ByteTrack configured with the REAL frame rate of the video
- DwellTracker       : how long each track ID stays in each zone (queue wait time)
- FrameBuffer        : OpenCV capture in a background thread + bounded queue
                       (the "OpenCV Frame Buffer" box of the architecture diagram)
- encode_jpeg / post_json : small helpers for events and alerts
"""
import base64
import hashlib
import os
import queue
import threading

import cv2
import numpy as np
import supervision as sv

MODEL_PATH = "yolov8n.pt"
SEG_MODEL_PATH = "yolov8n-seg.pt"
PERSON_CLASS = 0               # COCO class 0 = "person"

# Zones are given as FRACTIONS of the frame (x, y in 0..1) so they work for any
# resolution. A zone is any polygon: the queue zone is a trapezoid that follows
# the perspective of the floor, the entrance zone is a quadrilateral.
ZONES = {
    "queue zone":    [(0.05, 0.40), (0.45, 0.40), (0.48, 0.95), (0.02, 0.95)],
    "entrance zone": [(0.55, 0.20), (0.95, 0.20), (0.95, 0.70), (0.55, 0.70)],
}
ZONE_COLORS = [sv.Color.from_hex("#ff4d4d"), sv.Color.from_hex("#33cc66")]


def make_zones(w, h):
    """Convert the fractional polygons to pixels. A person is 'in' a zone when the
    bottom-center of the box (the feet) is inside the polygon."""
    return {
        name: sv.PolygonZone(polygon=(np.array(pts) * [w, h]).astype(int),
                             triggering_anchors=(sv.Position.BOTTOM_CENTER,))
        for name, pts in ZONES.items()
    }


def make_tracker(fps):
    """ByteTrack with the real frame rate. supervision keeps a lost track for
    lost_track_buffer * fps / 30 frames: with the right fps that is always 1 s."""
    return sv.ByteTrack(frame_rate=int(round(fps)), lost_track_buffer=30)


def tracker_ids(detections):
    """detections.tracker_id as a plain int array (never None)."""
    if detections.tracker_id is None:
        return np.array([], dtype=int)
    return detections.tracker_id.astype(int)


class DwellTracker:
    """Measures how long each track ID stays inside each zone (dwell time).

    update() is called every frame with the IDs inside one zone. A visit ends when
    the ID has been absent for more than `grace_frames` (so a person hidden for a
    few frames does not split one visit into two). Times are VIDEO times
    (frame_index / fps), so the result does not depend on how fast the CPU is.
    """

    def __init__(self, fps, grace_seconds=0.5):
        self.fps = fps
        self.grace_frames = max(1, int(grace_seconds * fps))
        self.entered = {}    # (zone, track_id) -> first frame inside
        self.last_seen = {}  # (zone, track_id) -> last frame inside

    def update(self, frame_idx, zone, ids_inside):
        """Returns the visits that just ended: list of dicts."""
        for tid in ids_inside:
            key = (zone, int(tid))
            self.entered.setdefault(key, frame_idx)
            self.last_seen[key] = frame_idx
        finished = []
        for key in [k for k in self.entered if k[0] == zone]:
            if frame_idx - self.last_seen[key] > self.grace_frames:
                finished.append(self._close(key))
        return finished

    def current_seconds(self, zone, tid):
        """Time already spent in the zone by an ID that is still inside (or None)."""
        key = (zone, int(tid))
        if key not in self.entered:
            return None
        return (self.last_seen[key] - self.entered[key] + 1) / self.fps

    def flush(self):
        """End of the video: close every visit still open."""
        return [self._close(key) for key in list(self.entered)]

    def _close(self, key):
        start, end = self.entered.pop(key), self.last_seen.pop(key)
        return {
            "zone": key[0],
            "track_id": key[1],
            "entered_s": round(start / self.fps, 2),
            "exited_s": round((end + 1) / self.fps, 2),
            "dwell_seconds": round((end - start + 1) / self.fps, 2),
        }


class FrameBuffer:
    """Reads frames in a background thread into a bounded queue.

    Decouples video I/O from inference: the capture thread never waits for YOLO.
    - video FILE : the reader waits when the queue is full -> no frame is lost.
    - LIVE stream (webcam / RTSP): the oldest frame is dropped when the queue is
      full -> the worker always processes recent frames (real-time behaviour).
    read() returns None at the end of the stream.
    """

    def __init__(self, source, maxsize=64):
        self.cap = cv2.VideoCapture(source)
        ok, first = self.cap.read()
        if not ok:
            raise SystemExit(f"Could not read from source: {source}")
        self.live = not (isinstance(source, str) and os.path.isfile(source))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30   # webcams sometimes report 0
        self.height, self.width = first.shape[:2]
        self.queue = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._stop = threading.Event()
        self.queue.put(first)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self.queue.put(item, timeout=0.1)
                return
            except queue.Full:
                if self.live and item is not None:
                    try:
                        self.queue.get_nowait()      # drop the oldest frame
                        self.dropped += 1
                    except queue.Empty:
                        pass

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            self._put(frame)
        self.cap.release()
        self._put(None)                              # end-of-stream marker

    def read(self):
        return self.queue.get()

    def stop(self):
        self._stop.set()


def encode_jpeg(image):
    """JPEG-encode an image -> (base64 string, sha256 hex). The hash lets the
    receiver check that the image it got is exactly the one that was saved."""
    ok, buf = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    data = buf.tobytes()
    return base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest()


def post_json(url, payload, timeout=5):
    """POST JSON; never crashes the vision loop. Returns the HTTP status or None."""
    import httpx
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.status_code
    except httpx.HTTPError as e:
        print(f"  POST {url} failed ({type(e).__name__})")
        return None
