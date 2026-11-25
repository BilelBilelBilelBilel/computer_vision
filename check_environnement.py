"""
Checks that the environment is ready before the demo.

How to run : python check_environnement.py
It prints the version of every library, checks the model files and the test video,
and ends with OK or with the list of problems to fix.
"""
import importlib
import os
import sys

PACKAGES = ["ultralytics", "cv2", "supervision", "fastapi", "uvicorn", "httpx",
            "pydantic", "numpy", "onnxruntime", "streamlit", "pandas"]
FILES = ["yolov8n.pt", "yolov8n-seg.pt", "data/people.mp4"]

problems = []
print(f"Python {sys.version.split()[0]}")
for name in PACKAGES:
    try:
        module = importlib.import_module(name)
        print(f"  [ok] {name:<12} {getattr(module, '__version__', '')}")
    except ImportError:
        print(f"  [--] {name:<12} MISSING")
        problems.append(f"install {name}  (pip install -r requirements.txt)")

for path in FILES:
    if os.path.exists(path):
        print(f"  [ok] {path}")
    elif path.endswith(".pt"):
        print(f"  [..] {path} not found: Ultralytics downloads it on first run")
    else:
        print(f"  [--] {path} MISSING")
        problems.append(f"add the test video {path} (see RUN.md, section 3)")

try:
    import cv2
    cap = cv2.VideoCapture("data/people.mp4")
    ok, frame = cap.read()
    if ok:
        print(f"  [ok] video readable: {frame.shape[1]}x{frame.shape[0]} @ "
              f"{cap.get(cv2.CAP_PROP_FPS):.0f} fps, {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames")
    cap.release()
except ImportError:
    pass

os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)
print("\nOK: everything is ready." if not problems else "\nTo fix:\n- " + "\n- ".join(problems))
