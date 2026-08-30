"""
Robust ELP open.  MSMF commonly returns E_PENDING (-2147483638) for the first frames
after opening, and it can hold a stale handle briefly after a previous script exits.
Aborting on the first failed grab - as the earlier scripts did - turns that into a
spurious "camera broken".
"""
import cv2, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from devlist import find_elp

def open_elp(w=3200, h=1200, fps=60, warmup_s=8.0):
    dev = find_elp(verbose=False)
    if dev is None:
        sys.exit("ELP not found - is it powered on at the hub?")
    cap = cv2.VideoCapture(dev, cv2.CAP_MSMF)
    if not cap.isOpened():
        sys.exit(f"ELP at index {dev} would not open - another script may still hold it")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    t0 = time.time(); n = 0
    while time.time() - t0 < warmup_s:
        ok, f = cap.read(); n += 1
        if ok and f is not None:
            print(f"ELP index {dev}: {int(cap.get(3))}x{int(cap.get(4))} "
                  f"(streaming after {n} attempt{'s' if n>1 else ''})")
            return cap
        time.sleep(0.1)
    cap.release()
    sys.exit(f"ELP opened but delivered no frames in {warmup_s:.0f}s.\n"
             "  Close any other script using the camera (check for stray python processes),\n"
             "  or unplug/replug the ELP.")

def read_tolerant(cap, misses, limit=60):
    """Return (frame_or_None, misses). Transient grab failures are normal on MSMF."""
    ok, f = cap.read()
    if ok and f is not None: return f, 0
    misses += 1
    if misses > limit: return None, misses
    time.sleep(0.02)
    return False, misses
