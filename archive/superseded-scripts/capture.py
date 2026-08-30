"""
Live capture for stereo calibration of the ELP dual-lens camera.

Shows a preview with ChArUco detection for BOTH eyes so you can see whether the
board is actually usable before you spend a shot on it.

  SPACE  capture (only accepted when both eyes see enough corners)
  F      force-capture anyway
  Q/ESC  finish
"""
import cv2, json, sys, os, time
import numpy as np

OUT = "captures"
MIN_CORNERS = 12          # per eye, to accept a shot

p = json.load(open("board.json"))
adict = cv2.aruco.getPredefinedDictionary(p["dict"])
board = cv2.aruco.CharucoBoard((p["squares_x"], p["squares_y"]),
                               p["square_mm"], p["marker_mm"], adict)
detector = cv2.aruco.CharucoDetector(board)

os.makedirs(OUT, exist_ok=True)

from devlist import find_elp
if len(sys.argv) > 1:
    dev = int(sys.argv[1])
else:
    dev = find_elp(verbose=False)
    if dev is None:
        sys.exit("ELP not found - check the USB connection.")
    print(f"found ELP at index {dev}")
cap = cv2.VideoCapture(dev, cv2.CAP_MSMF)
if not cap.isOpened():
    sys.exit(f"Could not open camera {dev}. Is the passthrough layer holding it? Close SteamVR/DCS first.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3200)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
cap.set(cv2.CAP_PROP_FPS, 60)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"capturing at {W}x{H} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps  ->  {W//2}x{H} per eye")
if (W, H) != (3200, 1200):
    print(f"WARNING: expected 3200x1200, got {W}x{H} - calibration will describe this mode, not the one you fly.")

def corners_for(half):
    g = cv2.cvtColor(half, cv2.COLOR_BGR2GRAY)
    cc, cid, _, _ = detector.detectBoard(g)
    return (cc, cid, 0 if cid is None else len(cid))

shot = len([f for f in os.listdir(OUT) if f.endswith(".png")])
print(f"{shot} existing captures in ./{OUT}\n")

while True:
    ok, frame = cap.read()
    if not ok:
        print("frame grab failed"); break

    half = frame.shape[1] // 2
    L, R = frame[:, :half], frame[:, half:]
    (lc, lid, ln), (rc, rid, rn) = corners_for(L), corners_for(R)

    prev = cv2.resize(frame, (1600, 600))
    sx = 1600 / frame.shape[1]; sy = 600 / frame.shape[0]
    for cc, off in ((lc, 0), (rc, half)):
        if cc is not None:
            for pt in cc.reshape(-1, 2):
                cv2.circle(prev, (int((pt[0] + off) * sx), int(pt[1] * sy)), 3, (0, 255, 255), -1)

    good = ln >= MIN_CORNERS and rn >= MIN_CORNERS
    col = (0, 220, 0) if good else (0, 0, 255)
    cv2.putText(prev, f"L:{ln:2d}  R:{rn:2d}   shots:{shot}", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    cv2.putText(prev, "SPACE capture   F force   Q done", (12, 578),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    cv2.line(prev, (800, 0), (800, 600), (90, 90, 90), 1)
    cv2.imshow("ELP calibration capture", prev)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27):
        break
    if k == ord(' ') and not good:
        print(f"  rejected - need {MIN_CORNERS}+ corners per eye (L:{ln} R:{rn})")
    if (k == ord(' ') and good) or k == ord('f'):
        path = os.path.join(OUT, f"shot_{shot:03d}.png")
        cv2.imwrite(path, frame)
        shot += 1
        print(f"  saved {path}   L:{ln} R:{rn}")

cap.release(); cv2.destroyAllWindows()
print(f"\n{shot} captures in ./{OUT}")
