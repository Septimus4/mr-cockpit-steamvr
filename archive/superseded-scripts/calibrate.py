"""
Solves per-eye intrinsics + distortion for the ELP dual-lens camera, and the
stereo relationship between the two lenses. Emits values ready for Rectus's
config.ini.
"""
import cv2, json, glob, sys
import numpy as np

p = json.load(open("board.json"))
SQ = float(sys.argv[1]) if len(sys.argv) > 1 else p["square_mm"]
if SQ != p["square_mm"]:
    print(f"using measured square size {SQ} mm (board designed at {p['square_mm']} mm)\n")

adict = cv2.aruco.getPredefinedDictionary(p["dict"])
board = cv2.aruco.CharucoBoard((p["squares_x"], p["squares_y"]),
                               SQ / 1000.0, p["marker_mm"] / 1000.0 * (SQ / p["square_mm"]), adict)
detector = cv2.aruco.CharucoDetector(board)

files = sorted(glob.glob("captures/*.png"))
if not files:
    sys.exit("no captures found in ./captures - run capture.py first")

obj = {0: [], 1: []}
img = {0: [], 1: []}
paired = []
size = None

for f in files:
    frame = cv2.imread(f)
    if frame is None:
        continue
    half = frame.shape[1] // 2
    size = (half, frame.shape[0])
    per_eye = {}
    for eye in (0, 1):
        sub = frame[:, :half] if eye == 0 else frame[:, half:]
        g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        cc, cid, _, _ = detector.detectBoard(g)
        if cid is None or len(cid) < 8:
            continue
        op, ip = board.matchImagePoints(cc, cid)
        if op is None or len(op) < 8:
            continue
        obj[eye].append(op); img[eye].append(ip)
        per_eye[eye] = (op, ip, cid)
    if len(per_eye) == 2:
        ids0, ids1 = per_eye[0][2].flatten(), per_eye[1][2].flatten()
        shared = np.intersect1d(ids0, ids1)
        if len(shared) >= 8:
            m0 = np.isin(ids0, shared); m1 = np.isin(ids1, shared)
            paired.append((per_eye[0][0][m0], per_eye[0][1][m0], per_eye[1][1][m1]))

print(f"{len(files)} captures   usable views  L:{len(obj[0])}  R:{len(obj[1])}   stereo pairs:{len(paired)}\n")
if min(len(obj[0]), len(obj[1])) < 8:
    sys.exit("need at least 8 usable views per eye - capture more, varying angle and distance")

FLAGS = cv2.CALIB_FIX_K3   # Rectus stores k1,k2,p1,p2 only
K, D, RMS = {}, {}, {}
for eye in (0, 1):
    rms, k, d, _, _ = cv2.calibrateCamera(obj[eye], img[eye], size, None, None, flags=FLAGS)
    K[eye], D[eye], RMS[eye] = k, d.flatten(), rms
    name = "LEFT " if eye == 0 else "RIGHT"
    print(f"{name}  rms={rms:.3f} px")
    print(f"   fx={k[0,0]:9.3f}  fy={k[1,1]:9.3f}   cx={k[0,2]:8.3f}  cy={k[1,2]:8.3f}")
    print(f"   k1={d[0]: .5f}  k2={d[1]: .5f}   p1={d[2]: .5f}  p2={d[3]: .5f}")
    hfov = 2*np.degrees(np.arctan(size[0]/2 / k[0,0]))
    print(f"   implied HFOV {hfov:.1f} deg   (spec sheet claims 85)\n")

if len(paired) >= 6:
    o = [a for a, _, _ in paired]; i0 = [b for _, b, _ in paired]; i1 = [c for _, _, c in paired]
    srms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        o, i0, i1, K[0], D[0].reshape(-1,1), K[1], D[1].reshape(-1,1), size,
        flags=cv2.CALIB_FIX_INTRINSIC)
    base = np.linalg.norm(T) * 1000.0
    rvec, _ = cv2.Rodrigues(R)
    ang = np.degrees(rvec.flatten())
    print(f"STEREO rms={srms:.3f} px")
    print(f"   measured baseline  {base:.2f} mm   (you measured 64 by hand)")
    print(f"   lens misalignment  pitch={ang[0]:+.3f}  yaw={ang[1]:+.3f}  roll={ang[2]:+.3f} deg")
    print(f"   -> pitch is the one that causes eye strain; >0.5 deg is worth compensating\n")
else:
    base, ang = None, None
    print("not enough stereo pairs for a baseline measurement (need 6+)\n")

print("=" * 66)
print("config.ini  [Camera]")
print("=" * 66)
for eye in (0, 1):
    k, d = K[eye], D[eye]
    n = f"Camera{eye}"
    print(f"{n}_IntrinsicsFocalX = {k[0,0]:.5f}")
    print(f"{n}_IntrinsicsFocalY = {k[1,1]:.5f}")
    print(f"{n}_IntrinsicsCenterX = {k[0,2]:.5f}")
    print(f"{n}_IntrinsicsCenterY = {k[1,2]:.5f}")
    print(f"{n}_IntrinsicsSensorPixelsX = {size[0]}")
    print(f"{n}_IntrinsicsSensorPixelsY = {size[1]}")
    print(f"{n}_IntrinsicsDistR1 = {d[0]:.6f}")
    print(f"{n}_IntrinsicsDistR2 = {d[1]:.6f}")
    print(f"{n}_IntrinsicsDistT1 = {d[2]:.6f}")
    print(f"{n}_IntrinsicsDistT2 = {d[3]:.6f}")
if base:
    print(f"; measured baseline {base:.2f} mm -> TranslationX = -{base/2000:.4f} / +{base/2000:.4f}")

json.dump({"size": size,
           "K": {str(e): K[e].tolist() for e in K},
           "D": {str(e): D[e].tolist() for e in D},
           "rms": {str(e): RMS[e] for e in RMS},
           "baseline_mm": base,
           "stereo_rot_deg": None if ang is None else ang.tolist()},
          open("calibration.json", "w"), indent=2)
print("\nwritten: calibration.json")
