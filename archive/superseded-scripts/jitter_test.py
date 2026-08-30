"""
Pose jitter test - the metric that actually decides anchoring quality.

Detection rate says a marker DECODES.  This says how much the solved pose WOBBLES
when nothing is moving, which is what you would see as the cockpit overlay shimmering.

  PROP THE CAMERA SO IT CANNOT MOVE.  Any camera motion is measured as jitter.

  Q / ESC   finish and print the summary
"""
import cv2, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from devlist import find_elp

# left camera, from config-final-calibrated.ini (Camera0, 1600x1200)
K = np.array([[1072.26851867, 0, 788.49299729],
              [0, 1072.31519651, 614.54444602],
              [0, 0, 1]], np.float64)
D = np.array([0.08313216691950971, -0.10744697901181298,
             -0.00016821021003468, 0.00025331486744491, 0.0], np.float64)

from marker_ids import DIAG_MAP as SIZES
GROUPS = sorted(set(SIZES.values()))
FLIP_DEG = 15.0                       # beyond this = planar pose ambiguity flip

def objp(mm):                          # marker corners, ArUco order, in metres
    h = mm/2000.0
    return np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], np.float64)

def rsigma(v):                         # robust sigma via MAD, immune to flips
    v = np.asarray(v); return 1.4826*np.median(np.abs(v-np.median(v))) if len(v) else 0.0

dev = find_elp(verbose=False)
if dev is None: sys.exit("ELP not found - is it powered on at the hub?")
cap = cv2.VideoCapture(dev, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200); cap.set(cv2.CAP_PROP_FPS,60)
print(f"ELP at index {dev}: {int(cap.get(3))}x{int(cap.get(4))}")
print("KEEP THE CAMERA STILL.  Q when the counts stop climbing (~300 frames).\n")

det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                              cv2.aruco.DetectorParameters())
T = {i:[] for i in SIZES}              # translations, metres
R = {i:[] for i in SIZES}              # rotation matrices
frames = 0

while True:
    ok, f = cap.read()
    if not ok: break
    frames += 1
    L = f[:, :f.shape[1]//2]
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY))
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i not in SIZES: continue
            ok2, rvec, tvec = cv2.solvePnP(objp(SIZES[i]), c.reshape(4,2).astype(np.float64),
                                           K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok2:
                T[i].append(tvec.ravel()); R[i].append(cv2.Rodrigues(rvec)[0])

    prev = cv2.resize(L,(800,600))
    if ids is not None: cv2.aruco.drawDetectedMarkers(prev, [c*0.5 for c in corners], ids)
    yy = 26
    for g in GROUPS:
        n = min(len(T[i]) for i in SIZES if SIZES[i]==g)
        s = [rsigma([t[k] for t in T[i]])*1000 for i in SIZES if SIZES[i]==g for k in (2,)]
        cv2.putText(prev,f"{g:5.1f}mm  n={n:4d}  depth sigma {max(s) if s else 0:5.2f}mm",
                    (10,yy),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,0) if n>200 else (0,190,220),2); yy += 26
    cv2.putText(prev,"HOLD STILL - Q to finish",(10,590),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1)
    cv2.imshow("pose jitter (left eye)", prev)
    if (cv2.waitKey(1)&0xFF) in (ord('q'),27): break

cap.release(); cv2.destroyAllWindows()
print(f"{frames} frames\n")
print(f"{'size':>7} {'dist':>7} {'sigX':>7} {'sigY':>7} {'sigZ':>8} {'sigAng':>8} {'flips':>7}   worst-case shift")
print(f"{'':>7} {'cm':>7} {'mm':>7} {'mm':>7} {'mm':>8} {'deg':>8} {'%':>7}   at 1 m projected")
for g in GROUPS:
    ids_g = [i for i in SIZES if SIZES[i]==g and len(T[i]) > 30]
    if not ids_g:
        print(f"{g:6.1f}m {'':>7} too few detections to measure"); continue
    sx=sy=sz=sa=fl=0.0; dist=0.0
    for i in ids_g:
        t = np.array(T[i]); Rs = R[i]
        sx=max(sx,rsigma(t[:,0])*1000); sy=max(sy,rsigma(t[:,1])*1000); sz=max(sz,rsigma(t[:,2])*1000)
        dist=max(dist,np.median(t[:,2])*100)
        Rm = Rs[len(Rs)//2]
        ang = np.array([np.degrees(np.arccos(np.clip((np.trace(Rm.T@r)-1)/2,-1,1))) for r in Rs])
        ang = ang[np.isfinite(ang)]
        sa=max(sa,rsigma(ang)); fl=max(fl,100*np.mean(ang>FLIP_DEG))
    shift = np.tan(np.radians(sa))*1000
    print(f"{g:6.1f}m {dist:7.1f} {sx:7.2f} {sy:7.2f} {sz:8.2f} {sa:8.3f} {fl:7.1f}   {shift:5.1f} mm")
print("\nsigmas are robust (MAD), so a flip inflates 'flips' not the sigmas.")
print("angular jitter dominates: 1 deg of marker wobble = 17 mm of overlay shift at 1 m.")
