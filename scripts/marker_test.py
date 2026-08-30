"""
Point the ELP at the on-screen (or printed) size test: reports, per marker size, the
real detection rate and measured pixel size, using the same detector the anchoring
would use.  Counts DISTINCT ids, so a second copy of the sheet in view cannot corrupt
the rate - it is reported as a warning instead.

  Q / ESC   finish and print the summary
"""
import cv2, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from devlist import find_elp

from marker_ids import DIAG_MAP as SIZES
GROUPS = sorted(set(SIZES.values()))

dev = find_elp(verbose=False)
if dev is None: sys.exit("ELP not found - is it powered on at the hub?")
cap = cv2.VideoCapture(dev, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200); cap.set(cv2.CAP_PROP_FPS,60)
print(f"ELP at index {dev}: {int(cap.get(3))}x{int(cap.get(4))}\n")

det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                              cv2.aruco.DetectorParameters())
frames = 0
hits   = {g:0 for g in GROUPS}      # frames where all 3 DISTINCT ids were seen
idhits = {i:0 for i in SIZES}       # per-marker frame count
pxsize = {g:[] for g in GROUPS}
dup    = {g:0 for g in GROUPS}      # frames with a duplicate id -> sheet in view twice

while True:
    ok, f = cap.read()
    if not ok: break
    frames += 1
    L = f[:, :f.shape[1]//2]
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY))
    uniq = {g:set() for g in GROUPS}
    ndet = {g:0   for g in GROUPS}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i not in SIZES: continue
            grp = SIZES[i]
            if i not in uniq[grp]: idhits[i] += 1        # first sighting this frame only
            uniq[grp].add(i); ndet[grp] += 1
            p = c.reshape(4,2)
            pxsize[grp].append(np.mean([np.linalg.norm(p[k]-p[(k+1)%4]) for k in range(4)]))
    for g in GROUPS:
        if len(uniq[g]) == 3:      hits[g] += 1
        if ndet[g] > len(uniq[g]): dup[g]  += 1

    prev = cv2.resize(L,(800,600))
    if ids is not None: cv2.aruco.drawDetectedMarkers(prev, [c*0.5 for c in corners], ids)
    yy = 26
    for g in GROUPS:
        rate = 100*hits[g]/max(frames,1)
        px = np.median(pxsize[g][-90:]) if pxsize[g] else 0
        col = (0,220,0) if rate>90 else (0,190,220) if rate>50 else (0,0,255)
        warn = "  DUPLICATE!" if dup[g] else ""
        cv2.putText(prev,f"{g:5.1f}mm  all3:{rate:5.1f}%  {px:5.1f}px{warn}",(10,yy),
                    cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,0,255) if warn else col,2); yy += 26
    cv2.putText(prev,"Q to finish",(10,590),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1)
    cv2.imshow("marker size test (left eye)", prev)
    if (cv2.waitKey(1)&0xFF) in (ord('q'),27): break

cap.release(); cv2.destroyAllWindows()
print(f"{frames} frames\n")
print(f"{'size':>7} {'all-3':>8} {'worst id':>9} {'px/marker':>10} {'px/module':>10}  note")
for g in GROUPS:
    px = np.median(pxsize[g]) if pxsize[g] else 0
    worst = min(100*idhits[i]/max(frames,1) for i in SIZES if SIZES[i]==g)
    note = f"DUPLICATES in {100*dup[g]/max(frames,1):.0f}% of frames - result invalid" if dup[g] else ""
    print(f"{g:6.1f}m {100*hits[g]/max(frames,1):7.1f}% {worst:8.1f}% {px:10.1f} {px/6:10.1f}  {note}")
print("\npx/module is now a MEDIAN (robust to a scaled second copy).")
print("rule of thumb: >3 px/module decodes, >6 px/module gives usable pose")
