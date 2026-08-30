"""
Pose jitter, single marker vs constellation.

The planar pose ambiguity makes a single small marker's ORIENTATION bistable: two
solutions fit almost equally, and the solver flips between them.  Position is
unaffected.  Solving one pose from several markers at once spans a long baseline,
which breaks the ambiguity.  This measures both, side by side.

  PROP THE CAMERA.  Show screen-size-test.png full screen.
  Q / ESC to finish.
"""
import cv2, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from devlist import find_elp
from marker_ids import DIAG_BASE, DIAG_SIZES

K = np.array([[1072.26851867,0,788.49299729],[0,1072.31519651,614.54444602],[0,0,1]], np.float64)
D = np.array([0.08313216691950971,-0.10744697901181298,-0.00016821021003468,0.00025331486744491,0.0], np.float64)

# ---- rebuild screen-size-test.png geometry exactly (see make_screen_test.py) ----
PITCH = 530.0/2560
mmpx  = lambda v:int(round(v/PITCH))
CORNERS = {}                                   # id -> 4x3 object points, metres, Y down
y, mid = mmpx(18), DIAG_BASE
for modpx in [16,18,24,30]:
    mk, q = modpx*6, modpx; cell = mk+2*q
    x = mmpx(12)
    for _ in range(3):
        cx, cy = (x+q+mk/2)*PITCH/1000.0, (y+q+mk/2)*PITCH/1000.0
        h = (mk*PITCH/1000.0)/2
        CORNERS[mid] = np.array([[cx-h,cy-h,0],[cx+h,cy-h,0],[cx+h,cy+h,0],[cx-h,cy+h,0]], np.float64)
        x += cell + mmpx(10); mid += 1
    y += cell + mmpx(16)
SIZES  = {i: DIAG_SIZES[(i-DIAG_BASE)//3] for i in CORNERS}
GROUPS = sorted(set(SIZES.values()))

def single_obj(mm_):
    h = mm_/2000.0
    return np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]], np.float64)

def rsigma(v):
    v=np.asarray(v,float); return 1.4826*np.median(np.abs(v-np.median(v))) if len(v) else 0.0

def ang_between(A,B):
    return np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1)))

def branch_stats(Rs):
    """
    Return (dominant %, within-cluster sigma deg, bimodal?).

    A genuine ambiguity flip is DISCRETE: the sorted angular deviations split into two
    clusters with a clear gap.  Continuous noise has no gap.  The previous midrange
    threshold could not distinguish them - it reported ~50% for unimodal noise and ~0%
    for a tight cluster with one outlier, so its output was meaningless either way.
    """
    n = len(Rs)
    if n < 30: return 0.0, 0.0, False
    sub = Rs[::max(1, n//60)]
    ref = sub[int(np.argmin([sum(ang_between(r,s2) for s2 in sub) for r in sub]))]
    a = np.array([ang_between(ref, r) for r in Rs])
    order = np.argsort(a); asort = a[order]
    d = np.diff(asort)
    if len(d) == 0: return 100.0, 0.0, False
    k = int(np.argmax(d)); gap = d[k]
    med = np.median(d[d > 0]) if (d > 0).any() else 0.0
    lo = k+1; hi = n-lo
    bimodal = (med > 0 and gap/med > 15 and min(lo,hi) > 0.05*n)
    if bimodal:
        keep = order[:lo] if lo >= hi else order[lo:]
        dom = 100.0*max(lo,hi)/n
    else:
        keep = np.arange(n); dom = 100.0
    Rk = [Rs[i] for i in keep]
    ref2 = Rk[len(Rk)//2]
    return dom, rsigma([ang_between(ref2, r) for r in Rk]), bimodal

dev = find_elp(verbose=False)
if dev is None: sys.exit("ELP not found")
cap = cv2.VideoCapture(dev, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,3200); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1200); cap.set(cv2.CAP_PROP_FPS,60)
print(f"ELP at index {dev}: {int(cap.get(3))}x{int(cap.get(4))}")
print("KEEP THE CAMERA STILL.  Q after ~300 frames.\n")
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                              cv2.aruco.DetectorParameters())
sT={i:[] for i in SIZES}; sR={i:[] for i in SIZES}; amb={i:[] for i in SIZES}
cT={g:[] for g in GROUPS}; cR={g:[] for g in GROUPS}
frames=0
while True:
    ok,f = cap.read()
    if not ok: break
    frames+=1
    L=f[:,:f.shape[1]//2]
    corners,ids,_=det.detectMarkers(cv2.cvtColor(L,cv2.COLOR_BGR2GRAY))
    found={}
    if ids is not None:
        for c,i in zip(corners,ids.flatten()):
            i=int(i)
            if i not in SIZES: continue
            found[i]=c.reshape(4,2).astype(np.float64)
            # single-marker solve, both ambiguity branches
            r_,rv,tv,err=cv2.solvePnPGeneric(single_obj(SIZES[i]),found[i],K,D,
                                             flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if r_:
                sT[i].append(tv[0].ravel()); sR[i].append(cv2.Rodrigues(rv[0])[0])
                if len(err)>1 and err[1][0]>1e-9: amb[i].append(float(err[0][0]/err[1][0]))
    # constellation solve: all markers of a size, one joint pose
    for g in GROUPS:
        got=[i for i in SIZES if SIZES[i]==g and i in found]
        if len(got)>=3:
            op=np.vstack([CORNERS[i] for i in got]); ip=np.vstack([found[i] for i in got])
            ok2,rv,tv=cv2.solvePnP(op,ip,K,D,flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2: cT[g].append(tv.ravel()); cR[g].append(cv2.Rodrigues(rv)[0])
    prev=cv2.resize(L,(800,600))
    if ids is not None: cv2.aruco.drawDetectedMarkers(prev,[c*0.5 for c in corners],ids)
    yy=26
    for g in GROUPS:
        cv2.putText(prev,f"{g:5.1f}mm  single n={len(sR[min(i for i in SIZES if SIZES[i]==g)]):4d}"
                         f"  constellation n={len(cR[g]):4d}",(10,yy),
                    cv2.FONT_HERSHEY_SIMPLEX,0.58,(0,220,0) if len(cR[g])>200 else (0,190,220),2); yy+=26
    cv2.putText(prev,"HOLD STILL - Q to finish",(10,590),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1)
    cv2.imshow("jitter: single vs constellation",prev)
    if (cv2.waitKey(1)&0xFF) in (ord('q'),27): break
cap.release(); cv2.destroyAllWindows()

print(f"{frames} frames\n")
print("SINGLE MARKER")
print(f"{'size':>7} {'dominant':>9} {'sigAng':>9} {'ambig':>7}   {'shift@1m':>9}")
print(f"{'':>7} {'branch %':>9} {'in-branch':>9} {'e1/e2':>7}   {'mm':>9}")
for g in GROUPS:
    ids_g=[i for i in SIZES if SIZES[i]==g and len(sR[i])>30]
    if not ids_g: print(f"{g:6.1f}m   too few"); continue
    dom=sa=ar=0.0; bi=False
    for i in ids_g:
        d_,s_,bm=branch_stats(sR[i]); dom=min(dom,d_) if dom else d_; sa=max(sa,s_); bi=bi or bm
        if amb[i]: ar=max(ar,np.median(amb[i]))
    print(f"{g:6.1f}m {dom:8.1f}% {sa:9.3f} {ar:7.2f}   {np.tan(np.radians(sa))*1000:9.1f}   {'BIMODAL' if bi else 'unimodal'}")
print("\nCONSTELLATION (3 markers, one joint solve)")
print(f"{'size':>7} {'sigX':>7} {'sigY':>7} {'sigZ':>8} {'sigAng':>9} {'minor%':>7}   {'shift@1m':>9}")
for g in GROUPS:
    if len(cR[g])<30: print(f"{g:6.1f}m   too few detections"); continue
    t=np.array(cT[g])
    dom,sa,bi = branch_stats(cR[g])
    print(f"{g:6.1f}m {rsigma(t[:,0])*1000:7.2f} {rsigma(t[:,1])*1000:7.2f} {rsigma(t[:,2])*1000:8.2f}"
          f" {sa:9.3f} {100-dom:7.1f}   {np.tan(np.radians(sa))*1000:9.1f}   {'BIMODAL' if bi else 'unimodal'}")
print("\nambig e1/e2 near 1.00 = the two solutions fit equally = orientation is bistable.")

import time
np.savez(f"archive/jitter_raw_{time.strftime('%H%M%S')}.npz",
         **{f"sR_{i}": np.array(v) for i,v in sR.items() if len(v)>10},
         **{f"sT_{i}": np.array(v) for i,v in sT.items() if len(v)>10},
         **{f"cR_{int(g*10)}": np.array(v) for g,v in cR.items() if len(v)>10},
         **{f"cT_{int(g*10)}": np.array(v) for g,v in cT.items() if len(v)>10})
print("\nraw poses saved to archive/ - re-analysable without re-capturing")
