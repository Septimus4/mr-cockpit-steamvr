"""
Measure ROW vs SQUARE constellation stability.  Same marker size, same lateral span,
same camera, same frames - the only difference is vertical extent.

  PROP THE CAMERA.  Show screen-layout-test.png via show_1to1.py.  Q to finish.
"""
import cv2, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from cam import open_elp, read_tolerant

K=np.array([[1072.26851867,0,788.49299729],[0,1072.31519651,614.54444602],[0,0,1]],np.float64)
D=np.array([0.08313216691950971,-0.10744697901181298,-0.00016821021003468,0.00025331486744491,0.0],np.float64)

g=np.load("archive/layout_geometry_swapped.npz")
IDS=list(g["ids"]); CTR={int(i):c for i,c in zip(g["ids"],g["centres"])}; MM=float(g["marker_mm"])
GROUPS={"ROW (3.6:1)":[35,36,37], "SQUARE (1:1)":[44,45,46,47]}
h=MM/2
OBJ={i:np.array([[CTR[i][0]-h,CTR[i][1]-h,0],[CTR[i][0]+h,CTR[i][1]-h,0],
                 [CTR[i][0]+h,CTR[i][1]+h,0],[CTR[i][0]-h,CTR[i][1]+h,0]],np.float64)/1000.0
     for i in CTR}

def rsig(v): v=np.asarray(v,float); return 1.4826*np.median(np.abs(v-np.median(v)))

cap=open_elp()
det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                            cv2.aruco.DetectorParameters())
print("KEEP STILL.  Q after ~300 frames.")
T={k:[] for k in GROUPS}; R={k:[] for k in GROUPS}; frames=0; misses=0
while True:
    f,misses=read_tolerant(cap,misses)
    if f is None:
        print(f"  camera stopped delivering after {frames} frames"); break
    if f is False: continue
    frames+=1
    L=f[:,:f.shape[1]//2]
    c,ids,_=det.detectMarkers(cv2.cvtColor(L,cv2.COLOR_BGR2GRAY))
    found={int(i):cc.reshape(4,2).astype(np.float64) for cc,i in zip(c,ids.flatten())} if ids is not None else {}
    for name,mem in GROUPS.items():
        got=[i for i in mem if i in found]
        if len(got)>=3:
            ok2,rv,tv=cv2.solvePnP(np.vstack([OBJ[i] for i in got]),
                                   np.vstack([found[i] for i in got]),K,D,flags=cv2.SOLVEPNP_ITERATIVE)
            if ok2: T[name].append(tv.ravel()); R[name].append(cv2.Rodrigues(rv)[0])
    prev=cv2.resize(L,(800,600))
    if ids is not None: cv2.aruco.drawDetectedMarkers(prev,[x*0.5 for x in c],ids)
    yy=26
    for name in GROUPS:
        cv2.putText(prev,f"{name}  n={len(R[name]):4d}",(10,yy),cv2.FONT_HERSHEY_SIMPLEX,0.62,
                    (0,220,0) if len(R[name])>200 else (0,190,220),2); yy+=28
    cv2.putText(prev,"HOLD STILL - Q to finish",(10,590),cv2.FONT_HERSHEY_SIMPLEX,0.5,(220,220,220),1)
    cv2.imshow("row vs square",prev)
    if (cv2.waitKey(1)&0xFF) in (ord('q'),27): break
cap.release(); cv2.destroyAllWindows()

print(f"{frames} frames\n")
print(f"{'layout':>14} {'n':>5} {'sigX':>7} {'sigY':>7} {'sigZ':>8}   {'rot about x':>12} {'y':>7} {'z':>7}")
print(f"{'':>14} {'':>5} {'mm':>7} {'mm':>7} {'mm':>8}   {'deg':>12} {'deg':>7} {'deg':>7}")
res={}
for name in GROUPS:
    if len(R[name])<30: print(f"{name:>14}   too few detections"); continue
    t=np.array(T[name]); ref=R[name][len(R[name])//2]
    v=np.array([cv2.Rodrigues(ref.T@r)[0].ravel() for r in R[name]])*180/np.pi
    sd=[rsig(v[:,k]) for k in range(3)]
    res[name]=sd
    print(f"{name:>14} {len(t):5d} {rsig(t[:,0])*1000:7.2f} {rsig(t[:,1])*1000:7.2f} {rsig(t[:,2])*1000:8.2f}"
          f"   {sd[0]:12.3f} {sd[1]:7.3f} {sd[2]:7.3f}")
if len(res)==2:
    a,b=res["ROW (3.6:1)"],res["SQUARE (1:1)"]
    print(f"\n  square improves rotation about x by {a[0]/b[0]:.1f}x  (the weakly-constrained axis)")
    print(f"  square improves rotation about y by {a[1]/b[1]:.1f}x")
