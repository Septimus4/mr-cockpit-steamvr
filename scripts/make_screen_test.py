"""On-screen marker size test for the Lenovo L24q-35 (2560x1440 over 530x300 mm)."""
import cv2, numpy as np, sys, os
sys.path.insert(0,os.path.dirname(__file__))
from marker_ids import DIAG_BASE
PITCH = 530.0/2560          # mm per native pixel = 0.2070
W, H = 1440, 2560           # native, rotated to portrait
mm = lambda v:int(round(v/PITCH))
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
page = np.full((H,W),255,np.uint8)

sizes = [16,18,24,30]       # px per module -> 6 modules per marker
y, mid = mm(18), DIAG_BASE
for modpx in sizes:
    mk = modpx*6; q = modpx
    size_mm = mk*PITCH
    cv2.putText(page,f"{size_mm:.1f} mm marker   ({size_mm*1074/600:.0f} px at 60 cm)",
                (mm(12), y-mm(4)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 0, 2)
    x = mm(12)
    for _ in range(3):
        cell = mk+2*q
        tile = np.full((cell,cell),255,np.uint8)
        tile[q:q+mk, q:q+mk] = cv2.aruco.generateImageMarker(d, mid, mk, borderBits=1)
        page[y:y+cell, x:x+cell] = tile
        x += cell + mm(10); mid += 1
    y += mk+2*q + mm(16)

ry, rx = H-mm(28), mm(12)
cv2.line(page,(rx,ry),(rx+mm(100),ry),0,3)
for t in range(11):
    xx=rx+mm(t*10); cv2.line(page,(xx,ry),(xx,ry-mm(7 if t%5==0 else 4)),0,3)
    if t%5==0: cv2.putText(page,str(t*10),(xx-mm(4),ry-mm(10)),cv2.FONT_HERSHEY_SIMPLEX,0.6,0,2)
cv2.putText(page,"MEASURE THIS LINE ON SCREEN WITH A REAL RULER",(rx,ry+mm(9)),cv2.FONT_HERSHEY_SIMPLEX,0.7,0,2)
cv2.putText(page,"then tell me the number - sizes scale by (measured / 100)",(rx,ry+mm(16)),cv2.FONT_HERSHEY_SIMPLEX,0.55,90,1)

cv2.imwrite("PRINT-THESE/screen-size-test.png",page)
det=cv2.aruco.ArucoDetector(d,cv2.aruco.DetectorParameters())
_,ids,_=det.detectMarkers(page)
print(f"  image {W}x{H} px for the L24q-35 at {PITCH:.4f} mm/px")
print(f"  sizes {[f'{s*6*PITCH:.1f}' for s in sizes]} mm, 3 each")
print(f"  ids {DIAG_BASE}-{mid-1}, detection {0 if ids is None else len(ids)}/{mid-DIAG_BASE}")
