"""
Controlled test: ROW layout vs SQUARE layout, same marker size, same lateral span.
The only variable is vertical extent, which is what constrains rotation about the
horizontal axis.  Row = ids 35-37 (22.4 mm), square = ids 44-47 (22.4 mm).
"""
import cv2, numpy as np, sys, os
sys.path.insert(0,os.path.dirname(__file__))

PITCH=530.0/2560; W,H=1440,2560
mm=lambda v:int(round(v/PITCH))
d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
page=np.full((H,W),255,np.uint8)
MODPX=18                      # 18*6 = 108 px = 22.36 mm, matches the 30 mm sticker
MK=MODPX*6; HALF=MK/2
SPAN=80.0                     # mm, identical for both layouts

def put(mid, cx_mm, cy_mm):
    x,y=int(round(cx_mm/PITCH-HALF)), int(round(cy_mm/PITCH-HALF))
    page[y:y+MK, x:x+MK]=cv2.aruco.generateImageMarker(d,mid,MK,borderBits=1)
    return (cx_mm,cy_mm)

LAYOUT={}
cv2.putText(page,"ROW  (ids 35-37)  lateral 80mm, vertical 22mm",(mm(20),mm(170)),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,0,2)
for k,mid in enumerate((35,36,37)):
    LAYOUT[mid]=put(mid, 60+k*(SPAN/2), 200)

cv2.putText(page,"SQUARE  (ids 44-47)  lateral 80mm, vertical 80mm",(mm(20),mm(30)),
            cv2.FONT_HERSHEY_SIMPLEX,0.8,0,2)
for mid,(dx,dy) in zip((44,45,46,47),((0,0),(SPAN,0),(SPAN,SPAN),(0,SPAN))):
    LAYOUT[mid]=put(mid, 60+dx, 60+dy)

ry=H-mm(28); rx=mm(12)
cv2.line(page,(rx,ry),(rx+mm(100),ry),0,3)
for t in range(11):
    xx=rx+mm(t*10); cv2.line(page,(xx,ry),(xx,ry-mm(7 if t%5==0 else 4)),0,3)
cv2.putText(page,"100 mm reference - verify with a real ruler",(rx,ry+mm(9)),
            cv2.FONT_HERSHEY_SIMPLEX,0.7,0,2)

cv2.imwrite("PRINT-THESE/screen-layout-test-swapped.png",page)
np.savez("archive/layout_geometry_swapped.npz",
         ids=np.array(sorted(LAYOUT)),
         centres=np.array([LAYOUT[i] for i in sorted(LAYOUT)]),
         marker_mm=MK*PITCH)
det=cv2.aruco.ArucoDetector(d,cv2.aruco.DetectorParameters())
_,ids,_=det.detectMarkers(page)
print(f"  marker {MK*PITCH:.2f} mm, lateral span {SPAN} mm for both layouts")
print(f"  row    ids 35-37: vertical extent {MK*PITCH:.1f} mm  (aspect {SPAN/(MK*PITCH):.1f}:1)")
print(f"  square ids 44-47: vertical extent {SPAN:.1f} mm  (aspect 1.0:1)")
print(f"  detection {0 if ids is None else len(ids)}/7")
