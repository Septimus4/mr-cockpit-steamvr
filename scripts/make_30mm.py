"""30 x 30 mm sticker sheet - 22.4 mm marker, 1-module quiet zone. Compact option."""
import cv2, numpy as np
from PIL import Image
DICT, MODULES, DPI, MODPX = cv2.aruco.DICT_4X4_50, 6, 300, 44   # 44*8 = 352 px = 29.8 mm
COLS, ROWS = 5, 4
ppmm = DPI/25.4; mm = lambda v:int(round(v*ppmm)); px2mm = lambda p:p/ppmm
d = cv2.aruco.getPredefinedDictionary(DICT)
mk, q = MODPX*MODULES, MODPX
cellsz = mk + 2*q
BLEED = mm(3)
W, H = COLS*cellsz+2*BLEED, ROWS*cellsz+2*BLEED+mm(14)
page = np.full((H,W),255,np.uint8)
i=0
for r in range(ROWS):
    for c in range(COLS):
        x,y = BLEED+c*cellsz, BLEED+r*cellsz
        page[y+q:y+q+mk, x+q:x+q+mk] = cv2.aruco.generateImageMarker(d,i,mk,borderBits=1)
        cv2.rectangle(page,(x,y),(x+cellsz-1,y+cellsz-1),210,1)
        cv2.putText(page,str(i),(x+mm(1.2),y+cellsz-mm(1.2)),cv2.FONT_HERSHEY_SIMPLEX,0.32,150,1)
        i+=1
ry = ROWS*cellsz+2*BLEED+mm(8)
cv2.line(page,(BLEED,ry),(BLEED+mm(100),ry),0,2)
for t in range(11):
    x=BLEED+mm(t*10); cv2.line(page,(x,ry),(x,ry-mm(6 if t%5==0 else 3)),0,2)
cv2.putText(page,f"100 mm ref | sticker {px2mm(cellsz):.1f} mm sq | marker {px2mm(mk):.1f} mm | MATTE | 100% scale",
            (BLEED,ry+mm(5)),cv2.FONT_HERSHEY_SIMPLEX,0.38,90,1)
cv2.imwrite("archive/aruco_stickers_30mm.png",page)
Image.open("archive/aruco_stickers_30mm.png").convert("L").save("PRINT-THESE/2b-markers-STICKERS-30mm.pdf","PDF",resolution=300.0)
print(f"  {i} stickers @ {px2mm(cellsz):.1f} mm sq, marker {px2mm(mk):.1f} mm, sheet {px2mm(W):.0f}x{px2mm(H):.0f} mm")
det=cv2.aruco.ArucoDetector(d,cv2.aruco.DetectorParameters())
_,ids,_=det.detectMarkers(cv2.imread("archive/aruco_stickers_30mm.png",cv2.IMREAD_GRAYSCALE))
print(f"  detection: {0 if ids is None else len(ids)}/{i}")
