"""5x5cm sticker sheet + a paper test sheet spanning marker sizes, to find the practical minimum."""
import cv2, json, numpy as np, sys, os
sys.path.insert(0,os.path.dirname(__file__))
from marker_ids import CLASSES, DIAG_BASE
from PIL import Image

DICT, MODULES, DPI, QUIET_MOD = cv2.aruco.DICT_4X4_50, 6, 300, 1
ppmm = DPI/25.4; mm = lambda v:int(round(v*ppmm)); px2mm = lambda p:p/ppmm
d = cv2.aruco.getPredefinedDictionary(DICT)

def cell(modpx, mid, quiet_mod=QUIET_MOD):
    mk = modpx*MODULES; q = modpx*quiet_mod; c = mk+2*q
    tile = np.full((c,c),255,np.uint8)
    tile[q:q+mk, q:q+mk] = cv2.aruco.generateImageMarker(d, mid, mk, borderBits=1)
    return tile, c, mk

# ---------- 5x5 cm stickers ----------
MODPX = 74                      # 74*8 = 592 px = 50.1 mm total, marker 444 px = 37.6 mm
COLS, ROWS, BLEED = 3, 4, mm(3)
ID_BASE = CLASSES[1][0]          # 50 mm class starts at id 20
_, c, mk = cell(MODPX, ID_BASE)
W, H = COLS*c+2*BLEED, ROWS*c+2*BLEED+mm(14)
page = np.full((H,W),255,np.uint8)
i = ID_BASE
for r in range(ROWS):
    for x in range(COLS):
        t,_,_ = cell(MODPX, i)
        px_, py_ = BLEED+x*c, BLEED+r*c
        page[py_:py_+c, px_:px_+c] = t
        cv2.rectangle(page,(px_,py_),(px_+c-1,py_+c-1),210,1)
        cv2.putText(page,str(i),(px_+mm(2),py_+c-mm(2)),cv2.FONT_HERSHEY_SIMPLEX,0.4,150,1)
        i += 1
ry = ROWS*c+2*BLEED+mm(8)
cv2.line(page,(BLEED,ry),(BLEED+mm(100),ry),0,2)
for t_ in range(11):
    x=BLEED+mm(t_*10); cv2.line(page,(x,ry),(x,ry-mm(6 if t_%5==0 else 3)),0,2)
cv2.putText(page,f"100 mm ref | sticker {px2mm(c):.1f} mm sq | marker {px2mm(mk):.1f} mm | MATTE",
            (BLEED,ry+mm(5)),cv2.FONT_HERSHEY_SIMPLEX,0.4,90,1)
cv2.imwrite("archive/aruco_stickers_50mm.png",page)
Image.open("archive/aruco_stickers_50mm.png").convert("L").save("PRINT-THESE/2-markers-STICKERS-50mm.pdf","PDF",resolution=300.0)
print(f"50mm sheet : ids {ID_BASE}-{i-1}, 12 stickers @ {px2mm(c):.1f} mm sq, marker {px2mm(mk):.1f} mm, sheet {px2mm(W):.0f}x{px2mm(H):.0f} mm")

# ---------- size test sheet ----------
sizes = [(39,"19.8"),(44,"22.4"),(59,"30.0"),(74,"37.6")]
page2 = np.full((mm(297),mm(210)),255,np.uint8)
y = mm(18); mid = DIAG_BASE
for modpx,label in sizes:
    t,c2,mk2 = cell(modpx, mid)
    x = mm(20)
    for rep in range(3):
        t2,_,_ = cell(modpx, mid)
        page2[y:y+c2, x:x+c2] = t2
        x += c2 + mm(8); mid += 1
    cv2.putText(page2,f"{px2mm(mk2):.1f} mm marker  ({px2mm(mk2)*1074/600:.0f} px at 60 cm)",
                (mm(20), y-mm(3)), cv2.FONT_HERSHEY_SIMPLEX,0.5,0,1)
    y += c2 + mm(14)
cv2.line(page2,(mm(20),mm(280)),(mm(120),mm(280)),0,2)
for t_ in range(11):
    x=mm(20+t_*10); cv2.line(page2,(x,mm(280)),(x,mm(280)-mm(6 if t_%5==0 else 3)),0,2)
cv2.putText(page2,"100 mm ref - cut out, mount at working distance, see which sizes track reliably",
            (mm(20),mm(286)),cv2.FONT_HERSHEY_SIMPLEX,0.42,90,1)
cv2.imwrite("archive/aruco_size_test.png",page2)
Image.open("archive/aruco_size_test.png").convert("L").save("PRINT-THESE/1-size-test-PAPER.pdf","PDF",resolution=300.0)
print(f"test sheet : sizes {[s[1] for s in sizes]} mm, 3 each")
