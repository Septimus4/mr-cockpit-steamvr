"""Kiss-cut sticker sheet artwork: 12 markers, each a 58.4 mm square including quiet zone."""
import cv2, json, numpy as np
from PIL import Image

DICT, DICT_BITS, BORDER = cv2.aruco.DICT_4X4_50, 4, 1
MODULES, MOD_PX, QUIET_MOD, DPI = DICT_BITS + 2*BORDER, 69, 2, 300
COLS, ROWS = 3, 4
BLEED_MM = 3

ppmm = DPI/25.4; mm = lambda v: int(round(v*ppmm)); px2mm = lambda p: p/ppmm
marker_px, quiet_px = MOD_PX*MODULES, MOD_PX*QUIET_MOD
cell = marker_px + 2*quiet_px
bleed = mm(BLEED_MM)

page_w, page_h = COLS*cell + 2*bleed, ROWS*cell + 2*bleed + mm(16)
page = np.full((page_h, page_w), 255, np.uint8)
d = cv2.aruco.getPredefinedDictionary(DICT)

mid = 0
for r in range(ROWS):
    for c in range(COLS):
        cx, cy = bleed + c*cell, bleed + r*cell
        img = cv2.aruco.generateImageMarker(d, mid, marker_px, borderBits=BORDER)
        page[cy+quiet_px:cy+quiet_px+marker_px, cx+quiet_px:cx+quiet_px+marker_px] = img
        cv2.rectangle(page,(cx,cy),(cx+cell-1,cy+cell-1),210,1)      # faint kiss-cut guide
        cv2.putText(page,f"{mid}",(cx+mm(2),cy+cell-mm(2)),cv2.FONT_HERSHEY_SIMPLEX,0.42,150,1)
        mid += 1

ry, rx = ROWS*cell + 2*bleed + mm(9), bleed
cv2.line(page,(rx,ry),(rx+mm(100),ry),0,2)
for t in range(11):
    x = rx+mm(t*10); cv2.line(page,(x,ry),(x,ry-mm(t%5==0 and 7 or 4)),0,2)
cv2.putText(page,"100 mm reference - each sticker 58.4 mm square - MATTE only",
            (rx,ry+mm(5)),cv2.FONT_HERSHEY_SIMPLEX,0.42,90,1)

cv2.imwrite("aruco_stickers.png", page)
im = Image.open("aruco_stickers.png").convert("L")
im.save("aruco_stickers.pdf","PDF",resolution=300.0)
im.save("aruco_stickers.tif","TIFF",dpi=(300,300),compression="tiff_lzw")

json.dump({"sticker_mm": round(px2mm(cell),2), "marker_mm": round(px2mm(marker_px),2),
           "quiet_mm": round(px2mm(quiet_px),2), "bleed_mm": BLEED_MM,
           "sheet_mm": [round(px2mm(page_w),1), round(px2mm(page_h),1)],
           "count": mid, "dict": "DICT_4X4_50"}, open("stickers.json","w"), indent=2)
print(f"  sheet        {px2mm(page_w):.1f} x {px2mm(page_h):.1f} mm")
print(f"  stickers     {mid} @ {px2mm(cell):.1f} mm square (marker {px2mm(marker_px):.1f} + quiet {px2mm(quiet_px):.1f} each side)")
print(f"  bleed        {BLEED_MM} mm")
