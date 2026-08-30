"""
Individually cuttable ArUco markers for cockpit anchoring.

Design notes:
  - module-aligned pixel sizes, so marker edges land on exact pixel boundaries
    (no resampling blur, which is what actually degrades detection at small sizes)
  - 2 modules of quiet zone on every side, kept INSIDE the cut line
  - ID printed outside the quiet zone so it can't interfere with detection
  - pure 0/255 for the markers themselves
"""
import cv2, json, numpy as np

DICT      = cv2.aruco.DICT_4X4_50
DICT_BITS = 4
BORDER    = 1
MODULES   = DICT_BITS + 2 * BORDER          # 6
DPI       = 300
MOD_PX    = 69                              # -> marker 414 px = 35.05 mm
QUIET_MOD = 2                               # quiet zone in modules
COLS, ROWS = 3, 4

ppmm   = DPI / 25.4
mm     = lambda v: int(round(v * ppmm))
px2mm  = lambda p: p / ppmm

marker_px = MOD_PX * MODULES
quiet_px  = MOD_PX * QUIET_MOD
cell_w    = marker_px + 2 * quiet_px
label_px  = mm(6)
cell_h    = cell_w + label_px

page = np.full((mm(297), mm(210)), 255, np.uint8)
d = cv2.aruco.getPredefinedDictionary(DICT)

total_w = COLS * cell_w
x0 = (mm(210) - total_w) // 2
y0 = mm(13)

layout = {"dict": "DICT_4X4_50", "marker_mm": round(px2mm(marker_px), 2),
          "quiet_mm": round(px2mm(quiet_px), 2), "module_mm": round(px2mm(MOD_PX), 3),
          "dpi": DPI, "ids": []}

mid = 0
for r in range(ROWS):
    for c in range(COLS):
        cx, cy = x0 + c * cell_w, y0 + r * cell_h
        cv2.rectangle(page, (cx, cy), (cx + cell_w - 1, cy + cell_w - 1), 170, 1)
        img = cv2.aruco.generateImageMarker(d, mid, marker_px, borderBits=BORDER)
        page[cy + quiet_px: cy + quiet_px + marker_px,
             cx + quiet_px: cx + quiet_px + marker_px] = img
        cv2.putText(page, f"ID {mid}", (cx + 4, cy + cell_w + mm(4.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 110, 1)
        layout["ids"].append(mid)
        mid += 1

# 100 mm verification ruler
ry, rx = y0 + ROWS * cell_h + mm(13), x0
cv2.line(page, (rx, ry), (rx + mm(100), ry), 0, 2)
for t in range(11):
    x = rx + mm(t * 10)
    cv2.line(page, (x, ry), (x, ry - mm(9 if t % 5 == 0 else 5)), 0, 2)
    if t % 5 == 0:
        cv2.putText(page, str(t*10), (x - mm(3), ry - mm(11)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
for i, s in enumerate([
        "MEASURE THIS LINE: must be exactly 100 mm.  Print A4 at 100% / Actual size.",
        f"Marker {px2mm(marker_px):.1f} mm, quiet zone {px2mm(quiet_px):.1f} mm - cut ON the grey line, never inside it.",
        "DICT_4X4_50, IDs 0-11 (38 spare). Matte paper. Keep flat.  Each marker mounts independently."]):
    cv2.putText(page, s, (rx, ry + mm(7 + i * 5.5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 90, 1)

cv2.imwrite("aruco_markers_A4.png", page)
json.dump(layout, open("markers.json", "w"), indent=2)
print(f"aruco_markers_A4.png  {COLS*ROWS} markers, IDs 0-{mid-1}")
print(f"  marker      {px2mm(marker_px):.2f} mm ({marker_px} px, {MOD_PX} px/module - exact)")
print(f"  quiet zone  {px2mm(quiet_px):.2f} mm ({QUIET_MOD} modules)")
print(f"  cut cell    {px2mm(cell_w):.1f} mm square")
