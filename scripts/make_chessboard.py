"""
Printable chessboard matching camera-calibration.exe's defaults:
8 x 5 INNER corners, 30 mm squares. 9 x 6 squares on A4 landscape.
Asymmetric corner count resolves the 180-degree orientation ambiguity.
"""
import cv2, numpy as np

SQ_MM   = 30.0
COLS, ROWS = 9, 6              # squares -> inner corners 8 x 5
DPI     = 300
PAGE_W, PAGE_H = 297.0, 210.0  # A4 landscape

ppmm = DPI / 25.4
mm = lambda v: int(round(v * ppmm))
page = np.full((mm(PAGE_H), mm(PAGE_W)), 255, np.uint8)

bw, bh = COLS * SQ_MM, ROWS * SQ_MM
ox, oy = (PAGE_W - bw) / 2, 14.0     # centred horizontally, biased up for the ruler

for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:
            continue
        x0, y0 = mm(ox + c * SQ_MM), mm(oy + r * SQ_MM)
        page[y0:y0 + mm(SQ_MM), x0:x0 + mm(SQ_MM)] = 0

# 100 mm verification ruler
ry, rx = mm(oy + bh + 20), mm(ox)
cv2.line(page, (rx, ry), (rx + mm(100), ry), 0, 2)
for t in range(11):
    x = rx + mm(t * 10)
    cv2.line(page, (x, ry), (x, ry - mm(9 if t % 5 == 0 else 5)), 0, 2)
    if t % 5 == 0:
        cv2.putText(page, str(t*10), (x - mm(3), ry - mm(11)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
cv2.putText(page, "MEASURE: must be exactly 100 mm", (rx, ry + mm(7)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1)

tx = mm(ox + 118)
for i, line in enumerate([
        "camera-calibration.exe settings:",
        "   Number of inner corners:  8 x 5",
        "   Square side (cm):         3.0",
        "Print A4 LANDSCAPE at 100% / Actual size.",
        "Glue to card or foamboard - flatness matters",
        "more than print quality."]):
    cv2.putText(page, line, (tx, ry - mm(6) + mm(6.5) * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0 if i < 3 else 90, 1)

cv2.imwrite("chessboard_8x5_30mm_A4.png", page)
print(f"chessboard_8x5_30mm_A4.png   {COLS}x{ROWS} squares @ {SQ_MM:.0f} mm")
print(f"inner corners                {COLS-1} x {ROWS-1}   <- enter this in the tool")
print(f"board area                   {bw:.0f} x {bh:.0f} mm on A4 landscape")
print(f"quiet zone                   {ox:.0f} mm sides, {oy:.0f} mm top")
