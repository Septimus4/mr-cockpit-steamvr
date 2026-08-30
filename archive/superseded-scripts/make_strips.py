"""
Anchoring marker strips. Each strip carries 4 ArUco markers at known spacing, so
the geometry *within* a strip is exact by construction - only the strip's position
on the rig has to be measured later.
"""
import cv2, json, numpy as np

DICT        = cv2.aruco.DICT_4X4_50
MARKER_MM   = 35.0
GAP_MM      = 12.0
STRIPS      = 4
PER_STRIP   = 4
MARGIN_MM   = 6.0
DPI         = 300

d = cv2.aruco.getPredefinedDictionary(DICT)
ppmm = DPI / 25.4
mm = lambda v: int(round(v * ppmm))

strip_w = PER_STRIP * MARKER_MM + (PER_STRIP - 1) * GAP_MM + 2 * MARGIN_MM
strip_h = MARKER_MM + 2 * MARGIN_MM

# A4 portrait at 300dpi
page = np.full((mm(297), mm(210)), 255, np.uint8)
layout = {"dict": int(DICT), "marker_mm": MARKER_MM, "gap_mm": GAP_MM, "strips": []}

y = mm(15)
mid = mm(8)
for s in range(STRIPS):
    ids = list(range(s * PER_STRIP, (s + 1) * PER_STRIP))
    x0 = mm((210 - strip_w) / 2)
    cv2.rectangle(page, (x0, y), (x0 + mm(strip_w), y + mm(strip_h)), 160, 1)
    for i, mid_ in enumerate(ids):
        img = cv2.aruco.generateImageMarker(d, mid_, mm(MARKER_MM))
        mx = x0 + mm(MARGIN_MM + i * (MARKER_MM + GAP_MM))
        my = y + mm(MARGIN_MM)
        page[my:my + img.shape[0], mx:mx + img.shape[1]] = img
    cv2.putText(page, f"strip {s}  ids {ids[0]}-{ids[-1]}", (x0, y - mm(3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, 120, 1)
    # marker centres along the strip, x in mm from the strip's left edge
    layout["strips"].append({
        "strip": s, "ids": ids,
        "centres_mm": [MARGIN_MM + i * (MARKER_MM + GAP_MM) + MARKER_MM / 2 for i in range(PER_STRIP)],
        "centre_y_mm": MARGIN_MM + MARKER_MM / 2,
        "strip_w_mm": strip_w, "strip_h_mm": strip_h})
    y += mm(strip_h + 14)

# --- print-scale verification ruler: 100 mm with 10 mm ticks ---
ry = mm(258)
rx = mm(15)
cv2.line(page, (rx, ry), (rx + mm(100), ry), 0, 2)
for t in range(11):
    x = rx + mm(t * 10)
    h = 9 if t % 5 == 0 else 5
    cv2.line(page, (x, ry), (x, ry - mm(h)), 0, 2)
    if t % 5 == 0:
        cv2.putText(page, str(t * 10), (x - mm(3), ry - mm(11)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
cv2.putText(page, "MEASURE THIS LINE: it must be exactly 100 mm.",
            (rx, ry + mm(7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1)
cv2.putText(page, "If it is not, reprint at 100% / Actual size (not Fit to page),",
            (rx, ry + mm(13)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 90, 1)
cv2.putText(page, "or tell the software the real marker size you measured.",
            (rx, ry + mm(18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 90, 1)

cv2.putText(page, "cut along the outlines - keep matte, keep flat",
            (mm(15), mm(287)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 120, 1)
cv2.imwrite("aruco_strips_A4.png", page)
json.dump(layout, open("strips.json", "w"), indent=2)
print(f"aruco_strips_A4.png  {STRIPS} strips x {PER_STRIP} markers, {MARKER_MM:.0f}mm each")
print(f"strip size           {strip_w:.0f} x {strip_h:.0f} mm")
print(f"ids                  0-{STRIPS*PER_STRIP-1} (DICT_4X4_50, plenty spare)")
