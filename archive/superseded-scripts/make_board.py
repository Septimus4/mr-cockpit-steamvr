"""Generate a ChArUco calibration board sized for A4 portrait, plus its parameters."""
import cv2, json, numpy as np

SQUARES_X, SQUARES_Y = 5, 7          # fits A4 portrait with margin
SQUARE_MM, MARKER_MM = 35.0, 26.0
DPI = 300
DICT = cv2.aruco.DICT_4X4_50

d = cv2.aruco.getPredefinedDictionary(DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM, d)

px_per_mm = DPI / 25.4
w = int(round(SQUARES_X * SQUARE_MM * px_per_mm))
h = int(round(SQUARES_Y * SQUARE_MM * px_per_mm))
margin = int(round(6 * px_per_mm))

img = board.generateImage((w + 2 * margin, h + 2 * margin), marginSize=margin, borderBits=1)
cv2.imwrite("charuco_board_A4.png", img)

params = dict(squares_x=SQUARES_X, squares_y=SQUARES_Y,
              square_mm=SQUARE_MM, marker_mm=MARKER_MM, dict=int(DICT), dpi=DPI)
json.dump(params, open("board.json", "w"), indent=2)

print(f"charuco_board_A4.png  {img.shape[1]}x{img.shape[0]} px at {DPI} dpi")
print(f"printed size          {SQUARES_X*SQUARE_MM:.0f} x {SQUARES_Y*SQUARE_MM:.0f} mm  (A4 portrait, fits with margin)")
print(f"square {SQUARE_MM:.0f} mm / marker {MARKER_MM:.0f} mm / DICT_4X4_50")
