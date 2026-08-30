from PIL import Image
import os
for name in ["aruco_markers_A4", "chessboard_8x5_30mm_A4"]:
    src = f"{name}.png"
    if not os.path.exists(src):
        print(f"  (skip {src})"); continue
    im = Image.open(src).convert("L")
    w_mm, h_mm = im.width/300*25.4, im.height/300*25.4
    im.save(f"{name}.pdf", "PDF", resolution=300.0)
    im.save(f"{name}.tif", "TIFF", dpi=(300,300), compression="tiff_lzw")
    print(f"  {name}: {im.width}x{im.height}px -> {w_mm:.1f} x {h_mm:.1f} mm")
    for ext in ("pdf","tif"):
        print(f"      {name}.{ext:3s} {os.path.getsize(f'{name}.{ext})' if False else f'{name}.{ext}')/1024:8.0f} KB")
