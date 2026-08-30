"""
Marker PLATE templates.

A plate is a rigid surface carrying several markers in a KNOWN arrangement - a USB
panel, a blank bay, a piece of ply.  Because the within-plate geometry is fixed by the
template, the solver only has to find each plate's pose, not each marker's, which is
faster, better conditioned, and self-validating (a marker that has moved shows up as a
reprojection outlier against its own plate).

Prints a 1:1 placement guide: lay it on the panel, mark the corners, peel and stick.
Placement need not be exact - bundle adjustment refines it - but a good start helps.

  python scripts/make_plate.py --w 117 --h 149 --name winctrl --ids 0,1,2,3
"""
import cv2, numpy as np, argparse, sys, os, json
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from marker_ids import size_of

ap = argparse.ArgumentParser()
ap.add_argument("--w", type=float, default=117.0, help="usable flat width, mm")
ap.add_argument("--h", type=float, default=149.0, help="usable flat height, mm")
ap.add_argument("--name", default="winctrl")
ap.add_argument("--ids", default="0,1,2,3")
ap.add_argument("--sticker", type=float, default=30.0, help="sticker square, mm")
ap.add_argument("--margin", type=float, default=6.0, help="sticker edge to panel edge, mm")
a = ap.parse_args()

IDS = [int(x) for x in a.ids.split(",")]
if len(IDS) != 4: sys.exit("this template places 4 markers, one per corner")
MK = size_of(IDS[0])
if MK is None: sys.exit(f"id {IDS[0]} has no size class - see marker_ids.py")
if any(size_of(i) != MK for i in IDS): sys.exit("all four ids must be the same size class")

off = a.margin + a.sticker/2                       # marker centre inset from edge
if 2*off >= min(a.w, a.h): sys.exit("panel too small for that sticker + margin")
CTR = [(off, off), (a.w-off, off), (a.w-off, a.h-off), (off, a.h-off)]   # TL TR BR BL

# conditioning check - the thing that actually matters (see anchoring-config.md)
P = np.array(CTR); P = P - P.mean(0)
sv = np.linalg.svd(P, compute_uv=False)
aspect = sv[0]/sv[1]
verdict = "GOOD" if aspect < 2 else ("MARGINAL" if aspect < 3 else "TOO COLLINEAR")

DPI=300; ppmm=DPI/25.4; mm=lambda v:int(round(v*ppmm))
BLEED=mm(10); W,H=mm(a.w)+2*BLEED, mm(a.h)+2*BLEED+mm(22)
page=np.full((H,W),255,np.uint8)
cv2.rectangle(page,(BLEED,BLEED),(BLEED+mm(a.w),BLEED+mm(a.h)),0,2)     # panel outline
d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for mid,(cx,cy) in zip(IDS,CTR):
    x,y=BLEED+mm(cx),BLEED+mm(cy); s=mm(a.sticker)//2
    cv2.rectangle(page,(x-s,y-s),(x+s,y+s),150,1)                        # sticker outline
    m=mm(MK); page[y-m//2:y-m//2+m, x-m//2:x-m//2+m]=cv2.aruco.generateImageMarker(d,mid,m,borderBits=1)
    cv2.line(page,(x-s-mm(4),y),(x-s-mm(1),y),0,1); cv2.line(page,(x,y-s-mm(4)),(x,y-s-mm(1)),0,1)
    cv2.putText(page,f"id {mid}",(x-s,y+s+mm(4)),cv2.FONT_HERSHEY_SIMPLEX,0.42,90,1)

ry=BLEED+mm(a.h)+mm(10)
cv2.line(page,(BLEED,ry),(BLEED+mm(100),ry),0,2)
for t in range(11):
    xx=BLEED+mm(t*10); cv2.line(page,(xx,ry),(xx,ry-mm(5 if t%5==0 else 3)),0,2)
cv2.putText(page,f"100 mm ref | plate '{a.name}' {a.w:.0f}x{a.h:.0f} mm | spread "
                 f"{a.w-2*off:.0f}x{a.h-2*off:.0f} mm | aspect {aspect:.2f}:1 {verdict}",
            (BLEED,ry+mm(6)),cv2.FONT_HERSHEY_SIMPLEX,0.4,0,1)
cv2.putText(page,"PRINT AT 100% - verify the 100 mm reference before use",
            (BLEED,ry+mm(12)),cv2.FONT_HERSHEY_SIMPLEX,0.4,90,1)

os.makedirs("PRINT-THESE/plates",exist_ok=True)
png=f"PRINT-THESE/plates/plate-{a.name}.png"
cv2.imwrite(png,page)
Image.open(png).convert("L").save(f"PRINT-THESE/plates/plate-{a.name}.pdf","PDF",resolution=300.0)
geom=dict(name=a.name, panel_mm=[a.w,a.h], sticker_mm=a.sticker, marker_mm=MK,
          ids=IDS, centres_mm=CTR, spread_mm=[a.w-2*off, a.h-2*off],
          aspect=round(float(aspect),3), verdict=verdict)
json.dump(geom, open(f"PRINT-THESE/plates/plate-{a.name}.json","w"), indent=2)
det=cv2.aruco.ArucoDetector(d,cv2.aruco.DetectorParameters())
_,ids,_=det.detectMarkers(page)
print(f"  plate '{a.name}' {a.w:.0f}x{a.h:.0f} mm, ids {IDS}, marker {MK} mm")
print(f"  marker spread {a.w-2*off:.0f} x {a.h-2*off:.0f} mm   aspect {aspect:.2f}:1  {verdict}")
print(f"  detection {0 if ids is None else len(ids)}/4  ->  {png[:-4]}.pdf + .json")
