"""
Render a marker PLATE on a USB display panel at an exact physical size, with the
usable area adjusted INTERACTIVELY.

Cockpit panels are rarely fully visible: a coaming lip covers the top, a bracket eats
a corner, the panel sits recessed. The usable region is therefore a sub-rectangle of
the display, and it differs per panel. It is set by eye, in the cockpit, not from a
spec sheet.

  list monitors:
    python scripts/show_plate.py --list

  scale check first (measure the line; the correction is exactly linear):
    python scripts/show_plate.py --monitor 2 --panel-w 110 --ruler

  place the plate, then adjust until all four markers sit in visible area:
    python scripts/show_plate.py --monitor 2 --panel-w 110 --ids 0,1,2,3

  KEYS   Tab      select edge (top/right/bottom/left/ALL)
         arrows   move that edge by 1 PIXEL   (Shift = 10 pixels)
         [ ]      marker size  -  +
         b B      background dim / bright
         s        save geometry      r  reset
         ESC/q    close
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpi import make_dpi_aware, monitors
MODE = make_dpi_aware()

import numpy as np, cv2
import tkinter as tk
from PIL import Image, ImageTk
from marker_ids import check_plate_ids

ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true")
ap.add_argument("--monitor", type=int)
ap.add_argument("--panel-w", type=float, help="VISIBLE display width in mm (measured)")
ap.add_argument("--ids", default="0,1,2,3")
ap.add_argument("--marker-mm", type=float, default=None)
ap.add_argument("--margin", type=float, default=8.0, help="quiet zone, mm")
ap.add_argument("--bg", type=int, default=255)
ap.add_argument("--inset", default="0,0,0,0", help="top,right,bottom,left mm hidden by structure")
ap.add_argument("--ruler", action="store_true")
ap.add_argument("--edge-ruler", action="store_true",
                help="red bars spanning the full framebuffer, edge to edge - measure between "
                     "them to get the panel size directly, with nothing assumed")
ap.add_argument("--name", default=None)
ap.add_argument("--square", action="store_true")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--sim", default=None, help="dry-run only: WxH px")
a = ap.parse_args()

mons = monitors()
if a.sim:
    w_, h_ = [int(v) for v in a.sim.lower().split("x")]
    mons = mons + [dict(x=0, y=0, w=w_, h=h_, primary=False, dev="SIMULATED")]
    a.monitor = len(mons) - 1
    a.dry_run = True

if a.list or a.monitor is None:
    print(f"  DPI awareness: {MODE}")
    for i, m in enumerate(mons):
        o = "portrait" if m["h"] > m["w"] else "landscape"
        print(f"  monitor {i}: origin ({m['x']},{m['y']})  {m['w']}x{m['h']} px {o}"
              f"{'  PRIMARY' if m['primary'] else ''}")
    if a.monitor is None:
        sys.exit("\n  pass --monitor N --panel-w MM")

if MODE == "NONE":
    sys.exit("  ABORT: no DPI awareness, physical scale cannot be guaranteed")
if a.panel_w is None and not a.edge_ruler:
    sys.exit("  --panel-w required: the measured visible width in mm")

mon = mons[a.monitor]

if a.edge_ruler:
    # Two dimension lines spanning the framebuffer exactly, each with perpendicular end
    # caps. Nothing is assumed - what you measure between a pair of caps IS the panel
    # size. A missing cap means that edge is hidden by the mount, so one picture answers
    # both "how big is it" and "can I see all of it".
    W, H = mon["w"], mon["h"]
    page = np.zeros((H, W, 3), np.uint8)

    RED = (0, 0, 255)
    t = max(3, W // 200)
    capH = max(40, H // 12)
    capW = max(40, W // 12)
    ymid, xmid = H // 2, W // 2

    # Horizontal dimension line, edge to edge, with VERTICAL caps at each end.
    cv2.line(page, (0, ymid), (W - 1, ymid), RED, t)
    cv2.rectangle(page, (0, ymid - capH // 2), (t - 1, ymid + capH // 2), RED, -1)
    cv2.rectangle(page, (W - t, ymid - capH // 2), (W - 1, ymid + capH // 2), RED, -1)

    # Vertical dimension line, edge to edge, with HORIZONTAL caps at each end.
    cv2.line(page, (xmid, 0), (xmid, H - 1), RED, t)
    cv2.rectangle(page, (xmid - capW // 2, 0), (xmid + capW // 2, t - 1), RED, -1)
    cv2.rectangle(page, (xmid - capW // 2, H - t), (xmid + capW // 2, H - 1), RED, -1)

    y = ymid + capH
    for line in ("MEASURE CAP TO CAP", "",
                 "4 caps visible =", "nothing is hidden", "",
                 f"{W} x {H} px"):
        if line:
            cv2.putText(page, line, (xmid + 16, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
        y += 26

    page = page[:, :, ::-1].copy()

    print(f"  monitor {a.monitor}: {W} x {H} px, caps at the exact framebuffer edges")
    print("  Measure cap to cap - that is the panel size, with nothing assumed.")
    print(f"  height/width must come out at {H}/{W} = {H/W:.4f}, which is the check.")
    print("  A missing cap means that edge is hidden by the mount.")

PITCH = (a.panel_w / mon["w"]) if a.panel_w else 1.0
PANEL_H = mon["h"] * PITCH
px = lambda v: int(round(v / PITCH))
if not a.edge_ruler:
    print(f"  monitor {a.monitor}: {mon['w']}x{mon['h']} px over {a.panel_w:.1f}x{PANEL_H:.1f} mm"
          f"  ({PITCH:.5f} mm/px)")

d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
EDGES = ["top", "right", "bottom", "left", "ALL"]
ruler_page = None

if a.edge_ruler:
    ruler_page = page
    a.ruler = True                 # reuse the ruler display path
elif a.ruler:
    ruler_page = np.full((mon["h"], mon["w"]), a.bg, np.uint8)
    y = mon["h"] // 2
    cv2.line(ruler_page, (px(10), y), (px(110), y), 0, 3)
    for t in range(11):
        x = px(10 + t * 10)
        cv2.line(ruler_page, (x, y), (x, y - px(7 if t % 5 == 0 else 4)), 0, 3)
    cv2.putText(ruler_page, "MEASURE THIS LINE", (px(10), y + px(9)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
    print("  SCALE CHECK: measure the drawn line with a real ruler.")
    print(f"    reads 100 mm -> --panel-w {a.panel_w:.1f} is correct")
    print(f"    reads M mm   -> --panel-w {a.panel_w:.1f} * M/100 :")
    for M in (90, 95, 105, 110):
        print(f"        {M} -> {a.panel_w * M / 100:.1f}")
    IDS = []
    ST = {}
else:
    IDS = [int(x) for x in a.ids.split(",")]
    ok, msg = check_plate_ids(IDS)
    if not ok:
        sys.exit(f"  invalid plate: {msg}")
    t_, r_, b_, l_ = [float(v) for v in a.inset.split(",")]
    ST = dict(top=t_, right=r_, bottom=b_, left=l_,
              mk=a.marker_mm or 0.0, bg=a.bg, edge=0, showText=True)


def build():
    """Render the plate for the current state; returns (image, geometry, status)."""
    ux0, uy0 = ST["left"], ST["top"]
    uw = a.panel_w - ST["left"] - ST["right"]
    uh = PANEL_H - ST["top"] - ST["bottom"]
    if uw <= 20 or uh <= 20:
        return None, None, "usable area too small"
    mk = ST["mk"] or min(40.0, 0.28 * min(uw, uh))
    modpx = max(3, int(round((mk / PITCH) / 6)))
    mkpx = modpx * 6
    mk = mkpx * PITCH
    off = a.margin + mk / 2
    if 2 * off >= min(uw, uh):
        return None, None, f"marker {mk:.0f} mm too big for the usable area"
    sw, sh = uw - 2 * off, uh - 2 * off
    if a.square:
        s = min(sw, sh)
        cx0, cy0 = ux0 + uw / 2 - s / 2, uy0 + uh / 2 - s / 2
        sw = sh = s
    else:
        cx0, cy0 = ux0 + off, uy0 + off
    CTR = [(cx0, cy0), (cx0 + sw, cy0), (cx0 + sw, cy0 + sh), (cx0, cy0 + sh)][:len(IDS)]

    page = np.full((mon["h"], mon["w"]), ST["bg"], np.uint8)
    cv2.rectangle(page, (px(ux0), px(uy0)), (px(ux0 + uw) - 1, px(uy0 + uh) - 1), 160, 1)
    for mid, (cx, cy) in zip(IDS, CTR):
        x, y = px(cx) - mkpx // 2, px(cy) - mkpx // 2
        x = max(0, min(x, mon["w"] - mkpx))
        y = max(0, min(y, mon["h"] - mkpx))
        page[y:y + mkpx, x:x + mkpx] = cv2.aruco.generateImageMarker(d, mid, mkpx, borderBits=1)

    P = np.array(CTR) - np.array(CTR).mean(0)
    sv = np.linalg.svd(P, compute_uv=False)
    asp = sv[0] / max(sv[1], 1e-9)
    verdict = "GOOD" if asp < 2 else ("MARGINAL" if asp < 3 else "TOO COLLINEAR")
    _, fids, _ = det.detectMarkers(page)
    n = 0 if fids is None else len(fids)

    txt = [f"edge: {EDGES[ST['edge']]}",
           f"inset mm  T{ST['top']:.1f} R{ST['right']:.1f} B{ST['bottom']:.1f} L{ST['left']:.1f}",
           f"inset px  T{ST['top']/PITCH:.0f} R{ST['right']/PITCH:.0f} "
           f"B{ST['bottom']/PITCH:.0f} L{ST['left']/PITCH:.0f}   ({PITCH:.4f} mm/px)",
           f"marker {mk:.1f} mm   spread {sw:.0f}x{sh:.0f} mm   {asp:.2f}:1 {verdict}",
           f"self-detect {n}/{len(IDS)}",
           "Tab edge  arrows 1px (Shift 10px)  [ ] size  h hide  s save  ESC"]

    # The overlay goes in the clear band BETWEEN the marker rows, never over a marker.
    # Dark text across a marker's white modules corrupts its bit pattern - it may still
    # decode, but only by luck, and the failure would look like a detection problem
    # rather than a drawing one.
    if ST["showText"]:
        band_top = cy0 + mk / 2.0
        band_bot = cy0 + sh - mk / 2.0
        line_mm = 6.5
        block = line_mm * len(txt)

        if band_bot - band_top >= block:
            y0 = band_top + (band_bot - band_top - block) / 2.0
            for i, t in enumerate(txt):
                cv2.putText(page, t, (px(ux0 + 2), px(y0 + line_mm * (i + 0.75))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, 60, 1)
        else:
            cv2.putText(page, "h for info", (px(ux0 + 2), px((band_top + band_bot) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, 60, 1)

    geom = dict(name=a.name or f"panel{a.monitor}", kind="display", monitor=a.monitor,
                panel_px=[mon["w"], mon["h"]],
                panel_mm=[round(a.panel_w, 2), round(PANEL_H, 2)],
                pitch_mm_px=round(PITCH, 6),
                usable_mm=dict(x=round(ux0, 2), y=round(uy0, 2),
                               w=round(uw, 2), h=round(uh, 2)),
                inset_mm=dict(top=ST["top"], right=ST["right"],
                              bottom=ST["bottom"], left=ST["left"]),
                marker_mm=round(mk, 3), ids=IDS,
                centres_mm=[[round(c, 3) for c in p] for p in CTR],
                aspect=round(float(asp), 3), verdict=verdict,
                self_detect=f"{n}/{len(IDS)}")
    return page, geom, " | ".join(txt[1:4])


if a.dry_run:
    if not a.ruler:
        _, _, status = build()
        print(f"  {status}")
    print("  dry run - nothing displayed")
    sys.exit(0)

root = tk.Tk(); root.withdraw()
win = tk.Toplevel(); win.overrideredirect(True)
win.geometry(f"{mon['w']}x{mon['h']}+{mon['x']}+{mon['y']}")
win.attributes("-topmost", True)
cvs = tk.Canvas(win, width=mon["w"], height=mon["h"], highlightthickness=0, borderwidth=0)
cvs.pack()
holder = {}


def redraw():
    if a.ruler:
        img = ruler_page
    else:
        img, geom, status = build()
        if img is None:
            print(f"  {status}")
            return
        holder["geom"] = geom
    holder["ph"] = ImageTk.PhotoImage(Image.fromarray(img))
    cvs.delete("all")
    cvs.create_image(0, 0, anchor="nw", image=holder["ph"])


def _close(_=None):
    try:
        root.destroy()
    except Exception:
        pass


def key(e):
    if a.ruler:
        return
    k = e.keysym
    # Pixels, not millimetres. The render is pixel-quantised so a fractional-pixel step
    # changes nothing visible, and on a fine panel 1 mm can be several pixels - too
    # coarse for nudging a marker clear of a bezel lip.
    step = PITCH * (10.0 if (e.state & 0x1) else 1.0)
    if k == "Tab":
        ST["edge"] = (ST["edge"] + 1) % len(EDGES)
    elif k in ("Up", "Down", "Left", "Right"):
        sign = {"Up": -1, "Left": -1, "Down": 1, "Right": 1}[k]
        vert = k in ("Up", "Down")
        sel = EDGES[ST["edge"]]
        if sel == "ALL":
            tgt = ["top", "bottom"] if vert else ["left", "right"]
        elif (sel in ("top", "bottom")) == vert:
            tgt = [sel]
        else:
            tgt = []
        for t in tgt:
            delta = sign * step * (1 if t in ("top", "left") else -1)
            ST[t] = max(0.0, ST[t] + delta)
    elif k == "bracketleft":
        ST["mk"] = max(8.0, (ST["mk"] or 30.0) - 1)
    elif k == "bracketright":
        ST["mk"] = min(60.0, (ST["mk"] or 30.0) + 1)
    elif k == "b":
        ST["bg"] = max(40, ST["bg"] - 15)
    elif k == "B":
        ST["bg"] = min(255, ST["bg"] + 15)
    elif k == "h":
        # Hide the overlay once placement is settled, so nothing but markers is on the
        # panel while it is being used for real.
        ST["showText"] = not ST["showText"]
    elif k == "r":
        ST.update(top=0.0, right=0.0, bottom=0.0, left=0.0, mk=0.0, bg=a.bg)
    elif k == "s":
        g = holder.get("geom")
        if g:
            os.makedirs("PRINT-THESE/plates", exist_ok=True)
            p = f"PRINT-THESE/plates/plate-{g['name']}.json"
            json.dump(g, open(p, "w"), indent=2)
            print(f"  saved {p}")
            print(f"    usable {g['usable_mm']}  marker {g['marker_mm']} mm  "
                  f"{g['aspect']}:1 {g['verdict']}")
        return
    else:
        return
    redraw()


# Deliberately NOT click-to-close, unlike show_1to1.py. This window is an editor: you
# have to click it to give it keyboard focus, and closing on that click makes the Tab and
# arrow controls unreachable. Click focuses instead.
def _focus(_=None):
    try:
        win.focus_force()
    except Exception:
        pass

for w in (win, cvs):
    for b in ("<Escape>", "<KeyPress-q>"):
        w.bind(b, _close)
    w.bind("<Button-1>", _focus)
    w.bind("<Key>", key)
root.bind_all("<Escape>", _close)
root.bind_all("<Key>", key)
win.focus_force()


def _poll():
    import ctypes
    if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
        _close()
        return
    root.after(80, _poll)


root.after(80, _poll)
redraw()
print("  adjust with Tab/arrows, s to save, ESC to close")
root.mainloop()
