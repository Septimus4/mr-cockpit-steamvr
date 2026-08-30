"""
Display an image at EXACTLY one image pixel per physical screen pixel.

Ordinary viewers scale silently - Windows DPI scaling, fit-to-window, PDF "100%" zoom.
This declares per-monitor-v2 DPI awareness so Windows does not touch the window, and
REPORTS the awareness mode so a failure is visible rather than silent.

  python scripts/show_1to1.py [image] [-m N]

  -m   monitor index (default: the portrait one, else primary).  Indices are printed.

  ESC / Q to close.  Then measure the 100 mm reference with a real ruler.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpi import make_dpi_aware, monitors      # must run before any window exists

MODE = make_dpi_aware()

import tkinter as tk
from PIL import Image, ImageTk

ap = argparse.ArgumentParser()
ap.add_argument("image", nargs="?", default="PRINT-THESE/screen-size-test.png")
ap.add_argument("-m", type=int, default=None)
a = ap.parse_args()

print(f"  DPI awareness: {MODE}")
if MODE == "NONE":
    print("  ABORT: cannot guarantee 1:1 without DPI awareness."); sys.exit(1)

mons = monitors()
for i, m in enumerate(mons):
    print(f"  monitor {i}: origin ({m['x']},{m['y']})  {m['w']}x{m['h']}"
          f"{'  portrait' if m['h']>m['w'] else ''}{'  PRIMARY' if m['primary'] else ''}")

if a.m is not None: mon = mons[a.m]
else:
    port = [m for m in mons if m['h'] > m['w']]
    mon = port[0] if port else next(m for m in mons if m['primary'])

img = Image.open(a.image)
print(f"\n  {a.image}: {img.width}x{img.height} px on a {mon['w']}x{mon['h']} monitor")
if img.width > mon['w'] or img.height > mon['h']:
    print(f"  WARNING: image is larger than the monitor - it WILL be cropped, "
          f"markers near the edges may be cut off")
else:
    print(f"  fits with {mon['w']-img.width} x {mon['h']-img.height} px to spare")

root = tk.Tk(); root.withdraw()
win = tk.Toplevel(); win.overrideredirect(True)
win.geometry(f"{mon['w']}x{mon['h']}+{mon['x']}+{mon['y']}")
win.configure(bg="white"); win.attributes("-topmost", True)
ph = ImageTk.PhotoImage(img)
cv = tk.Canvas(win, width=mon['w'], height=mon['h'], bg="white",
               highlightthickness=0, borderwidth=0)
cv.pack()
cv.create_image((mon['w']-img.width)//2, (mon['h']-img.height)//2, anchor="nw", image=ph)
# an overrideredirect window may never take keyboard focus, so bind on all widgets,
# bind_all as a catch-all, poll as a last resort, and accept a click to close.
def _close(_=None):
    try: root.destroy()
    except Exception: pass
for w in (win, cv):
    for k in ("<Escape>", "<KeyPress-q>", "<KeyPress-Q>", "<Button-1>"):
        w.bind(k, _close)
root.bind_all("<Escape>", _close); root.bind_all("<KeyPress-q>", _close)
win.focus_force()
try:
    win.grab_set_global()          # keyboard goes here even without focus
except Exception: pass
def _poll():                        # survives focus being stolen entirely
    import ctypes
    if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000: _close(); return
    root.after(80, _poll)
root.after(80, _poll)
print("  ESC, Q, or click to close.  Measure the 100 mm reference with a real ruler.")
root.mainloop()
