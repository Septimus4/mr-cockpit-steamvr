"""
Put markers on every configured display panel at once.

Each panel needs its own window on its own monitor, so this launches one show_plate.py
per saved plate and keeps them up until you stop it. Geometry comes from the plate JSON
files, so what appears is what was measured, not what was typed again.

  python scripts/show_all_plates.py            markers up, ENTER to take them down
  python scripts/show_all_plates.py --list     show what is configured and exit
  python scripts/show_all_plates.py --overlay  keep the status text (hidden by default)
"""

import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLATES = os.path.join(ROOT, "PRINT-THESE", "plates")


def load_plates():
    out = []

    for path in sorted(glob.glob(os.path.join(PLATES, "plate-*.json"))):
        with open(path) as f:
            g = json.load(f)

        if g.get("kind") != "display":
            continue                    # sticker guides are printed, not shown

        out.append(g)

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--overlay", action="store_true",
                    help="keep the status overlay; hidden by default so the panels show "
                         "nothing but markers")
    a = ap.parse_args()

    plates = load_plates()

    if not plates:
        print(f"  No display plates in {PLATES}")
        print("  Measure one first:  python scripts/show_plate.py --monitor N --panel-w MM --ruler")
        return 1

    print(f"  {len(plates)} display plate(s) configured:\n")
    for g in plates:
        u, i = g["usable_mm"], g["inset_mm"]
        print(f"    {g['name']:14} monitor {g['monitor']}  ids {g['ids']}  "
              f"marker {g['marker_mm']} mm  {g['aspect']}:1 {g['verdict']}")
        print(f"    {'':14} usable {u['w']:.1f} x {u['h']:.1f} mm, "
              f"inset T{i['top']:.2f} R{i['right']:.2f} B{i['bottom']:.2f} L{i['left']:.2f}")

    # Ids must be unique across panels: two markers claiming one identity corrupts pose
    # solving rather than degrading it.
    ids = [i for g in plates for i in g["ids"]]
    dupes = sorted({i for i in ids if ids.count(i) > 1})

    if dupes:
        print(f"\n  DUPLICATE IDS {dupes} across panels. Two physically different markers")
        print("  would claim one identity, which corrupts the solve. Fix before using these.")
        return 1

    if a.list:
        return 0

    procs = []
    print()

    for g in plates:
        i = g["inset_mm"]
        cmd = [sys.executable, os.path.join(HERE, "show_plate.py"),
               "--monitor", str(g["monitor"]),
               "--panel-w", str(g["panel_mm"][0]),
               "--inset", f"{i['top']},{i['right']},{i['bottom']},{i['left']}",
               "--ids", ",".join(str(x) for x in g["ids"]),
               "--name", g["name"],
               "--marker-mm", str(g["marker_mm"])]

        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
        print(f"    up: {g['name']} on monitor {g['monitor']}, ids {g['ids']}")

    if not a.overlay:
        print("\n  Press h on each window to hide its status text, so the panel shows")
        print("  nothing but markers. Detection works either way - the text is drawn")
        print("  between the markers, never over one.")

    print("\n  Markers are up. Solve them with:")
    print("    python scripts/solve_anchors.py")
    print("\n  ENTER here takes them down.")

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    for p in procs:
        p.terminate()

    print("  markers down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
