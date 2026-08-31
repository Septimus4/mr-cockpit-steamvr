"""
Turn solved markers into cutouts in the layer's config.

This closes the loop. solve_anchors.py finds where the markers are; this puts a cutout on
each panel they belong to, so the passthrough hole lands on the physical panel instead of
on eighteen numbers typed by hand.

It replays the LAST CAPTURE by default - no camera, no headset. Sweep once with
solve_anchors.py, then place as many times as you like.

Nothing is written without --write. The layer and the settings menu both own config.ini,
so a dry run that shows exactly what would change is the sane default.

  python scripts/place_cutouts.py                 show what would be written
  python scripts/place_cutouts.py --write         write it
  python scripts/place_cutouts.py --margin 4      grow each cutout 4 mm all round
  python scripts/place_cutouts.py --start 2       leave Quad0 and Quad1 alone

A cutout is only as good as its solve. Anything the capture reported as square-on or
badly scattered is reported again here, because it is here that it becomes visible.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchors.place import (
    cover_all, flattening_cost_mm, place_from_markers, place_from_plate,
)
from anchors.solver import solve_markers
from tracing.config_io import MAX_QUADS, DEFAULT_CONFIG_PATH, QuadConfig, read_quads, write_quad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATES = os.path.join(ROOT, "PRINT-THESE", "plates")

# Above this, the solved constellation disagrees with the panel's measured geometry by
# more than the out-of-plane budget allows. Measured on hardware: good captures land at
# 1.8-7.0 mm against a ~20 mm budget, so 12 mm means something is wrong rather than noisy.
RESIDUAL_WARN_MM = 12.0


def load_display_plates():
    out = []

    for path in sorted(glob.glob(os.path.join(PLATES, "plate-*.json"))):
        with open(path) as f:
            g = json.load(f)

        if g.get("kind") == "display":
            out.append(g)

    return out


def placements(solutions, plates, margin_mm):
    """Cutouts for every plate with enough solved markers, and what was left over."""
    placed = []
    claimed = set()

    for plate in plates:
        got = place_from_plate(plate, solutions, margin_mm)

        if got is None:
            continue

        placed.append(got)
        claimed.update(got.marker_ids)

    loose = sorted(i for i in solutions if i not in claimed)
    return placed, loose


def head_frame(cameras):
    """
    Where the head was during the capture, and which way it faced.

    Returns (position, yaw_forward) in stage coordinates, or None. Used only to say where
    to LOOK: a cutout is a small object, and "0.37 m away, 20 degrees down" is the
    difference between finding it and concluding the feature is broken.
    """
    if not cameras:
        return None

    cams = list(cameras.values())
    position = np.mean([c.camera_to_world[:3, 3] for c in cams], axis=0)

    # Camera forward is +Z in the OpenCV frame the capture uses.
    forward = np.mean([c.camera_to_world[:3, :3] @ np.array([0.0, 0.0, 1.0])
                       for c in cams], axis=0)
    forward[1] = 0.0                     # yaw only; pitch is what we are reporting
    n = np.linalg.norm(forward)

    if n < 1e-6:
        return None

    return position, forward / n


def where_to_look(placement, head):
    """One line saying where a cutout is, from where the capture was taken."""
    position, forward = head
    d = placement.position - position

    right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)

    ahead = float(d @ forward)
    across = float(d @ right)
    up = float(d[1])

    horizontal = float(np.hypot(ahead, across))
    pitch = np.degrees(np.arctan2(up, horizontal)) if horizontal > 1e-6 else 0.0
    yaw = np.degrees(np.arctan2(across, ahead)) if abs(ahead) > 1e-6 else 0.0

    return (f"{np.linalg.norm(d):.2f} m away, look "
            f"{abs(pitch):.0f} deg {'up' if pitch > 0 else 'down'} and "
            f"{abs(yaw):.0f} deg {'right' if yaw > 0 else 'left'}")


def describe(placed, loose, solutions, head=None, single=False):
    """
    `single` says the placements came from --cover-all, where flatness_mm means the
    markers' spread about a COMMON plane rather than one panel's fit residual. Reporting
    the two the same way would turn an expected and already-priced cost into a fault.
    """
    for c in placed:
        p = c.position
        print(f"\n  {c.name}")
        print(f"    position  {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}  m")
        print(f"    rotation  {c.euler_deg[0]:+.2f} {c.euler_deg[1]:+.2f} "
              f"{c.euler_deg[2]:+.2f}  deg")
        print(f"    size      {c.width * 1000:.1f} x {c.height * 1000:.1f} mm")
        print(f"    markers   {c.marker_ids}")
        print(f"    fit       {c.flatness_mm:.1f} mm "
              f"{'out of a common plane' if single else 'residual'}, "
              f"{c.spread_mm:.1f} mm mean per-view spread")

        if head is not None:
            print(f"    from you  {where_to_look(c, head)}")

        if not single and c.flatness_mm > RESIDUAL_WARN_MM:
            print(f"    ^ the solved markers disagree with this panel's measured geometry")
            print(f"      by {c.flatness_mm:.0f} mm. Either a marker moved, or the panel was")
            print(f"      measured wrong. Placing it will land the cutout visibly off.")

        square_on = [i for i in c.marker_ids
                     if solutions[i].max_skew < 0.06]
        if square_on:
            print(f"    ^ markers {square_on} were only ever seen SQUARE-ON, which is")
            print(f"      ambiguous. Capture again looking at this panel from an angle.")

    if loose:
        print(f"\n  {len(loose)} marker(s) belong to no display plate: {loose}")
        print("  These are stickers. They anchor the cockpit frame; they do not define a")
        print("  panel, so no cutout is made from them.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", default="captures/anchors.npz",
                    help="capture to replay (default captures/anchors.npz)")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--write", action="store_true",
                    help="actually write. Without this it only shows what it would do.")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="grow every cutout by this many mm on all sides")
    ap.add_argument("--start", type=int, default=0,
                    help="first quad index to use, so earlier ones are left alone")
    ap.add_argument("--cover-all", nargs="?", const="530x430", default=None,
                    metavar="WxH",
                    help="ONE cutout over the whole assembly instead of one per panel, "
                         "in mm (default 530x430). The console carries no markers, so "
                         "this is the only way to reach it today.")
    a = ap.parse_args()

    if not os.path.exists(a.capture):
        print(f"  No capture at {a.capture}")
        print("  Sweep the cockpit first:  python scripts/solve_anchors.py")
        return 1

    z = np.load(a.capture, allow_pickle=True)
    observations = list(z["observations"])
    cameras = {int(k): v for k, v in zip(z["frames"], z["cameras"])}

    solutions = solve_markers(observations, cameras)
    plates = load_display_plates()

    print(f"  {len(observations)} observations over {len(cameras)} frames")
    print(f"  {len(solutions)} marker(s) solved, {len(plates)} display plate(s) configured")

    placed, loose = placements(solutions, plates, a.margin)

    if a.cover_all:
        try:
            w_mm, h_mm = (float(v) for v in a.cover_all.lower().split("x"))
        except ValueError:
            print(f"  --cover-all wants WxH in mm, for example 530x430, not {a.cover_all!r}")
            return 1

        head_now = head_frame(cameras)
        viewpoint = head_now[0] if head_now else np.array([0.0, 1.2, 0.0])
        single = cover_all(placed, solutions, w_mm + a.margin, h_mm + a.margin, viewpoint)

        if single is None:
            print("\n  Not enough solved markers to fit a common plane.")
            return 1

        distance = float(np.linalg.norm(single.position - viewpoint)) if head_now else 0.45
        cost = flattening_cost_mm(single.flatness_mm, distance)

        print(f"\n  Covering all {len(placed)} panels with ONE cutout.")
        print(f"  The markers sit within {single.flatness_mm:.0f} mm of a common plane over")
        print(f"  their {w_mm:.0f} x {h_mm:.0f} mm span, which at {distance:.2f} m costs about "
              f"{cost:.0f} mm of")
        print("  sideways misalignment where the surfaces bow away from it. Fine for")
        print("  reaching buttons; per-panel cutouts are tighter if you want the edges to")
        print("  line up.")

        placed = [single]
        loose = []

    if not placed:
        print("\n  No panel had three solved markers, so nothing can be placed.")
        print("  Put the markers up (python scripts/show_all_plates.py) and sweep again.")
        return 1

    head = head_frame(cameras)
    describe(placed, loose, solutions, head, single=bool(a.cover_all))

    if head is not None:
        total = sum(c.width * c.height for c in placed) * 1e4
        print(f"\n  Together these cutouts are {total:.0f} square cm of passthrough, and")
        print("  they are the ONLY passthrough if QuadsExclusive is on.")

        # Below roughly a hand's span, a cutout is easy to miss entirely - which is what
        # happened on the first real placement, and reads as the feature being broken.
        if total < 600.0:
            print("  That is a small target: looking straight ahead you will see nothing")
            print("  at all. Look where the lines above say, or run --cover-all once to")
            print("  put one big hole over the whole assembly and find it that way.")

    if a.start + len(placed) > MAX_QUADS:
        print(f"\n  {len(placed)} cutouts from index {a.start} would exceed the layer's "
              f"{MAX_QUADS}-quad limit.")
        return 1

    if not a.write:
        print(f"\n  Dry run. Add --write to put these into {a.config}")
        return 0

    if not os.path.exists(a.config):
        print(f"\n  No config at {a.config}")
        print("  Run the passthrough layer once so it writes its defaults.")
        return 1

    backup = a.config + ".bak"
    shutil.copy2(a.config, backup)

    existing = read_quads(a.config)

    for n, c in enumerate(placed):
        index = a.start + n
        quad = QuadConfig(index, enabled=True, name=c.name,
                          position=c.position, euler_deg=c.euler_deg,
                          width=c.width, height=c.height,
                          # Outlines are per-panel artwork; a rectangle is the honest
                          # default and keeping any existing one would silently apply the
                          # wrong shape to a newly placed panel.
                          points=[])
        write_quad(quad, a.config)
        print(f"  Quad{index} <- {c.name}"
              + ("  (replaced " + existing[index].label + ")"
                 if existing[index].enabled else ""))

    print(f"\n  Written. Previous config kept at {backup}")
    print("  Restart the game, or reload the config from the passthrough menu.")
    print("\n  If a cutout sits beside its panel rather than on it, the camera offset is")
    print("  the likely cause - it is the one number in this chain that was measured with")
    print("  a ruler rather than solved.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
