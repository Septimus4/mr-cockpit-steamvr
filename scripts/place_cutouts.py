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

from anchors.camera_rig import (
    CAMERA_OFFSET, CORNER_BIAS_PX as CAMERA_CORNER_BIAS, apply_offset_delta,
)
from anchors.detect import plate_size_overrides, refresh_observations
from anchors.place import (
    cover_all, flattening_cost_mm, measure_range_scale, place_from_markers,
    place_from_plate, shaped_cutout,
)
from anchors.solver import Observation, solve_markers
from tracing.config_io import (
    DEFAULT_CONFIG_PATH, MAX_POINTS, MAX_QUADS, QuadConfig, read_quads, write_quad,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLATES = os.path.join(ROOT, "PRINT-THESE", "plates")

# Above this, the solved constellation disagrees with the panel's measured geometry by
# more than the out-of-plane budget allows. Measured on hardware: good captures land at
# 1.8-7.0 mm against a ~20 mm budget, so 12 mm means something is wrong rather than noisy.
RESIDUAL_WARN_MM = 12.0


# The settings menu holds the whole config in memory and writes ALL of it back whenever
# anything changes. A file written underneath it is not merged, it is discarded the moment
# the user touches a slider - and the symptom is the worst kind: the tool reports success,
# the layer loads the OLD values, and it looks like placement simply does not work.
MENU_PROCESS = "passthrough-menu.exe"


def menu_is_running():
    """
    Whether the passthrough settings menu is up. None if it cannot be determined.

    Not knowing is reported as not knowing: refusing to write on a failed check would
    block the tool on machines where tasklist is unavailable, and writing silently would
    reintroduce exactly the bug this guards.
    """
    import subprocess

    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {MENU_PROCESS}"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None

    if out.returncode != 0:
        return None

    return MENU_PROCESS.lower() in out.stdout.lower()


def _polygon_area(points):
    """Shoelace area, for reporting how much of the bounding box an outline actually uses."""
    n = len(points)

    if n < 3:
        return 0.0

    return abs(sum(points[i][0] * points[(i + 1) % n][1] -
                   points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0


def reset_quads(config_path, force):
    """
    Put every cutout back to disabled and empty.

    A clean slate matters more than it sounds. Cutouts are hand-tuned, overwritten by the
    menu, and re-placed by this tool, so a config can end up a mixture of three different
    attempts - and a stale ENABLED quad from an old attempt looks exactly like the new
    placement being wrong.
    """
    import shutil

    if not os.path.exists(config_path):
        print(f"  No config at {config_path}")
        return 1

    running = menu_is_running()

    if running and not force:
        print(f"  REFUSING: {MENU_PROCESS} is running and would write its copy back over")
        print("  this. Close it, run again, then reopen it.")
        return 1

    backup = config_path + ".bak"
    shutil.copy2(config_path, backup)

    existing = read_quads(config_path)
    had = [q.label for q in existing if q.enabled]

    for i in range(MAX_QUADS):
        write_quad(QuadConfig(i), config_path)

    print(f"  {MAX_QUADS} cutouts cleared" + (f" (was: {', '.join(had)})" if had else ""))
    print(f"  Previous config kept at {backup}")
    print()
    print("  Place them again with:")
    print("    python scripts/place_cutouts.py --exclude-screens --write")

    return 0


def calibrate_corner_bias(plates, observations, cameras):
    """
    Solve the detector's corner bias, in camera pixels, from this capture.

    The marker's size is known EXACTLY - a whole number of rendered pixels at a measured
    pitch - so the thing to calibrate is not the size but how far inside the true edge the
    detector puts the corners. That is one number for the camera, not one per panel, and
    it stays correct at any distance, which a faked size does not.

    Search is a plain scan: the cost is smooth in one variable and the whole range of
    plausible bias is two pixels wide.
    """
    from anchors.detect import dilate_corners
    from anchors.place import measure_range_scale
    from anchors.solver import Observation

    def scale_at(bias):
        obs = [Observation(o.marker_id, dilate_corners(o.corners_px, bias),
                           o.camera_to_world, o.size_mm, o.frame) for o in observations]
        rows = measure_range_scale(plates, solve_markers(obs, cameras))

        return (float(np.mean([k for _, k, _ in rows])), rows) if rows else (None, [])

    print("  Solving the detector's corner bias. The marker size stays exactly as drawn.")
    print()
    print(f"  These observations already carry the current {CAMERA_CORNER_BIAS:.2f} px "
          f"correction, so the")
    print("  scan is for what is left on top of it. The total is what goes in the file.")
    print()
    print(f"  {'total px':>9} {'mean scale':>12}   (want 1.0000)")

    best = None

    for bias in np.arange(-1.0, 1.51, 0.125):
        mean, rows = scale_at(float(bias))

        if mean is None:
            print("  No plate had three solved markers, so there is nothing to calibrate.")
            return 1

        print(f"  {CAMERA_CORNER_BIAS + bias:9.3f} {mean:12.4f}")

        if best is None or abs(mean - 1.0) < abs(best[1] - 1.0):
            best = (float(bias), mean, rows)

    extra, mean, rows = best
    bias = CAMERA_CORNER_BIAS + extra

    print()
    print(f"  BEST {bias:.3f} px per edge, mean scale {mean:.4f}")

    for name, k, residual in rows:
        print(f"    {name:14} scale {k:.4f}  residual {residual:.2f} mm")

    print()
    if abs(extra) < 1e-6:
        print(f"  That is the value already in anchors/camera_rig.py - nothing to change.")
    else:
        print(f"  Put this in anchors/camera_rig.py:   CORNER_BIAS_PX = {bias:.2f}")
    print()
    print("  It is a property of THIS camera against THIS kind of surface. Re-solve it if")
    print("  the exposure or the panel brightness changes, and expect a different number")
    print("  for printed stickers, which do not glow.")

    return 0


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
    ap.add_argument("--force", action="store_true",
                    help="write even if the settings menu is running. It will very "
                         "likely be overwritten; close the menu instead.")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="grow every cutout by this many mm on all sides")
    ap.add_argument("--start", type=int, default=0,
                    help="first quad index to use, so earlier ones are left alone")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure each display plate's EFFECTIVE marker size from this "
                         "capture and write it into the plate JSON, so every future "
                         "placement is correct without --range-scale.")
    ap.add_argument("--reset", action="store_true",
                    help="disable and clear EVERY cutout, then stop. Use when the config "
                         "has been hand-edited into a state you no longer trust.")
    ap.add_argument("--range-scale", type=float, default=1.0, metavar="K",
                    help="scale every solved RANGE by K. Below 1 pulls the pit closer. "
                         "Use when the cutouts look both too small and too far away - "
                         "that is one error, not two.")
    ap.add_argument("--exclude-screens", action="store_true",
                    help="cut each MFD screen out of the outline, so the SIM draws the "
                         "display and passthrough covers only the buttons. Implies "
                         "--shaped.")
    ap.add_argument("--screen-shrink", type=float, default=3.0,
                    help="pull each screen hole in by this many mm (default 3). Alignment "
                         "error then eats into the bezel instead of leaving a ring of "
                         "camera over the screen edge.")
    ap.add_argument("--shaped", action="store_true",
                    help="ONE cutout whose OUTLINE follows the panels - a T for three "
                         "across and one below - instead of a rectangle around them.")
    ap.add_argument("--cover-all", nargs="?", const="530x430", default=None,
                    metavar="WxH",
                    help="ONE cutout over the whole assembly instead of one per panel, "
                         "in mm (default 530x430). The console carries no markers, so "
                         "this is the only way to reach it today.")
    a = ap.parse_args()

    if a.reset:
        return reset_quads(a.config, a.force)

    if not os.path.exists(a.capture):
        print(f"  No capture at {a.capture}")
        print("  Sweep the cockpit first:  python scripts/solve_anchors.py")
        return 1

    z = np.load(a.capture, allow_pickle=True)
    observations = list(z["observations"])
    cameras = {int(k): v for k, v in zip(z["frames"], z["cameras"])}

    capture_offset = z["camera_offset"] if "camera_offset" in z.files else None
    cameras, delta = apply_offset_delta(cameras, capture_offset)

    if np.any(delta):
        print(f"  capture predates the current camera calibration; shifting cameras by "
              f"{np.round(delta * 1000, 1)} mm")

    # Marker sizes are re-read from the plates, so a calibration reaches captures already
    # taken. Without this, --calibrate would only help the NEXT sweep - and the number it
    # writes was measured from the sweep you already have.
    capture_bias = float(z["corner_bias"]) if "corner_bias" in z.files else 0.0
    observations, resized, bias_delta = refresh_observations(
        observations, plate_size_overrides(), capture_bias)

    if abs(bias_delta) > 1e-9:
        print(f"  correcting detector corner bias by {bias_delta:+.2f} px "
              f"(capture had {capture_bias:.2f}, calibration is {CAMERA_CORNER_BIAS:.2f})")

    if resized:
        first = next(iter(resized.values()))
        print(f"  marker sizes updated for {len(resized)} ids "
              f"({first[0]:.3f} -> {first[1]:.3f} mm and similar)")

    if a.range_scale != 1.0:
        # Range is inferred from a marker's APPARENT size: range = focal * size / pixels.
        # So scaling range IS scaling the assumed marker size - applying it there rather
        # than shifting the result keeps every direction untouched and only moves depth,
        # which is the whole point. Nudging PosZ instead rotates the cutout off its panel,
        # because world Z is not the direction from the eye to the panel.
        observations = [Observation(o.marker_id, o.corners_px, o.camera_to_world,
                                    o.size_mm * a.range_scale, o.frame)
                        for o in observations]
        print(f"  range scaled by {a.range_scale:.3f} - equivalent to every marker being "
              f"{(1 - a.range_scale) * 100:.1f}% smaller than declared")

    solutions = solve_markers(observations, cameras)
    plates = load_display_plates()

    print(f"  {len(observations)} observations over {len(cameras)} frames")
    print(f"  {len(solutions)} marker(s) solved, {len(plates)} display plate(s) configured")

    measured = measure_range_scale(plates, solutions)

    if measured and a.range_scale == 1.0:
        best_name, best_k, best_res = measured[0]

        if abs(best_k - 1.0) > 0.02:
            print()
            print("  RANGE LOOKS WRONG, measured against the plates' known geometry:")
            print(f"    {'plate':14} {'scale':>7} {'residual':>10}   correction")

            for name, k, res in measured:
                print(f"    {name:14} {k:7.4f} {res:9.2f} mm   --range-scale {1 / k:.3f}")

            print()
            print(f"  Best conditioned is {best_name} at {best_k:.4f}, so the solve places")
            print(f"  everything {(best_k - 1) * 100:+.1f}% too far. That is why a cutout can look")
            print("  too SMALL and too FAR at once - a fixed size further away subtends less.")
            print(f"    python scripts/place_cutouts.py --range-scale {1 / best_k:.3f} ...")
            print()
            print("  A uniform pixel-pitch error would NOT show here - it scales the marker")
            print("  size and the spacing together and cancels. This means the two disagree.")

    if a.calibrate:
        return calibrate_corner_bias(plates, observations, cameras)

    placed, loose = placements(solutions, plates, a.margin)

    if a.shaped or a.exclude_screens:
        head_now = head_frame(cameras)
        viewpoint = head_now[0] if head_now else np.array([0.0, 1.2, 0.0])
        by_name = {g["name"]: g for g in plates} if a.exclude_screens else None
        single = shaped_cutout(placed, solutions, viewpoint, a.margin,
                               exclude_screens=by_name,
                               screen_shrink_mm=a.screen_shrink)

        if single is None:
            print("\n  Not enough solved markers to fit a common plane.")
            return 1

        box = single.width * single.height
        area = _polygon_area(single.points)

        print(f"\n  Outline follows the {len(placed)} panels: {len(single.points)} points, "
              f"{area * 1e4:.0f} square cm")
        print(f"  against {box * 1e4:.0f} for a rectangle around them - {100 * area / box:.0f}% "
              f"of the box.")
        print("  The rest of the box is cockpit side wall, and passthrough there would")
        print("  cover the game rather than reveal a control.")

        if a.exclude_screens:
            print()
            print(f"  {len(placed)} screen(s) cut out, shrunk {a.screen_shrink:.0f} mm.")
            print("  The sim draws the MFDs; passthrough covers only the buttons around")
            print("  them. The shrink makes alignment error land on the bezel, where a")
            print("  little game is invisible, rather than on the screen, where a little")
            print("  camera is not.")

            if single.dropped_holes:
                print()
                print(f"  {single.dropped_holes} screen hole(s) DROPPED - hit the point limit")
                print(f"  the {MAX_POINTS}-point config limit. Those screens will be covered by")
                print("  passthrough. Use --shaped without --exclude-screens, or place the")
                print("  panels as separate cutouts.")

        placed = [single]
        loose = []

    elif a.cover_all:
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
    describe(placed, loose, solutions, head, single=bool(a.cover_all or a.shaped or a.exclude_screens))

    if head is not None:
        # An outlined cutout covers its polygon, not its bounding box, and the
        # difference is the whole point of having an outline.
        total = sum(_polygon_area(c.points) or c.width * c.height
                    for c in placed) * 1e4
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

    running = menu_is_running()

    if running and not a.force:
        print(f"\n  REFUSING TO WRITE: {MENU_PROCESS} is running.")
        print()
        print("  The menu holds the whole config in memory and writes ALL of it back the")
        print("  moment anything changes. Whatever is written now is discarded as soon as")
        print("  you touch a slider - and it fails silently: this tool reports success,")
        print("  the layer loads the old values, and placement looks broken.")
        print()
        print("  Close the menu, run this again, then reopen the menu.")
        print("  --force overrides, but the write will probably be lost.")
        return 1

    if running is None:
        print(f"\n  Could not tell whether {MENU_PROCESS} is running. If it is, close it")
        print("  first - it overwrites this file wholesale when anything changes.")

    backup = a.config + ".bak"
    shutil.copy2(a.config, backup)

    existing = read_quads(a.config)

    for n, c in enumerate(placed):
        index = a.start + n
        quad = QuadConfig(index, enabled=True, name=c.name,
                          position=c.position, euler_deg=c.euler_deg,
                          width=c.width, height=c.height,
                          # Any outline comes from THIS placement. Keeping whatever was
                          # there before would silently apply another panel's shape.
                          points=getattr(c, "points", []))
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
