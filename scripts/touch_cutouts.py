"""
Build cutouts by touching your cockpit with a controller.

Point at a real corner, pull the trigger, move to the next one. Four corners is a panel;
walk the edge for a console. That is the whole calibration.

No camera, no markers, no sweep, no holding still, no lens model, no marker-size
calibration, no range scale. The controller tip is already in the same coordinates the
cutouts live in, so a touched corner IS the corner - to Lighthouse accuracy, which is far
better than anything the camera path could reach.

The camera and markers keep the job they are good at: re-anchoring at runtime so a cutout
measured once stays put. They were never able to measure a BUTTON, only the screen that
draws them, and the buttons are the point.

  python scripts/touch_cutouts.py --tip          calibrate the tip, once per controller
  python scripts/touch_cutouts.py                measure cutouts
  python scripts/touch_cutouts.py --start 2      leave Quad0 and Quad1 alone

TRIGGER  record a point          GRIP  undo the last point
MENU     finish this cutout       MENU with nothing pending  finish and write
Ctrl+C   quit without writing
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchors.probe import (
    compare_fingerprints, fit_pivot, outline_from_touches, pivot_conditioning,
    pivot_uncertainty, tip_position,
)
from tracing.capture import hmd_matrix_to_numpy
from tracing.config_io import DEFAULT_CONFIG_PATH, MAX_QUADS, QuadConfig, read_quads, write_quad

TIP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config-backups", "controller-tip.json")

# The settings menu rewrites the whole config whenever anything changes, so anything
# written underneath it is discarded the moment a slider moves. See place_cutouts.py.
MENU_PROCESS = "passthrough-menu.exe"


def menu_is_running():
    import subprocess

    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {MENU_PROCESS}"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None

    return MENU_PROCESS.lower() in out.stdout.lower() if out.returncode == 0 else None


def survey(vr):
    """
    Everything SteamVR knows about, and whether it is actually usable.

    Reported rather than assumed, because "registered" and "tracking" are different
    things: a controller that is switched off still appears in the device list with its
    model name, and polling it forever looks like the tool hanging.
    """
    import openvr

    names = {openvr.TrackedDeviceClass_HMD: "HMD",
             openvr.TrackedDeviceClass_Controller: "controller",
             openvr.TrackedDeviceClass_GenericTracker: "tracker",
             openvr.TrackedDeviceClass_TrackingReference: "base station"}

    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)

    rows = []

    for i in range(openvr.k_unMaxTrackedDeviceCount):
        kind = vr.getTrackedDeviceClass(i)

        if kind == openvr.TrackedDeviceClass_Invalid:
            continue

        rows.append({
            "index": i,
            "kind": names.get(kind, str(kind)),
            "model": vr.getStringTrackedDeviceProperty(i, openvr.Prop_ModelNumber_String),
            "connected": bool(vr.isTrackedDeviceConnected(i)),
            "tracking": bool(poses[i].bPoseIsValid),
        })

    return rows


def stage_fingerprint(vr):
    """
    Where the base stations are, in stage coordinates, keyed by serial.

    They are bolted to the room, so these numbers describe the ORIGIN rather than the
    hardware. Recorded with every calibration so that a later run can tell whether the
    origin has been re-established - which silently invalidates every stored cutout.
    """
    import openvr

    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)

    out = {}

    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if vr.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_TrackingReference:
            continue

        if not poses[i].bPoseIsValid:
            continue

        serial = vr.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
        m = poses[i].mDeviceToAbsoluteTracking
        out[serial] = [float(m[j][3]) for j in range(3)]

    return out


def warn_if_origin_moved(vr, saved):
    """Say so, loudly, if the stage origin is not the one a calibration was made against."""
    if not saved:
        return

    moved, worst, detail = compare_fingerprints(saved, stage_fingerprint(vr))

    if not moved:
        return

    print()
    print(f"  WARNING: the stage origin has MOVED since the tip was calibrated ({detail}).")
    print("  Base stations are bolted to the room, so it is the origin that changed, not")
    print("  them - a recentre or a re-run room setup. Any cutout written against the old")
    print("  origin is now wrong by that much, plus however much the yaw turned.")
    print("  What you measure now will be correct; what was stored before will not.")


def check_repeatability(vr, index, tip):
    """
    Touch ONE fixed point repeatedly, holding the controller differently each time.

    This separates two faults that look identical in a finished cutout. The tip offset
    lives in CONTROLLER space and is rotated by each pose, so turning your wrist should
    change nothing - if the points scatter as you rotate, the tip offset is wrong. If they
    scatter only when you switch which PART of the controller is touching, the offset is
    fine and the contact point is the problem, which matters on an Index where the ring
    and strap block the front in a tight cockpit.

    The distinction is worth the thirty seconds: one is fixed by re-pivoting, the other by
    reaching the corner differently, and guessing wrong costs a headset session.
    """
    import openvr

    print("\n  REPEATABILITY CHECK")
    print("  Pick one fixed point - a screw head, a corner - and touch it several times,")
    print("  TURNING YOUR WRIST as much as you can between touches. Same physical contact")
    print("  point on the controller every time.")
    print("\n  TRIGGER record   MENU finish   Ctrl+C quit\n")

    buttons = Buttons()
    points = []
    angles = []

    while True:
        pose, mask = read_controller(vr, index)

        if pose is None:
            time.sleep(0.02)
            continue

        if buttons.pressed(mask, openvr.k_EButton_SteamVR_Trigger):
            p = tip_position(pose, tip)
            points.append(p)
            angles.append(pose[:3, :3] @ np.array([0.0, 0.0, -1.0]))

            spread = (float(np.max(np.linalg.norm(np.array(points) - np.mean(points, axis=0),
                                                  axis=1))) * 1000.0) if len(points) > 1 else 0.0
            print(f"    touch {len(points):2d}   worst {spread:5.1f} mm from the mean")

        if buttons.pressed(mask, openvr.k_EButton_ApplicationMenu):
            break

        buttons.previous = mask
        time.sleep(0.005)

    if len(points) < 2:
        print("\n  Need at least two touches.")
        return 1

    pts = np.array(points)
    centre = pts.mean(axis=0)
    worst = float(np.max(np.linalg.norm(pts - centre, axis=1))) * 1000.0
    rms = float(np.sqrt(np.mean(np.sum((pts - centre) ** 2, axis=1)))) * 1000.0

    dirs = np.array(angles)
    turned = float(np.degrees(np.arccos(np.clip(
        np.min(dirs @ dirs.T), -1.0, 1.0))))

    print(f"\n  {len(points)} touches, wrist turned through {turned:.0f} deg")
    print(f"  scatter: {rms:.1f} mm rms, {worst:.1f} mm worst")

    if turned < 45.0:
        print("\n  Not enough wrist rotation to test anything. Turn further and run it")
        print("  again - held one way, even a badly wrong offset looks perfect.")
        return 1

    # A ROUNDED controller has no tip. If its contact region were a true sphere this would
    # scatter by zero, because a sphere resting on a point keeps its CENTRE fixed under
    # rotation and the pivot fit finds exactly that centre. It is not a sphere, so the
    # centre of curvature wanders - and no amount of re-pivoting changes the shape of the
    # shell. Telling the user to re-calibrate here would send them round a loop that
    # cannot converge.
    #
    # So this is judged against the alignment budget, not against zero.
    if worst < 5.0:
        print("\n  Excellent - the contact point barely moves as you rotate.")
        return 0

    if worst < 15.0:
        print(f"\n  Workable. {worst:.0f} mm is inside the ~20 mm alignment budget, and about")
        print("  what the camera path managed - while measuring the BUTTONS rather than")
        print("  just the screen.")
        print()
        print("  This is the floor for a rounded controller, not a calibration fault: with")
        print("  no defined tip, the contact point slides over the shell as you turn. Do")
        print("  not re-run --tip to chase it.")
        print()
        print("  To do better, keep the controller at a SIMILAR ANGLE for every corner of")
        print("  one panel. The error then lands the same way each time, so it shifts the")
        print("  cutout slightly instead of distorting it - and a shifted cutout is one")
        print("  slider to fix, where a distorted one is four.")
        return 0

    print(f"\n  {worst:.0f} mm is too much even for reaching buttons. Either the contact")
    print("  point changed between touches - the ring or the strap forcing a different")
    print("  face onto the corner - or the pivot was done on a flat surface that let the")
    print("  tip slide. Re-run --check touching deliberately with one face first.")

    return 1


def find_controllers(vr, require_tracking=True):
    """Usable controllers or trackers, most-usable first."""
    rows = [r for r in survey(vr) if r["kind"] in ("controller", "tracker")]

    if require_tracking:
        rows = [r for r in rows if r["tracking"]]

    return [r["index"] for r in rows]


def report_survey(vr):
    print("\n  SteamVR devices:")

    for r in survey(vr):
        state = ("tracking" if r["tracking"]
                 else ("connected, not tracking" if r["connected"] else "NOT CONNECTED"))
        print(f"    {r['index']:2d}  {r['kind']:13} {r['model']:16} {state}")


class Buttons:
    """
    Edge-triggered button state, so one press records one point.

    Polling level rather than edges would record a hundred points per squeeze, and the
    outline would come out as a cloud - which looks like the tracking being broken.
    """

    def __init__(self):
        self.previous = 0

    def pressed(self, mask, bit):
        was = bool(self.previous & (1 << bit))
        now = bool(mask & (1 << bit))
        return now and not was


def read_controller(vr, index):
    """
    (pose_4x4 or None, button mask).

    getControllerStateWithPose pairs the buttons with the pose AT THE MOMENT THE BUTTON
    CHANGED, which is what a "touch this corner" tool wants - the hand is already moving
    away by the time the press is polled. It needs SteamVR's legacy input, though, so
    there is a fall back to reading the two separately.
    """
    import openvr

    got = vr.getControllerStateWithPose(openvr.TrackingUniverseStanding, index)

    if got is not None:
        ok, state, pose = got

        if ok and pose.bPoseIsValid:
            return hmd_matrix_to_numpy(pose.mDeviceToAbsoluteTracking), state.ulButtonPressed

    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)

    if not poses[index].bPoseIsValid:
        return None, 0

    ok, state = vr.getControllerState(index)
    mask = state.ulButtonPressed if ok else 0

    return hmd_matrix_to_numpy(poses[index].mDeviceToAbsoluteTracking), mask


def calibrate_tip(vr, index):
    """
    Solve where the controller's tip is, by pivoting it about one fixed point.

    Not taken from the render model: the reported tip describes the controller the driver
    THINKS you have, and a wrong tip offset displaces every later measurement by the same
    amount - a failure that looks like perfect tracking of the wrong object, with nothing
    in the residuals to give it away.
    """
    import openvr

    print("\n  TIP CALIBRATION")
    print("  Rest the controller's tip in one spot that will not move - a screw head, a")
    print("  panel corner, a seam - and ROLL THE CONTROLLER AROUND IT in every direction")
    print("  you can while keeping the tip planted.")
    print("\n  Hold the TRIGGER to record. It stops on its own once the tip is pinned")
    print("  down. Ctrl+C in this terminal quits without saving.\n")

    buttons = Buttons()
    poses = []
    last_report = 0.0

    while True:
        pose, mask = read_controller(vr, index)

        if pose is not None and (mask & (1 << openvr.k_EButton_SteamVR_Trigger)):
            poses.append(pose)

        if len(poses) >= 3 and time.time() - last_report > 0.4:
            last_report = time.time()
            spread, verdict = pivot_conditioning(poses)
            tip, _, residual = fit_pivot(poses)
            sigma = pivot_uncertainty(poses, residual)
            worst = np.max(sigma) * 1000 if np.all(np.isfinite(sigma)) else float("inf")

            print("\r  %4d samples  spread %5.1f deg %-9s wobble %5.2f mm  tip +/- %6.2f mm   "
                  % (len(poses), spread, verdict, residual * 1000, worst), end="", flush=True)

            if worst < 1.5 and residual < 0.004 and len(poses) > 120:
                print("\n\n  Enough. Release the trigger.")
                break

        if not (mask & (1 << openvr.k_EButton_SteamVR_Trigger)) and len(poses) > 120:
            break

        time.sleep(0.01)
        buttons.previous = mask

    tip, _, residual = fit_pivot(poses)
    spread, verdict = pivot_conditioning(poses)

    if tip is None:
        print("\n  Not enough samples.")
        return None

    print(f"\n  tip offset {np.round(tip, 4)} m from the controller origin")
    print(f"  wobble {residual * 1000:.2f} mm, rotation spread {spread:.1f} deg {verdict}")

    if verdict == "TOO FLAT":
        print("\n  REJECTED: the controller was only turned about one axis. That fits the")
        print("  data perfectly and still leaves the offset free along that axis - every")
        print("  point you measure afterwards would be wrong by the same amount, with")
        print("  nothing to show for it. Roll it in more directions and try again.")
        return None

    if residual > 0.004:
        print("\n  REJECTED: the tip moved more than 4 mm during the pivot. Plant it in")
        print("  something that locates it - a corner or a recess, not a flat surface.")
        return None

    os.makedirs(os.path.dirname(TIP_FILE), exist_ok=True)

    with open(TIP_FILE, "w") as f:
        json.dump({"tip_offset_m": [float(v) for v in tip],
                   "wobble_mm": round(residual * 1000, 3),
                   "sigma_mm": [round(float(v) * 1000, 3) for v in sigma],
                   "spread_deg": round(spread, 1),
                   "samples": len(poses),
                   "stage_reference": stage_fingerprint(vr),
                   # Kept so the calibration can be re-analysed - or re-solved under a
                   # better method - without asking for the pivot to be done again.
                   "poses": [[float(v) for v in m.reshape(16)] for m in poses]},
                  f, indent=2)
        f.write("\n")

    print(f"\n  Saved to {TIP_FILE}")
    return tip


def load_stage_reference():
    if not os.path.exists(TIP_FILE):
        return None

    with open(TIP_FILE) as f:
        return json.load(f).get("stage_reference")


def load_tip():
    if not os.path.exists(TIP_FILE):
        return None

    with open(TIP_FILE) as f:
        return np.array(json.load(f)["tip_offset_m"], float)


def measure_cutouts(vr, index, tip, start_index):
    """Walk the user through touching one cutout after another."""
    import openvr

    print("\n  MEASURING")
    print("  Touch the corners of what you want to see through. Four corners for a panel,")
    print("  or walk the edge for an irregular console - the order you touch IS the shape.")
    print("\n  TRIGGER record   GRIP undo   MENU finish this cutout")
    print("  MENU with nothing pending finishes and writes.")
    print("  Ctrl+C in this terminal quits without writing anything.\n")

    buttons = Buttons()
    cutouts = []
    points = []
    idle_menu = 0.0

    while True:
        pose, mask = read_controller(vr, index)

        if pose is None:
            time.sleep(0.02)
            continue

        if buttons.pressed(mask, openvr.k_EButton_SteamVR_Trigger):
            p = tip_position(pose, tip)
            points.append(p)
            print(f"    point {len(points):2d}  {p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}")

        if buttons.pressed(mask, openvr.k_EButton_Grip) and points:
            points.pop()
            print(f"    undo -> {len(points)} point(s)")

        if buttons.pressed(mask, openvr.k_EButton_ApplicationMenu):
            if not points:
                print("\n  Done.")
                break

            head = _head_position(vr)
            name = f"cutout{start_index + len(cutouts)}"
            got = outline_from_touches(name, points, head)

            if got is None:
                print("    need at least three points")
            else:
                cutouts.append(got)
                print(f"\n  {name}: {got.width * 1000:.0f} x {got.height * 1000:.0f} mm, "
                      f"{len(points)} points, {got.flatness_mm:.1f} mm out of plane")

                if got.flatness_mm > 8.0:
                    print("    ^ those touches are not flat. Did one slip off the bezel?")

                print("    next cutout, or MENU again to finish\n")
                points = []

        buttons.previous = mask
        time.sleep(0.005)

    return cutouts


def _head_position(vr):
    import openvr

    poses = vr.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
    hmd = poses[openvr.k_unTrackedDeviceIndex_Hmd]

    if hmd.bPoseIsValid:
        return hmd_matrix_to_numpy(hmd.mDeviceToAbsoluteTracking)[:3, 3]

    return np.array([0.0, 1.2, 0.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tip", action="store_true", help="calibrate the controller tip")
    ap.add_argument("--check", action="store_true",
                    help="touch ONE fixed point several times, holding the controller "
                         "differently each time, and see how far apart they land. That "
                         "isolates the tip calibration from everything else.")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import openvr

    if menu_is_running() and not a.force:
        print(f"  {MENU_PROCESS} is running and would overwrite whatever is written.")
        print("  Close it first. --force overrides, but the write will probably be lost.")
        return 1

    vr = openvr.init(openvr.VRApplication_Background)

    try:
        controllers = find_controllers(vr)

        if not controllers:
            report_survey(vr)

            idle = find_controllers(vr, require_tracking=False)

            print()
            if idle:
                print("  A controller is registered but not tracking. Wake it up (press")
                print("  the system button), and check the base stations are on - a device")
                print("  cannot report a pose without one in view.")
            else:
                print("  No controller or tracker found at all. Turn one on and try again.")

            return 1

        index = controllers[0]
        print(f"  controller {index}" + (f" (+{len(controllers) - 1} more, using the first)"
                                         if len(controllers) > 1 else ""))

        if a.check:
            tip = load_tip()

            if tip is None:
                print("\n  Calibrate the tip first:  --tip")
                return 1

            try:
                return check_repeatability(vr, index, tip)
            except KeyboardInterrupt:
                print("\n\n  Stopped.")
                return 1

        if a.tip:
            try:
                return 0 if calibrate_tip(vr, index) is not None else 1
            except KeyboardInterrupt:
                print("\n\n  Stopped. Nothing was saved.")
                return 1

        tip = load_tip()

        if tip is None:
            print("\n  No tip calibration yet. Run this first:")
            print("    python scripts/touch_cutouts.py --tip")
            return 1

        print(f"  tip offset {np.round(tip, 4)} m")
        warn_if_origin_moved(vr, load_stage_reference())

        try:
            cutouts = measure_cutouts(vr, index, tip, a.start)
        except KeyboardInterrupt:
            # Deliberately discards. Half-measured cutouts written to the config would be
            # worse than none: a stale enabled quad looks exactly like a bad measurement.
            print("\n\n  Stopped. Nothing was written.")
            return 1
    finally:
        openvr.shutdown()

    if not cutouts:
        print("  Nothing measured.")
        return 0

    if a.start + len(cutouts) > MAX_QUADS:
        print(f"\n  {len(cutouts)} cutouts from index {a.start} exceeds the layer's "
              f"{MAX_QUADS}-quad limit.")
        return 1

    if not os.path.exists(a.config):
        print(f"\n  No config at {a.config}. Run the layer once so it writes its defaults.")
        return 1

    shutil.copy2(a.config, a.config + ".bak")
    existing = read_quads(a.config)

    for n, c in enumerate(cutouts):
        i = a.start + n
        write_quad(QuadConfig(i, enabled=True, name=c.name, position=c.position,
                              euler_deg=c.euler_deg, width=c.width, height=c.height,
                              points=c.points), a.config)
        print(f"  Quad{i} <- {c.name}"
              + (f"  (replaced {existing[i].label})" if existing[i].enabled else ""))

    print(f"\n  Written. Previous config kept at {a.config}.bak")
    print("  Restart the game, or reload the config from the passthrough menu.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
