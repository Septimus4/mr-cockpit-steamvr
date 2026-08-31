"""
Keep your cutouts on their panels when the stage frame moves.

Cutouts live in stage coordinates, and that frame is not permanent. A SteamVR recentre or a
re-run room setup re-establishes it with a slide AND a yaw - measured here at 0.73 m and
60 degrees - and every cutout is then wrong by that much, with nothing in the config to say
so. It reads as the measurement having been bad.

Markers are stuck to the panels, so wherever the frame goes they go with it.

  BIND once, right after measuring your cutouts:
    python scripts/solve_anchors.py            sweep, so the markers are solved
    python scripts/reanchor.py bind            remember where they were

  RESTORE whenever the cutouts have stopped landing:
    python scripts/solve_anchors.py            sweep again
    python scripts/reanchor.py restore         carry the cutouts across

Nothing is written by `restore` without --write.

The division of labour is deliberate. The controller measures the SHAPE exactly, including
buttons no marker can see. The markers notice the frame has moved and put the shape back.
Neither does the other's job.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchors.camera_rig import apply_offset_delta
from anchors.detect import plate_size_overrides, refresh_observations
from anchors.rebind import (
    binding_is_plausible, frame_shift, move_pose, reference_from, shift_summary,
)
from anchors.solver import solve_markers
from tracing.config_io import (
    DEFAULT_CONFIG_PATH, MAX_QUADS, QuadConfig, read_quads, write_quad,
)

ANCHOR_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "config-backups", "anchor-binding.json")

MENU_PROCESS = "passthrough-menu.exe"


def menu_is_running():
    import subprocess

    try:
        out = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {MENU_PROCESS}"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None

    return MENU_PROCESS.lower() in out.stdout.lower() if out.returncode == 0 else None


def solve(capture):
    """Solve a capture under the CURRENT calibration."""
    if not os.path.exists(capture):
        print(f"  No capture at {capture}")
        print("  Sweep the cockpit first:  python scripts/solve_anchors.py")
        return None

    z = np.load(capture, allow_pickle=True)
    cameras = {int(k): v for k, v in zip(z["frames"], z["cameras"])}
    cameras, _ = apply_offset_delta(
        cameras, z["camera_offset"] if "camera_offset" in z.files else None)

    observations, _, _ = refresh_observations(
        list(z["observations"]), plate_size_overrides(),
        float(z["corner_bias"]) if "corner_bias" in z.files else 0.0)

    return solve_markers(observations, cameras)


def do_bind(a):
    solutions = solve(a.capture)

    if solutions is None:
        return 1

    if len(solutions) < 3:
        print(f"  Only {len(solutions)} marker(s) solved. Three is the minimum for an")
        print("  orientation, and more spread out is better.")
        return 1

    quads = [q for q in read_quads(a.config) if q.enabled]

    if not quads:
        print("  No cutouts are enabled, so there is nothing to anchor.")
        print("  Measure some first:  python scripts/touch_cutouts.py")
        return 1

    reference = reference_from(solutions)
    plausible, worst, per_cutout = binding_is_plausible(
        list(reference.values()), [q.position for q in quads])

    if not plausible and not a.force:
        print("  REFUSING: the markers and the cutouts were not measured in the same")
        print(f"  stage frame. Every cutout should sit within a panel-width of its markers;")
        print(f"  the worst here is {worst:.0f} mm away.\n")

        for q, d in zip(quads, per_cutout):
            print(f"    {q.label:14} nearest marker {d:6.0f} mm")

        print("\n  Binding these would record a frame change that never happened, and then")
        print("  'restore' would apply it. Sweep the markers again NOW, in the same frame")
        print("  the cutouts were touched in, and bind from that capture.")
        return 1

    binding = {
        "markers": reference,
        "cutouts": [{"index": q.index, "name": q.name,
                     "position": list(q.position), "euler_deg": list(q.euler_deg)}
                    for q in quads],
    }

    os.makedirs(os.path.dirname(ANCHOR_FILE), exist_ok=True)

    with open(ANCHOR_FILE, "w") as f:
        json.dump(binding, f, indent=2)
        f.write("\n")

    print(f"  Bound {len(quads)} cutout(s) to {len(solutions)} marker(s).")
    for q in quads:
        print(f"    Quad{q.index}  {q.label}")

    print(f"\n  Saved to {ANCHOR_FILE}")
    print("\n  If the cutouts stop landing on their panels, sweep again and run:")
    print("    python scripts/reanchor.py restore")

    return 0


def do_restore(a):
    if not os.path.exists(ANCHOR_FILE):
        print(f"  Nothing bound yet ({ANCHOR_FILE} does not exist).")
        print("  Bind once while the cutouts are still correct:")
        print("    python scripts/reanchor.py bind")
        return 1

    with open(ANCHOR_FILE) as f:
        binding = json.load(f)

    solutions = solve(a.capture)

    if solutions is None:
        return 1

    reference = {int(k): v for k, v in binding["markers"].items()}
    current = {int(i): s.position for i, s in solutions.items()}

    shift = frame_shift(reference, current)

    if shift is None:
        shared = set(reference) & set(current)
        print(f"  Only {len(shared)} marker(s) in common with the binding, and three are")
        print("  needed. Sweep so that more of the same markers are seen.")
        return 1

    rotation, translation, rms, ids = shift
    angle, yaw, distance = shift_summary(rotation, translation)

    print(f"  {len(ids)} marker(s) in common: {ids}")
    print(f"  frame moved {distance * 1000:.0f} mm and turned {angle:.1f} deg "
          f"({yaw:+.1f} deg of yaw)")
    print(f"  markers agree to {rms * 1000:.1f} mm after the fit")

    if rms > 0.010:
        print("\n  That residual is too high for a rigid frame change. The markers no")
        print("  longer sit where they did relative to each other, so one has been moved,")
        print("  or a panel has. Re-measure rather than carrying the old shapes across.")
        return 1

    if distance < 0.002 and angle < 0.2:
        print("\n  The frame has not moved. Nothing to do - if the cutouts are not landing,")
        print("  the cause is something else.")
        return 0

    print()
    moved = []

    for entry in binding["cutouts"]:
        position, euler = move_pose(entry["position"], entry["euler_deg"],
                                    rotation, translation)
        moved.append((entry, position, euler))

        print(f"  Quad{entry['index']:<2} {entry['name']:14} "
              f"({entry['position'][0]:+.3f},{entry['position'][1]:+.3f},"
              f"{entry['position'][2]:+.3f})  ->  "
              f"({position[0]:+.3f},{position[1]:+.3f},{position[2]:+.3f})")

    if not a.write:
        print(f"\n  Dry run. Add --write to put these into {a.config}")
        return 0

    if menu_is_running() and not a.force:
        print(f"\n  REFUSING: {MENU_PROCESS} is running and would write its own copy back")
        print("  over this. Close it, run again, then reopen it.")
        return 1

    shutil.copy2(a.config, a.config + ".bak")
    existing = read_quads(a.config)

    for entry, position, euler in moved:
        i = entry["index"]
        q = existing[i]

        # Size and outline are untouched: the panel did not change shape, only the frame
        # it is described in. Rewriting them from the binding would quietly undo any
        # adjustment made since.
        write_quad(QuadConfig(i, enabled=True, name=entry["name"], position=position,
                              euler_deg=euler, width=q.width, height=q.height,
                              points=q.points), a.config)

    # The binding now describes the new frame, so restoring twice is not a double move.
    binding["markers"] = reference_from(solutions)
    for entry, position, euler in moved:
        entry["position"] = [float(v) for v in position]
        entry["euler_deg"] = [float(v) for v in euler]

    with open(ANCHOR_FILE, "w") as f:
        json.dump(binding, f, indent=2)
        f.write("\n")

    print(f"\n  Written. Previous config kept at {a.config}.bak")
    print("  Restart the game, or reload the config from the passthrough menu.")

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["bind", "restore"])
    ap.add_argument("--capture", default="captures/anchors.npz")
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="bind even if the markers and cutouts look like different "
                         "frames, or write while the menu is running. Both are refusals "
                         "you should heed rather than override.")
    a = ap.parse_args()

    return do_bind(a) if a.action == "bind" else do_restore(a)


if __name__ == "__main__":
    sys.exit(main())
