"""
Solve where your markers are, from the camera.

LOOK, THEN HOLD STILL. It captures only while your head is nearly stationary, moves on,
and solves when you stop.

The holding still is not politeness, it is the measurement. A USB camera frame arrives
buffered by tens of milliseconds, and the headset pose is read now - so while the head is
moving, the image and the pose describe different instants. Measured on a continuous
sweep: 34 mm of per-view disagreement, which no camera calibration could fix. Neither
adjusting the camera rotation nor moving its offset by 92 mm improved it, while simply
filtering to slow frames halved it.

The maths lives in anchors/ and is covered by 104 tests; this file is only the camera,
the headset pose and the printing.

  python scripts/solve_anchors.py                 capture and solve
  python scripts/solve_anchors.py --seconds 20    sweep for longer
  python scripts/solve_anchors.py --dry-run       replay the last capture, no hardware

KEYS  q or ESC   stop sweeping and solve
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anchors.detect import detect_markers, make_detector, plate_size_overrides
from anchors.solver import constellation_conditioning, is_coplanar, solve_markers
from tracing.capture import hmd_matrix_to_numpy
from tracing.geometry import Camera, camera_to_world_from_hmd

K_LEFT = np.array([[1072.26851867, 0.0, 788.49299729],
                   [0.0, 1072.31519651, 614.54444602],
                   [0.0, 0.0, 1.0]])
D_LEFT = np.array([0.08313216691950971, -0.10744697901181298,
                   -0.00016821021003468, 0.00025331486744491, 0.0])
CAMERA_OFFSET = (-0.031, -0.047, -0.138)


def report(solutions):
    if not solutions:
        print("\n  No markers solved.")
        print("  Are they in view, and is the lighting enough for the camera?")
        return 1

    print(f"\n  {len(solutions)} marker(s) solved\n")
    print(f"  {'id':>4} {'obs':>4} {'x':>8} {'y':>8} {'z':>8}"
          f" {'spread mm':>10} {'ang deg':>8} {'reproj px':>10}")

    for marker_id in sorted(solutions):
        s = solutions[marker_id]
        p = s.position
        print(f"  {marker_id:4d} {s.observations:4d} {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f}"
              f" {s.position_spread_mm:10.2f} {s.angle_spread_deg:8.2f} {s.reprojection_px:10.2f}")

    ordered = [solutions[i] for i in sorted(solutions)]
    aspect, verdict, extent = constellation_conditioning(ordered)
    coplanar, out_of_plane = is_coplanar(ordered)

    print(f"\n  constellation extent {extent:.0f} mm, aspect {aspect:.2f}:1  {verdict}")

    if verdict in ("TOO COLLINEAR", "COLLINEAR"):
        print("  Markers are too close to a line. That AMPLIFIES systematic error rather")
        print("  than averaging it down, and accuracy then depends on where they land in")
        print("  the lens. Move one well off the line between the others.")
    elif verdict == "MARGINAL":
        print("  Workable, but spreading them further apart would measurably help.")

    print(f"  {'coplanar' if coplanar else 'non-coplanar'}, "
          f"{out_of_plane:.0f} mm out of plane")

    if coplanar:
        print("  Coplanar sets cannot resolve the planar tilt ambiguity. Markers at")
        print("  differing depths - coaming as well as panel - would fix that.")

    worst = max(s.position_spread_mm for s in ordered)
    if worst > 5.0:
        print(f"\n  Largest per-view disagreement is {worst:.1f} mm. Above about 5 mm,")
        print("  suspect a marker on something that flexes, or one printed at the wrong scale.")

    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--still-deg", type=float, default=0.35,
                    help="capture only when the head moves less than this between polls "
                         "(default 0.35 deg)")
    ap.add_argument("--rate", type=float, default=5.0,
                    help="observations per second (default 5; detection is 5-20 ms and "
                         "there is no reason to run it faster)")
    ap.add_argument("--out", default="captures/anchors.npz")
    ap.add_argument("--dry-run", action="store_true", help="replay --out, no hardware")
    a = ap.parse_args()

    if a.dry_run:
        z = np.load(a.out, allow_pickle=True)
        obs = list(z["observations"])
        cams = {int(k): v for k, v in zip(z["frames"], z["cameras"])}
        print(f"  replaying {len(obs)} observation(s) from {a.out}")
        return report(solve_markers(obs, cams))

    import cv2
    import openvr
    from cam import open_elp

    detector = make_detector()

    # Display panels render whatever marker size fits them, which the id cannot describe.
    overrides = plate_size_overrides()
    if overrides:
        print("  display plate sizes: " +
              ", ".join(f"{i}={overrides[i]:.1f}mm" for i in sorted(overrides)))

    vr = openvr.init(openvr.VRApplication_Background)
    cap = open_elp()

    observations = []
    cameras = {}
    frame = 0
    seen = set()

    print(f"\n  Sweep your head slowly across the cockpit for {a.seconds:.0f}s.")
    print("  q or ESC to stop early.\n")

    start = time.time()
    interval = 1.0 / max(a.rate, 0.5)
    nxt = 0.0
    last_pose = None
    still = False

    try:
        while time.time() - start < a.seconds:
            ok, img = cap.read()
            if not ok or img is None:
                continue

            now = time.time() - start
            left = img[:, :img.shape[1] // 2]

            if now >= nxt:
                nxt = now + interval

                poses = vr.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
                hmd = poses[openvr.k_unTrackedDeviceIndex_Hmd]

                still = False

                if hmd.bPoseIsValid:
                    pose_now = hmd_matrix_to_numpy(hmd.mDeviceToAbsoluteTracking)

                    if last_pose is not None:
                        r = last_pose[:3, :3].T @ pose_now[:3, :3]
                        moved = np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))
                        moved += np.linalg.norm(pose_now[:3, 3] - last_pose[:3, 3]) * 200.0
                        still = moved < a.still_deg

                    last_pose = pose_now

                if hmd.bPoseIsValid and still:
                    cam = Camera(K_LEFT, D_LEFT,
                                 camera_to_world_from_hmd(
                                     hmd_matrix_to_numpy(hmd.mDeviceToAbsoluteTracking),
                                     CAMERA_OFFSET),
                                 image_size=(left.shape[1], left.shape[0]))

                    got, rejected = detect_markers(left, cam, frame=frame, detector=detector,
                                                   size_overrides=overrides)

                    if got:
                        cameras[frame] = cam
                        observations.extend(got)
                        seen.update(o.marker_id for o in got)
                        frame += 1

                    for marker_id, why in rejected.items():
                        if marker_id not in seen:
                            print(f"  SKIPPED marker {marker_id}: {why}")
                            seen.add(marker_id)

            view = cv2.resize(left, (800, 600))
            colour = (0, 255, 0) if still else (0, 165, 255)
            cv2.putText(view, "CAPTURING - hold still" if still else "move, then HOLD STILL",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
            cv2.putText(view, f"{a.seconds - now:4.1f}s  {len(observations)} obs  "
                              f"ids {sorted(i for i in seen)}",
                        (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("sweep the cockpit - q to finish", view)

            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        openvr.shutdown()

    if not observations:
        print("\n  No markers seen at all.")
        return 1

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out,
                        observations=np.array(observations, dtype=object),
                        frames=np.array(sorted(cameras)),
                        cameras=np.array([cameras[k] for k in sorted(cameras)], dtype=object))
    print(f"\n  {len(observations)} observations over {len(cameras)} frames -> {a.out}")

    return report(solve_markers(observations, cameras))


if __name__ == "__main__":
    sys.exit(main())
