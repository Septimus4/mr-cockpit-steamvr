"""
Trace a cutout outline by clicking around a panel in a camera frame.

The maths lives in tracing/ and is unit tested; this file is only the window, the mouse,
and the file paths. Keeping it thin is deliberate - everything that could be wrong in a
way you would not notice is tested elsewhere.

  Make a synthetic capture to try the workflow with no hardware:
    python scripts/trace_cutout.py demo

  Grab a real frame (needs the camera, SteamVR, and the headset tracking):
    python scripts/trace_cutout.py capture --out captures/mip.npz

  Trace it onto cutout 0 and write the outline into the layer's config:
    python scripts/trace_cutout.py trace --capture captures/mip.npz --quad 0

KEYS   left click   add a point
       right click / backspace   undo the last point
       c   clear      s   simplify      Enter   save      ESC   cancel
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracing.capture import Capture, synthetic_capture
from tracing.config_io import DEFAULT_CONFIG_PATH, MAX_POINTS, read_quads, write_points
from tracing.geometry import (
    Camera, Plane, backproject_click, camera_to_world_from_hmd, is_simple_polygon,
    point_to_outline_distance, polygon_signed_area, pose_to_matrix, simplify_outline,
)

K_LEFT = np.array([[1072.26851867, 0.0, 788.49299729],
                   [0.0, 1072.31519651, 614.54444602],
                   [0.0, 0.0, 1.0]])
D_LEFT = np.array([0.08313216691950971, -0.10744697901181298,
                   -0.00016821021003468, 0.00025331486744491, 0.0])
CAMERA_OFFSET = (-0.031, -0.047, -0.138)


def cmd_demo(args):
    """A synthetic capture, so the tool can be exercised without a cockpit."""
    hmd = pose_to_matrix((0.0, 1.15, 0.0), (-18.0, 0.0, 0.0))
    cam = Camera(K_LEFT, D_LEFT, camera_to_world_from_hmd(hmd, CAMERA_OFFSET))
    plane = Plane((0.0, 1.02, -0.62), (-25.0, 0.0, 0.0))

    # An L-shaped panel, so the demo exercises a concave outline rather than a rectangle.
    outline = [(-0.10, -0.07), (0.10, -0.07), (0.10, 0.01),
               (0.02, 0.01), (0.02, 0.07), (-0.10, 0.07)]

    capture, pixels = synthetic_capture(cam, plane, outline)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    capture.save(args.out)

    print(f"  wrote {args.out}")
    print(f"  synthetic panel: {len(outline)} points on a plane at "
          f"{plane.position} rot {plane.euler_deg}")
    print(f"  trace it with:  python scripts/trace_cutout.py trace --capture {args.out} "
          f"--quad 0 --pose-from-demo")
    return 0


def cmd_capture(args):
    from tracing.capture import grab_capture

    capture = grab_capture(K=K_LEFT, dist=D_LEFT, camera_offset=CAMERA_OFFSET)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    capture.save(args.out)

    print(f"  wrote {args.out}  ({capture.size[0]}x{capture.size[1]})")
    print(f"  camera at {np.round(capture.camera().position, 4)}")
    return 0


def cmd_trace(args):
    import cv2

    capture = Capture.load(args.capture)
    cam = capture.camera()

    if args.pose_from_demo:
        plane = Plane((0.0, 1.02, -0.62), (-25.0, 0.0, 0.0))
        quad_label = "demo plane"
    else:
        quads = read_quads(args.config)
        quad = quads[args.quad]
        plane = Plane(quad.position, quad.euler_deg)
        quad_label = quad.label

        if not quad.enabled:
            print(f"  note: {quad_label} is not enabled in the config")

    print(f"  tracing onto {quad_label}")
    print(f"  plane at {np.round(plane.position, 3)} rot {plane.euler_deg}")
    print(f"  camera at {np.round(cam.position, 3)}")

    img = capture.image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    scale = min(1.0, 1400.0 / img.shape[1])
    state = {"clicks": [], "quit": False, "save": False}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["clicks"].append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and state["clicks"]:
            state["clicks"].pop()

    win = "trace cutout - Enter to save, ESC to cancel"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    while not state["quit"]:
        view = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img.copy()

        pts = state["clicks"]
        uv, dropped = [], 0

        for px, py in pts:
            r = backproject_click(cam, plane, px, py)
            if r is None:
                dropped += 1
            else:
                uv.append(r)

        for i, (px, py) in enumerate(pts):
            p = (int(px * scale), int(py * scale))
            cv2.circle(view, p, 4, (0, 220, 255), -1)
            if i > 0:
                q = (int(pts[i - 1][0] * scale), int(pts[i - 1][1] * scale))
                cv2.line(view, q, p, (0, 220, 255), 1)

        if len(pts) >= 3:
            first = (int(pts[0][0] * scale), int(pts[0][1] * scale))
            last = (int(pts[-1][0] * scale), int(pts[-1][1] * scale))
            cv2.line(view, last, first, (0, 140, 200), 1)

        simple = is_simple_polygon(uv) if len(uv) >= 3 else False
        lines = [f"points {len(pts)}/{MAX_POINTS}" + (f"  DROPPED {dropped}" if dropped else "")]

        if len(uv) >= 3:
            xs = [p[0] for p in uv]
            ys = [p[1] for p in uv]
            lines.append(f"size {(max(xs)-min(xs))*1000:.0f} x {(max(ys)-min(ys))*1000:.0f} mm")
            lines.append("outline OK" if simple else "SELF-INTERSECTING - fix before saving")
        else:
            lines.append("need at least 3 points")

        lines.append("L add  R undo  c clear  s simplify  Enter save  ESC cancel")

        for i, t in enumerate(lines):
            colour = (0, 0, 255) if "SELF-INTERSECT" in t or "DROPPED" in t else (0, 255, 0)
            cv2.putText(view, t, (10, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            state["quit"] = True
        elif key in (13, 10):
            if len(uv) < 3:
                print("  need at least 3 points")
            elif not simple:
                print("  outline self-intersects - the layer would fall back to a rectangle")
            else:
                state["save"] = True
                state["quit"] = True
        elif key == 8 and pts:
            pts.pop()
        elif key == ord("c"):
            state["clicks"] = []
        elif key == ord("s") and len(pts) > 3:
            before = len(state["clicks"])
            keep = simplify_outline(uv, tolerance_m=args.tolerance)
            # Map the kept plane points back to the clicks that produced them.
            kept_clicks = []
            for k in keep:
                j = int(np.argmin([np.hypot(p[0] - k[0], p[1] - k[1]) for p in uv]))
                kept_clicks.append(pts[j])
            state["clicks"] = kept_clicks

            # Report how far simplification actually moved the outline, so the user can
            # see it is far inside the alignment budget rather than take it on trust.
            worst = max(point_to_outline_distance(p, keep) for p in uv)
            print(f"  simplified {before} -> {len(kept_clicks)} points, "
                  f"outline moved at most {worst * 1000:.2f} mm")

    cv2.destroyAllWindows()

    if not state["save"]:
        print("  cancelled, nothing written")
        return 1

    final, _ = [], 0
    final = [backproject_click(cam, plane, px, py) for px, py in state["clicks"]]
    final = [p for p in final if p is not None]

    if len(final) > MAX_POINTS:
        final = simplify_outline(final, tolerance_m=args.tolerance)
        print(f"  simplified to {len(final)} points to fit the {MAX_POINTS}-point limit")

    if args.pose_from_demo:
        print(f"  demo mode - not writing to the config. Outline would be:")
        print("   ", ";".join(f"{x:.5f},{y:.5f}" for x, y in final))
        return 0

    value = write_points(args.quad, final, args.config)
    print(f"  wrote Quad{args.quad}_Points to {args.config or DEFAULT_CONFIG_PATH}")
    print(f"  {len(final)} points, area {abs(polygon_signed_area(final))*1e4:.1f} cm2")
    print("  restart the app, or toggle a setting in the menu, to make the layer re-read it")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="write a synthetic capture, no hardware needed")
    d.add_argument("--out", default="captures/demo.npz")
    d.set_defaults(func=cmd_demo)

    c = sub.add_parser("capture", help="grab a live frame and the headset pose")
    c.add_argument("--out", default="captures/capture.npz")
    c.set_defaults(func=cmd_capture)

    t = sub.add_parser("trace", help="click an outline onto a cutout's plane")
    t.add_argument("--capture", required=True)
    t.add_argument("--quad", type=int, default=0)
    t.add_argument("--config", default=None, help=f"default: {DEFAULT_CONFIG_PATH}")
    t.add_argument("--tolerance", type=float, default=0.002,
                   help="simplification tolerance in metres (default 2 mm)")
    t.add_argument("--pose-from-demo", action="store_true",
                   help="use the demo plane instead of reading a cutout's pose")
    t.set_defaults(func=cmd_trace)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
