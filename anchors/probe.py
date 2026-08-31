"""
Measuring cockpit geometry by touching it with a tracked controller.

This is the shortcut past most of the calibration chain. A controller tip is already in
STAGE coordinates - the same space cutouts live in - so touching the corner of a panel
gives its position directly. No camera, no marker, no PnP, no lens model, no detector
bias, no range scale. Just a point, to Lighthouse accuracy.

It also reaches what markers never could. Markers can only measure the SCREEN, because the
screen is what draws them; the bezel, the buttons, the UFC stack and the centre console
are invisible to them and have had to be typed in from a datasheet. A controller can touch
any of it.

Markers keep the job they are actually good at: re-anchoring at runtime, so a cutout
measured once stays put.

The one number that has to be right is the tip offset. Get it wrong and every point is
displaced by the same amount, which looks like perfect tracking of the wrong thing - so it
is solved here rather than taken from a render model that may not describe the controller
in hand.
"""

import numpy as np

from tracing.geometry import matrix_to_euler_xyz


def fit_pivot(poses):
    """
    Where the tip is, in the controller's own frame, from a pivot calibration.

    Rest the tip in one fixed spot - a seam, a screw head, the corner of a panel - and
    rotate the controller around it. Every pose then satisfies

        R_i @ tip + p_i = centre

    for one unknown `tip` (in controller space) and one unknown `centre` (in world space).
    That is linear in both, so it is a least-squares solve rather than an optimisation:

        [R_i  -I] @ [tip; centre] = -p_i

    Returns (tip_offset, centre, rms_residual_metres). The residual is the honest quality
    signal - it is how far the tip moved during the pivot, so a wobbly calibration reports
    itself instead of silently displacing every later measurement.

    Rotating about only one axis leaves the solution under-determined; the residual will
    look fine and the offset will be wrong, so `pivot_conditioning` exists to catch it.
    """
    poses = [np.asarray(m, float).reshape(4, 4) for m in poses]

    if len(poses) < 3:
        return None, None, float("inf")

    a = np.zeros((3 * len(poses), 6))
    b = np.zeros(3 * len(poses))

    for i, m in enumerate(poses):
        a[3 * i:3 * i + 3, :3] = m[:3, :3]
        a[3 * i:3 * i + 3, 3:] = -np.eye(3)
        b[3 * i:3 * i + 3] = -m[:3, 3]

    solution, *_ = np.linalg.lstsq(a, b, rcond=None)

    tip = solution[:3]
    centre = solution[3:]
    residual = float(np.sqrt(np.mean((a @ solution - b) ** 2)))

    return tip, centre, residual


def pivot_conditioning(poses):
    """
    Whether a pivot calibration was rotated enough to determine the tip offset.

    Returns (spread_deg, verdict). Turning the controller about a single axis fits the
    data perfectly and still leaves the offset free along that axis - the residual looks
    excellent and every later point is wrong by a constant. Measured as the angular spread
    of the controller's forward axis, which is what a user is actually varying when they
    "roll it around".
    """
    poses = [np.asarray(m, float).reshape(4, 4) for m in poses]

    if len(poses) < 3:
        return 0.0, "TOO FEW"

    # Where the controller pointed on each sample. A good pivot sweeps this over a cone.
    directions = np.array([m[:3, :3] @ np.array([0.0, 0.0, -1.0]) for m in poses])
    centred = directions - directions.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)

    spread = float(np.degrees(np.arcsin(np.clip(sv[1] / max(len(poses) ** 0.5, 1.0), 0, 1))))
    verdict = "GOOD" if spread > 12.0 else ("MARGINAL" if spread > 6.0 else "TOO FLAT")

    return spread, verdict


def tip_position(pose, tip_offset):
    """The tip's world position for one controller pose."""
    m = np.asarray(pose, float).reshape(4, 4)

    return m[:3, :3] @ np.asarray(tip_offset, float) + m[:3, 3]


def level_frame(r):
    """
    Rotate a plane frame about its own normal until Y points as close to world up as it can.

    A best-fit plane's in-plane axes come from SVD, which picks the LONGEST spread as X.
    For a 167 x 185 mm panel that is the tall axis, so Width and Height come out swapped
    and the stored rotation is 90 degrees from what anyone would write by hand. Harmless
    to the maths, actively confusing to a person reading the config or reaching for the
    Height slider.

    A panel lying flat has no meaningful "up" in its own plane; there the original axes
    are kept rather than snapping to an arbitrary choice.
    """
    normal = r[:, 2]
    world_up = np.array([0.0, 1.0, 0.0])

    y = world_up - normal * float(normal @ world_up)
    length = float(np.linalg.norm(y))

    if length < 1e-6:                        # the plane faces straight up or down
        return r

    y /= length
    x = np.cross(y, normal)
    x /= np.linalg.norm(x)

    return np.column_stack([x, y, normal])


def plane_from_touches(points, viewpoint):
    """
    A cutout frame fitted to touched points: (origin, rotation, worst_out_of_plane_m).

    Same plane fit the marker path uses, so a cutout measured by hand and one measured by
    camera are the same kind of object downstream. Oriented towards the viewer, because a
    best-fit plane has no notion of which side is the front, then levelled so its Y is up.
    """
    from .place import fit_plane_frame, orient_frame_towards

    pts = np.asarray(points, float).reshape(-1, 3)
    origin, r, _ = fit_plane_frame(pts)
    r = level_frame(orient_frame_towards(origin, r, viewpoint))

    worst = float(np.max(np.abs((pts - origin) @ r[:, 2])))

    return origin, r, worst


def outline_from_touches(name, points, viewpoint):
    """
    Turn touched points into a cutout: pose, size and outline, ready for the config.

    The touch ORDER is the outline order. Walking the edge of a panel is how a person
    naturally describes a shape, and re-sorting the points would quietly turn a deliberate
    concave outline into its convex hull - which is exactly the shape a cockpit console is
    not.

    Returns a CutoutPlacement, or None with fewer than three points.
    """
    from .place import CutoutPlacement

    pts = np.asarray(points, float).reshape(-1, 3)

    if len(pts) < 3:
        return None

    origin, r, worst = plane_from_touches(pts, viewpoint)

    local = (pts - origin) @ r
    outline = [(float(x), float(y)) for x, y in local[:, :2]]

    centre_local = np.array([(local[:, 0].max() + local[:, 0].min()) / 2.0,
                             (local[:, 1].max() + local[:, 1].min()) / 2.0, 0.0])
    position = origin + r @ centre_local

    # Re-express the outline about the cutout's own origin, which is where the mesh is
    # built from. Leaving it about the plane's centroid would offset every shape.
    outline = [(x - centre_local[0], y - centre_local[1]) for x, y in outline]

    placement = CutoutPlacement(
        name, position, matrix_to_euler_xyz(r),
        float(local[:, 0].max() - local[:, 0].min()),
        float(local[:, 1].max() - local[:, 1].min()),
        [], worst * 1000.0, 0.0)

    # Four touched corners describe a rectangle exactly, and a rectangle is better left as
    # Width/Height: it stays adjustable by the menu's sliders, which a traced outline is
    # not.
    placement.points = [] if len(pts) == 4 and _is_rectangular(outline) else outline

    return placement


def _is_rectangular(outline, tolerance=0.004):
    """
    Whether four points are close enough to an axis-aligned rectangle in the cutout plane.

    Four metres per thousand - 4 mm - is well inside what a hand touching a bezel corner
    achieves, and well outside the deliberate shapes anyone would trace.
    """
    if len(outline) != 4:
        return False

    xs = sorted(p[0] for p in outline)
    ys = sorted(p[1] for p in outline)

    return (abs(xs[0] - xs[1]) < tolerance and abs(xs[2] - xs[3]) < tolerance
            and abs(ys[0] - ys[1]) < tolerance and abs(ys[2] - ys[3]) < tolerance)
