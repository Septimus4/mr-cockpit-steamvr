"""
Turning solved markers into cutout poses.

This is the payoff: markers on a panel define where that panel IS, so the cutout can be
placed there instead of being eighteen numbers typed by hand.

A plate's four markers sit on one rigid surface, so a plane fitted through them IS the
panel's plane. The cutout inherits that pose, and its size follows from how far apart the
markers are plus however much panel lies outside them.
"""

import numpy as np

from tracing.geometry import matrix_to_euler_xyz


def fit_plane_frame(points):
    """
    A right-handed frame for a set of roughly coplanar points.

    Returns (origin, 3x3 rotation). X and Y span the plane, Z is its normal. The normal
    comes from the smallest singular value, which is the least-squares best-fit plane -
    not from any three chosen points, since that would make the result depend on which
    three.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    origin = pts.mean(axis=0)
    centred = pts - origin

    u, s, vt = np.linalg.svd(centred)

    x_axis = vt[0] / np.linalg.norm(vt[0])
    normal = vt[2] / np.linalg.norm(vt[2])
    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    r = np.column_stack([x_axis, y_axis, normal])

    if np.linalg.det(r) < 0:                 # keep it right-handed
        r = np.column_stack([x_axis, -y_axis, normal])

    return origin, r, float(s[2])


def orient_frame_towards(origin, r, viewpoint):
    """
    Flip the frame so its normal faces the viewer, and its Y points up.

    Without this the plane's orientation is arbitrary - SVD has no notion of which side
    is the front - and a cutout would face away half the time. A quad is double-sided so
    it would still draw, but its stored rotation would be meaningless to read.
    """
    to_viewer = np.asarray(viewpoint, float) - origin

    if r[:, 2] @ to_viewer < 0:
        r = np.column_stack([r[:, 0], -r[:, 1], -r[:, 2]])

    if r[:, 1][1] < 0:                       # Y axis pointing down in world terms
        r = np.column_stack([-r[:, 0], -r[:, 1], r[:, 2]])

    return r


class CutoutPlacement:
    """A cutout pose derived from markers, in the form the config stores."""

    def __init__(self, name, position, euler_deg, width, height, marker_ids,
                 flatness_mm, spread_mm):
        self.name = name
        self.position = position
        self.euler_deg = euler_deg
        self.width = width
        self.height = height
        self.marker_ids = marker_ids
        self.flatness_mm = flatness_mm
        self.spread_mm = spread_mm

    def __repr__(self):
        return (f"{self.name}: {np.round(self.position, 4)} "
                f"rot {tuple(round(v, 2) for v in self.euler_deg)} "
                f"{self.width * 1000:.0f} x {self.height * 1000:.0f} mm")


def place_from_markers(name, solutions, marker_ids, viewpoint, margin_mm=0.0):
    """
    Fit a cutout to a group of solved markers.

    `margin_mm` extends the cutout beyond the markers - a panel usually continues past
    the markers stuck or drawn on it.

    Returns None if fewer than three markers are available, since a plane needs three and
    two would give an arbitrary orientation rather than an error.
    """
    have = [i for i in marker_ids if i in solutions]

    if len(have) < 3:
        return None

    pts = np.array([solutions[i].position for i in have])
    origin, r, flatness = fit_plane_frame(pts)
    r = orient_frame_towards(origin, r, viewpoint)

    local = (pts - origin) @ r               # marker positions in the plane's own frame
    width = float(local[:, 0].max() - local[:, 0].min()) + margin_mm / 1000.0
    height = float(local[:, 1].max() - local[:, 1].min()) + margin_mm / 1000.0

    # Centre the cutout on the markers rather than on their bounding-box corner.
    centre_local = np.array([(local[:, 0].max() + local[:, 0].min()) / 2.0,
                             (local[:, 1].max() + local[:, 1].min()) / 2.0, 0.0])
    position = origin + r @ centre_local

    spread = float(np.mean([solutions[i].position_spread_mm for i in have]))

    return CutoutPlacement(name, position, matrix_to_euler_xyz(r), width, height,
                           have, flatness * 1000.0, spread)


def group_by_plate(solutions, plates):
    """
    Group solved markers by the plate they belong to.

    `plates` is a list of dicts with 'name' and 'ids', as the plate JSON files provide.
    Markers belonging to no plate are returned separately - they are the loose stickers,
    which anchor the cockpit frame rather than defining a panel.
    """
    grouped = []
    claimed = set()

    for plate in plates:
        ids = [int(i) for i in plate.get("ids", [])]
        present = [i for i in ids if i in solutions]

        if present:
            grouped.append((plate.get("name", "plate"), ids))
            claimed.update(present)

    loose = sorted(i for i in solutions if i not in claimed)
    return grouped, loose


def plate_local_points(plate):
    """
    Where a display plate's markers sit in the CUTOUT's own frame, in metres.

    The plate JSON records each marker centre in panel millimetres, together with the
    usable rectangle. That makes the layout KNOWN rather than inferred, which is worth a
    great deal: a known layout can be fitted rigidly, and the leftover residual is then a
    real measurement of how well the solve agrees with the physical panel.

    Panel coordinates run Y DOWN, like the screen. The cutout frame runs Y up, so Y is
    flipped here. Getting that wrong does not fail - it mirrors the cutout, which looks
    like a tracking fault rather than a sign error.

    Returns {marker_id: (x, y, 0)}.
    """
    u = plate["usable_mm"]
    cx = u["x"] + u["w"] / 2.0
    cy = u["y"] + u["h"] / 2.0

    out = {}

    for marker_id, (mx, my) in zip(plate["ids"], plate["centres_mm"]):
        out[int(marker_id)] = np.array([(mx - cx) / 1000.0, (cy - my) / 1000.0, 0.0])

    return out


def fit_rigid(source, target):
    """
    The rotation and translation carrying `source` onto `target` - Kabsch.

    Returns (rotation, translation, rms_error_metres). No scaling is fitted: the plate's
    size is measured, not a free parameter, so letting scale float would quietly absorb a
    range error that ought to show up as residual instead.
    """
    a = np.asarray(source, float).reshape(-1, 3)
    b = np.asarray(target, float).reshape(-1, 3)

    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)

    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:            # a reflection fits points but is not a pose
        vt[-1] *= -1
        r = vt.T @ u.T

    t = cb - r @ ca
    rms = float(np.sqrt(np.mean(np.sum((a @ r.T + t - b) ** 2, axis=1))))

    return r, t, rms


def place_from_plate(plate, solutions, margin_mm=0.0):
    """
    Fit a cutout to a display plate whose marker layout is known.

    Better than fitting a bare plane through the markers: the size comes from the
    measured usable rectangle rather than from where the markers happen to fall, the
    centre is the panel's centre rather than the markers' bounding box, and the residual
    is a genuine check - it is the disagreement between the solved constellation and the
    panel's real geometry.

    Returns None if fewer than three of the plate's markers were solved.
    """
    local = plate_local_points(plate)
    have = [i for i in plate["ids"] if int(i) in solutions and int(i) in local]

    if len(have) < 3:
        return None

    src = np.array([local[int(i)] for i in have])
    dst = np.array([solutions[int(i)].position for i in have])

    r, t, rms = fit_rigid(src, dst)

    w_mm, h_mm, dx_mm, dy_mm = cutout_extent(plate)

    width = (w_mm + margin_mm) / 1000.0
    height = (h_mm + margin_mm) / 1000.0

    # The unit's centre is not the screen's centre, and the offset lives in the CUTOUT's
    # plane, so it has to be rotated into the world before being added.
    position = t + r @ np.array([dx_mm / 1000.0, dy_mm / 1000.0, 0.0])

    spread = float(np.mean([solutions[int(i)].position_spread_mm for i in have]))

    return CutoutPlacement(plate.get("name", "plate"), position, matrix_to_euler_xyz(r),
                           width, height, [int(i) for i in have], rms * 1000.0, spread)


def cutout_extent(plate):
    """
    How big the cutout should be, and where its centre sits relative to the screen's.

    Returns (width_mm, height_mm, dx_mm, dy_mm).

    The markers can only ever measure the SCREEN, because that is what draws them. But
    the point of a cutout is the physical BUTTONS around the screen, so the screen's
    extent is the wrong answer by design - for the WinCtrl MFDs it is 118 x 118 mm inside
    a 167 x 185 mm unit, which is less than half the area that matters.

    `unit_mm` in the plate JSON carries the real thing: w, h, and an optional dx/dy for
    units whose screen aperture is not centred in the housing. Without it this falls back
    to the usable screen area, which is honest but small.
    """
    u = plate["usable_mm"]
    unit = plate.get("unit_mm")

    if not unit:
        return float(u["w"]), float(u["h"]), 0.0, 0.0

    return (float(unit["w"]), float(unit["h"]),
            float(unit.get("dx", 0.0)), float(unit.get("dy", 0.0)))


def cover_all(placed, solutions, width_mm, height_mm, viewpoint):
    """
    ONE cutout spanning every solved panel, on their common best-fit plane.

    The goal is the buttons and the console, not the screens - and the console carries no
    markers at all. A single hole over the whole assembly reaches all of it, is one mesh
    instead of several, and is by far the quickest way to find out whether anchoring works
    at all.

    The cost is flattening: the panels are not exactly coplanar, so a flat cutout sits
    slightly in front of or behind each surface, and passthrough at the wrong depth shifts
    sideways by roughly `baseline x deviation / distance^2`. Returns the placement with
    that deviation in `flatness_mm` so the caller can report the price rather than hide it.
    """
    if not placed:
        return None

    pts = np.array([s.position for s in solutions.values()])

    if len(pts) < 3:
        return None

    origin, r, _ = fit_plane_frame(pts)
    r = orient_frame_towards(origin, r, viewpoint)

    # Centre on the markers' own centroid, projected into the plane. Sizing is given, not
    # measured: the assembly extends well past the last marker, which is the whole point.
    deviations = (pts - origin) @ r[:, 2]
    span_mm = float(deviations.max() - deviations.min()) * 1000.0

    ids = sorted(int(i) for i in solutions)
    spread = float(np.mean([solutions[i].position_spread_mm for i in ids]))

    return CutoutPlacement("cockpit", origin, matrix_to_euler_xyz(r),
                           width_mm / 1000.0, height_mm / 1000.0, ids, span_mm, spread)


def flattening_cost_mm(deviation_mm, distance_m, baseline_m=0.15):
    """
    How far passthrough shifts sideways when a cutout sits at the wrong depth.

    The camera is not at the eye, so a surface rendered at the wrong distance lands in the
    wrong place: `baseline x deviation / distance^2`. This is the same relation that
    governs the whole project's depth budget, applied to the error a FLAT cutout makes
    over a not-quite-flat assembly.
    """
    if distance_m <= 1e-6:
        return 0.0

    return float(baseline_m * (deviation_mm / 1000.0) / (distance_m ** 2) * 1000.0)
