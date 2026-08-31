"""
Turning solved markers into cutout poses.

This is the payoff: markers on a panel define where that panel IS, so the cutout can be
placed there instead of being eighteen numbers typed by hand.

A plate's four markers sit on one rigid surface, so a plane fitted through them IS the
panel's plane. The cutout inherits that pose, and its size follows from how far apart the
markers are plus however much panel lies outside them.
"""

import numpy as np

from tracing.geometry import euler_xyz_to_matrix, matrix_to_euler_xyz


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

        # Traced outline in the cutout's own plane, in metres. Empty means the rectangle
        # from width/height, which is what the C++ falls back to as well.
        self.points = []

        # Screen holes that would not fit the config's point cap. Reported rather than
        # silently omitted: a missing hole puts passthrough over a rendered display.
        self.dropped_holes = 0

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


def panel_rects_in_plane(placements, origin, r):
    """
    Each panel's unit rectangle, as (xmin, xmax, ymin, ymax) in the common plane.

    The panels are not exactly coplanar, so projecting a tilted rectangle onto the common
    plane foreshortens it slightly. That is the right behaviour: the outline has to
    describe what the cutout covers ON ITS OWN PLANE, not what the panel measures on its.
    """
    rects = []

    for c in placements:
        hw, hh = c.width / 2.0, c.height / 2.0
        panel = euler_xyz_to_matrix(*c.euler_deg)

        corners = np.array([[-hw, -hh, 0.0], [hw, -hh, 0.0],
                            [hw, hh, 0.0], [-hw, hh, 0.0]])
        world = corners @ panel.T + c.position
        local = (world - origin) @ r

        rects.append((float(local[:, 0].min()), float(local[:, 0].max()),
                      float(local[:, 1].min()), float(local[:, 1].max())))

    return rects


def banded_outline(rects, close_gaps=True):
    """
    A staircase outline covering every rectangle, banded by row.

    A pit is not a rectangle. Three MFDs across the top with one below the centre is a T,
    and a bounding box around that wastes most of its area on cockpit sides the pilot did
    not ask to see - passthrough that is not wanted is passthrough that hides the game.

    Rectangles are grouped into rows by overlapping vertical extent, each row becomes one
    full-width band, and the bands are walked down one side and back up the other. That
    covers T, inverted T, cross, L and a single row without special cases, and yields four
    points per row - two rows is eight, well inside the 32-point cap.

    `close_gaps` pulls consecutive bands together so the outline is a single closed loop.
    A real pit has panels that touch or nearly touch, and a polygon in two disconnected
    pieces cannot be expressed as one outline at all.
    """
    if not rects:
        return []

    ordered = sorted(rects, key=lambda t: -t[3])          # topmost first
    bands = []

    for xmin, xmax, ymin, ymax in ordered:
        if bands and ymax > bands[-1][2] and ymin < bands[-1][3]:
            b = bands[-1]
            bands[-1] = [min(b[0], xmin), max(b[1], xmax),
                         min(b[2], ymin), max(b[3], ymax)]
        else:
            bands.append([xmin, xmax, ymin, ymax])

    if close_gaps:
        for i in range(len(bands) - 1):
            midpoint = (bands[i][2] + bands[i + 1][3]) / 2.0

            if bands[i][2] > bands[i + 1][3]:             # a gap, not an overlap
                bands[i][2] = midpoint
                bands[i + 1][3] = midpoint

    points = []

    for xmin, xmax, ymin, ymax in bands:                  # down the right side
        points.append((xmax, ymax))
        points.append((xmax, ymin))

    for xmin, xmax, ymin, ymax in reversed(bands):        # back up the left
        points.append((xmin, ymin))
        points.append((xmin, ymax))

    return _drop_collinear(points)


def _drop_collinear(points, tolerance=1e-6):
    """
    Remove points that lie on the line between their neighbours.

    Bands of equal width would otherwise contribute duplicate corners, which the ear
    clipper has to work around and which waste the 32-point budget for no shape.
    """
    if len(points) < 4:
        return points

    out = []
    n = len(points)

    for i in range(n):
        a = np.array(points[i - 1], float)
        b = np.array(points[i], float)
        c = np.array(points[(i + 1) % n], float)

        # numpy 2 removed the 2-vector cross product, so the z component is written out.
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])

        if abs(cross) > tolerance:
            out.append(points[i])

    return out if len(out) >= 3 else points


def shaped_cutout(placements, solutions, viewpoint, margin_mm=0.0,
                  exclude_screens=None, screen_shrink_mm=0.0):
    """
    ONE cutout whose OUTLINE follows the panels, instead of a rectangle around them.

    The pose and plane are the same common fit `cover_all` uses; only the shape differs.
    Returns the placement with its outline in `points`, in the cutout's own plane.

    `exclude_screens` is {plate name: plate}. When given, each panel's SCREEN is cut out
    of the outline as a hole, so the sim draws the MFD and passthrough covers only the
    buttons around it. `dropped_holes` records any that would not fit the 32-point budget.
    """
    pts = np.array([s.position for s in solutions.values()])

    if len(pts) < 3 or not placements:
        return None

    origin, r, _ = fit_plane_frame(pts)
    r = orient_frame_towards(origin, r, viewpoint)

    grow = margin_mm / 1000.0
    rects = [(a - grow, b + grow, c - grow, d + grow)
             for a, b, c, d in panel_rects_in_plane(placements, origin, r)]

    outline = banded_outline(rects)

    if len(outline) < 3:
        return None

    dropped = 0

    if exclude_screens is not None:
        holes = screen_rects_in_plane(placements, exclude_screens, origin, r, screen_shrink_mm)
        outline, dropped = outline_with_holes(outline, holes)

    xs = [p[0] for p in outline]
    ys = [p[1] for p in outline]

    deviations = (pts - origin) @ r[:, 2]
    span_mm = float(deviations.max() - deviations.min()) * 1000.0

    ids = sorted(int(i) for i in solutions)
    spread = float(np.mean([solutions[i].position_spread_mm for i in ids]))

    out = CutoutPlacement("cockpit", origin, matrix_to_euler_xyz(r),
                          max(xs) - min(xs), max(ys) - min(ys), ids, span_mm, spread)
    out.points = outline
    out.dropped_holes = dropped

    return out


def bridge_hole(outer, hole):
    """
    Cut a hole into an outline by bridging, so one closed loop describes both.

    The config format and the C++ ear clipper both take a SINGLE loop - there is no
    representation for a second contour. Bridging is the standard way round it: slit the
    outer boundary open at the point nearest the hole, walk the hole the OTHER way round,
    and come back along the same slit. The slit has zero width, so it draws nothing.

    The winding must be opposite, or the hole adds area instead of removing it - and the
    result would look like a slightly wrong shape rather than an obvious failure.

    Costs the hole's points plus two, so a rectangular hole is six of the 32-point budget.
    """
    outer = list(outer)
    hole = list(hole)

    if len(outer) < 3 or len(hole) < 3:
        return outer

    if _signed_area(hole) * _signed_area(outer) > 0:
        hole = hole[::-1]

    # The bridge must be able to REACH the hole without crossing anything - the closest
    # pair of vertices very often cannot.
    #
    # This was a real failure, not a theoretical one. Choosing purely by distance produced
    # an outline with 18 self-intersections on the actual cockpit, because once the first
    # hole is bridged the loop contains hole vertices and bridge segments, and the second
    # hole then bridges across them. The ear clipper rejected it and the layer fell back to
    # a rectangle - which covered every screen with camera video and looked merely like the
    # cutout being the wrong size.
    #
    # So candidates are tried nearest-first and the first VISIBLE one wins. n is at most 32,
    # so being thorough here costs nothing worth measuring.
    candidates = sorted(
        ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2, i, j)
        for i, a in enumerate(outer) for j, b in enumerate(hole))

    for _, i, j in candidates:
        if _bridge_is_clear(outer[i], hole[j], outer, hole):
            return (outer[:i + 1]
                    + hole[j:] + hole[:j + 1]
                    + outer[i:])

    return outer


def _segments_cross(a, b, c, d):
    """
    Whether segments ab and cd cross PROPERLY - touching does not count.

    All four orientations must be non-zero. A zero means an endpoint lies ON the other
    segment, which is touching, not crossing, and a bridged outline is full of those by
    construction: every bridge endpoint is shared with the contour it leaves from.

    Treating a shared endpoint as a crossing is not a harmless over-caution. It makes
    every candidate bridge look blocked, so the hole is silently dropped and the screen
    ends up covered with camera video.
    """
    def side(p, q, r):
        v = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    o1, o2 = side(a, b, c), side(a, b, d)
    o3, o4 = side(c, d, a), side(c, d, b)

    if 0 in (o1, o2, o3, o4):
        return False

    return o1 != o2 and o3 != o4


def _bridge_is_clear(a, b, outer, hole):
    """
    Whether the slit from `a` to `b` crosses any edge of either contour.

    Edges touching the bridge's own endpoints are skipped: they meet it, they do not cross
    it, and treating a shared endpoint as a crossing would reject every possible bridge.
    """
    for contour in (outer, hole):
        n = len(contour)

        for k in range(n):
            p, q = contour[k], contour[(k + 1) % n]

            if p in (a, b) or q in (a, b):
                continue

            if _segments_cross(a, b, p, q):
                return False

    return True


def _signed_area(points):
    n = len(points)

    if n < 3:
        return 0.0

    return sum(points[k][0] * points[(k + 1) % n][1] -
               points[(k + 1) % n][0] * points[k][1] for k in range(n)) / 2.0


def outline_with_holes(outer, holes):
    """
    One loop describing `outer` minus every rectangle in `holes`.

    Holes are added nearest-first so each bridge is as short as possible. A hole that
    would push the outline past the config's point cap is DROPPED rather than truncated:
    a truncated loop is not a polygon at all, and would draw as garbage or fall back to
    the rectangle with nothing to say why.

    Returns (points, dropped_count).
    """
    from tracing.config_io import MAX_POINTS

    loop = list(outer)
    dropped = 0

    centre = (sum(p[0] for p in outer) / len(outer),
              sum(p[1] for p in outer) / len(outer))

    ordered = sorted(holes, key=lambda h: min((p[0] - centre[0]) ** 2 +
                                              (p[1] - centre[1]) ** 2 for p in h))

    for hole in ordered:
        if len(loop) + len(hole) + 2 > MAX_POINTS:
            dropped += 1
            continue

        loop = bridge_hole(loop, hole)

    return loop, dropped


def rect_to_points(xmin, xmax, ymin, ymax):
    """A rectangle as four points, counter-clockwise."""
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def screen_rects_in_plane(placements, plates_by_name, origin, r, shrink_mm=0.0):
    """
    Each panel's SCREEN area projected into the common plane.

    These become holes: the sim draws the MFD content, and passthrough over it would
    replace a crisp rendered display with a camera photograph of a screen showing
    something else entirely.

    `shrink_mm` pulls each hole in, so alignment error eats into the bezel rather than
    leaving a ring of passthrough over the screen edge. Which way to err is not
    arbitrary - a little game over the bezel is invisible, a little camera over the screen
    is not.
    """
    rects = []

    for c in placements:
        plate = plates_by_name.get(c.name)

        if plate is None:
            continue

        u = plate["usable_mm"]
        w_mm, h_mm, dx_mm, dy_mm = cutout_extent(plate)

        # The screen sits where the unit's own offset says it does, mirrored: the unit was
        # moved by (dx, dy) away from the screen, so the screen is that far back.
        cx = -dx_mm / 1000.0
        cy = -dy_mm / 1000.0
        hw = max(u["w"] / 2.0 - shrink_mm, 1.0) / 1000.0
        hh = max(u["h"] / 2.0 - shrink_mm, 1.0) / 1000.0

        panel = euler_xyz_to_matrix(*c.euler_deg)
        corners = np.array([[cx - hw, cy - hh, 0.0], [cx + hw, cy - hh, 0.0],
                            [cx + hw, cy + hh, 0.0], [cx - hw, cy + hh, 0.0]])
        world = corners @ panel.T + c.position
        local = (world - origin) @ r

        rects.append([(float(x), float(y)) for x, y in local[:, :2]])

    return rects


def fit_similarity(source, target):
    """
    Like `fit_rigid`, but WITH scale - Umeyama.

    Returns (scale, rotation, translation, worst_residual_metres). `fit_rigid` deliberately
    refuses scale so a range error shows up as residual instead of being absorbed; this is
    the other half of that decision, used to MEASURE the error rather than hide it.
    """
    a = np.asarray(source, float).reshape(-1, 3)
    b = np.asarray(target, float).reshape(-1, 3)

    ca, cb = a.mean(axis=0), b.mean(axis=0)
    a0, b0 = a - ca, b - cb

    u, sv, vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T

    variance = float((a0 ** 2).sum())
    scale = 1.0 if variance < 1e-12 else float((sv * np.array([1.0, 1.0, d])).sum() / variance)

    t = cb - scale * (r @ ca)
    worst = float(np.max(np.linalg.norm(b0 - a0 @ (scale * r).T, axis=1)))

    return scale, r, t, worst


def measure_range_scale(plates, solutions):
    """
    How far off the solved RANGE is, measured against the plates' known geometry.

    This is the honest answer to "the pit looks too far away", and it needs no headset.
    Each plate's marker layout is known, so fitting it WITH scale reads the error straight
    off: a constellation solved k times too big is a constellation solved k times too far,
    because range and apparent size are the same measurement.

    The correction to apply is 1/k.

    A uniform pixel-pitch error would NOT show up here and must not be confused with this
    one: it scales the assumed marker size and the assumed spacing together, the two
    cancel, and k comes out 1. A non-unit k means the size and the spacing DISAGREE.

    Returns a list of (name, k, worst_residual_mm), best-conditioned first.
    """
    out = []

    for plate in plates:
        local = plate_local_points(plate)
        have = [int(i) for i in plate["ids"] if int(i) in solutions and int(i) in local]

        if len(have) < 3:
            continue

        source = np.array([local[i] for i in have])
        target = np.array([solutions[i].position for i in have])

        scale, _, _, worst = fit_similarity(source, target)
        out.append((plate.get("name", "plate"), scale, worst * 1000.0))

    return sorted(out, key=lambda row: row[2])
