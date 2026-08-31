"""
Solving marker poses from camera observations.

This is what makes cutouts place themselves. Today every cutout pose is typed by hand -
three cutouts is eighteen numbers. Here the camera sees the markers and the poses fall
out, to a precision no one could reach by eye.

Frames follow tracing.geometry: camera is OpenCV (X right, Y down, Z forward), world is
OpenXR stage (X right, Y up, -Z forward).

Marker frame
------------
A marker's own frame has X right, Y UP and Z out of its face, matching OpenCV's ArUco
convention. Its corners in that frame, in the order cv2.aruco.detectMarkers returns them
(top-left, top-right, bottom-right, bottom-left as seen in the image):

    (-s/2, +s/2, 0)  (+s/2, +s/2, 0)  (+s/2, -s/2, 0)  (-s/2, -s/2, 0)

Getting that order wrong does not fail - it yields a marker rotated by a multiple of 90
degrees, which looks like a mounting mistake rather than a code one.
"""

import numpy as np


def marker_object_points(size_mm):
    """A marker's four corners in its own frame, in METRES, in detector order."""
    h = float(size_mm) / 2000.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], np.float64)


class Observation:
    """One marker seen in one frame."""

    def __init__(self, marker_id, corners_px, camera_to_world, size_mm, frame=0):
        self.marker_id = int(marker_id)
        self.corners_px = np.asarray(corners_px, np.float64).reshape(4, 2)
        self.camera_to_world = np.asarray(camera_to_world, np.float64).reshape(4, 4)
        self.size_mm = float(size_mm)
        self.frame = int(frame)

    def __repr__(self):
        return (f"Observation(id={self.marker_id}, frame={self.frame}, "
                f"size={self.size_mm}mm)")


def solve_marker_pose(camera, corners_px, size_mm):
    """
    One marker's pose in WORLD coordinates, from one view.

    Returns (4x4 marker-to-world, reprojection error in pixels), or None if the solve
    fails. Uses IPPE_SQUARE, which is built for planar square markers and is markedly
    steadier than the iterative solver.

    Size comes from the caller because it comes from the marker's id - see
    scripts/marker_ids.py. A wrong size does not fail, it puts the error straight into
    range: a 22.4 mm marker solved as 37.3 mm sits 67% too far away.
    """
    import cv2

    obj = marker_object_points(size_mm)
    img = np.asarray(corners_px, np.float64).reshape(4, 2)

    ok, rvec, tvec = cv2.solvePnP(obj, img, camera.K, camera.dist,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None

    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera.K, camera.dist)
    err = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - img, axis=1)))

    marker_to_camera = np.eye(4)
    marker_to_camera[:3, :3] = cv2.Rodrigues(rvec)[0]
    marker_to_camera[:3, 3] = tvec.ravel()

    return camera.camera_to_world @ marker_to_camera, err


def average_rotations(mats):
    """
    Robust average of rotation matrices, via the medoid then a projected mean.

    A plain elementwise mean of rotation matrices is not a rotation, and the planar pose
    ambiguity means a minority of solves can be flipped by a large angle. Starting from
    the medoid - the sample closest to all the others - keeps a flipped minority from
    dragging the result.
    """
    mats = list(mats)

    if len(mats) == 1:
        return mats[0]

    def angle(a, b):
        return np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))

    cost = [sum(angle(m, other) for other in mats) for m in mats]
    medoid = mats[int(np.argmin(cost))]

    # Keep only samples near the medoid, so an ambiguity flip cannot pull the mean.
    keep = [m for m in mats if angle(medoid, m) < 20.0] or [medoid]

    m = np.mean(keep, axis=0)
    u, _, vt = np.linalg.svd(m)
    r = u @ vt

    if np.linalg.det(r) < 0:            # reflection, not a rotation
        u[:, -1] *= -1
        r = u @ vt

    return r


class MarkerSolution:
    """Where one marker ended up, and how much to trust it."""

    def __init__(self, marker_id, pose, observations, position_spread_mm,
                 angle_spread_deg, reprojection_px, max_skew=0.0, worst_mm=0.0):
        self.marker_id = marker_id
        self.pose = pose
        self.observations = observations
        self.position_spread_mm = position_spread_mm
        self.angle_spread_deg = angle_spread_deg
        self.reprojection_px = reprojection_px

        # How obliquely this marker was ever seen. A marker only ever viewed SQUARE-ON is
        # ambiguous: two poses reproject almost identically and the solver may pick
        # either. Measured on real cockpit data - a panel mounted perpendicular to the
        # pilot scattered 3x worse than its neighbours, and no amount of averaging fixed
        # it, because every view had the same problem.
        self.max_skew = max_skew

        # The single furthest view, kept separately: a marker that is usually fine
        # but occasionally wild is a different fault from one that is consistently
        # noisy, and the robust spread alone cannot tell them apart.
        self.worst_mm = worst_mm

    @property
    def position(self):
        return self.pose[:3, 3]

    def __repr__(self):
        return (f"Marker {self.marker_id}: {self.observations} obs, "
                f"spread {self.position_spread_mm:.2f} mm / "
                f"{self.angle_spread_deg:.2f} deg, reproj {self.reprojection_px:.2f} px")


def view_skew(corners_px):
    """
    How far from square-on a marker was seen: 0 is perfectly perpendicular.

    Computed from the difference between the two diagonals, which is zero for a square
    seen head-on and grows with obliquity. Cheap, and it needs no pose.
    """
    p = np.asarray(corners_px, float).reshape(4, 2)
    d0 = np.linalg.norm(p[0] - p[2])
    d1 = np.linalg.norm(p[1] - p[3])
    longest = max(d0, d1)

    return 0.0 if longest < 1e-9 else float(abs(d0 - d1) / longest)


def solve_markers(observations, cameras):
    """
    Solve every observed marker's pose in world coordinates.

    `cameras` maps a frame index to the Camera for that frame. Camera poses are taken as
    known: the headset is Lighthouse-tracked, so refining them would be fitting noise
    against a better measurement than the markers provide.

    Returns {marker_id: MarkerSolution}. The spreads are how much the per-view solves
    disagreed, which is the only honest confidence signal available without ground truth.
    """
    by_id = {}

    for obs in observations:
        cam = cameras.get(obs.frame)
        if cam is None:
            continue

        solved = solve_marker_pose(cam, obs.corners_px, obs.size_mm)
        if solved is None:
            continue

        pose, err = solved
        by_id.setdefault(obs.marker_id, []).append((pose, err, view_skew(obs.corners_px)))

    out = {}

    for marker_id, entries in by_id.items():
        poses = [p for p, _, _ in entries]
        errs = [e for _, e, _ in entries]
        skews = [k for _, _, k in entries]

        positions = np.array([p[:3, 3] for p in poses])
        rotation = average_rotations([p[:3, :3] for p in poses])
        position = np.median(positions, axis=0)

        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position

        # ROBUST spread, not the maximum. A max is dominated by one outlier and grows
        # with sample count by construction: it reported 65 mm for a marker whose real
        # scatter was 3.5 mm, and made captures of different lengths incomparable. This
        # is a MAD-based sigma - half the views lie within it.
        distances = np.linalg.norm(positions - position, axis=1) * 1000.0
        spread_mm = float(1.4826 * np.median(distances)) if len(positions) > 1 else 0.0
        worst_mm = float(np.max(distances)) if len(positions) > 1 else 0.0

        if len(poses) > 1:
            angles = [np.degrees(np.arccos(np.clip(
                (np.trace(rotation.T @ p[:3, :3]) - 1) / 2, -1, 1))) for p in poses]
            angle_spread = float(1.4826 * np.median(angles))
        else:
            angle_spread = 0.0

        out[marker_id] = MarkerSolution(marker_id, pose, len(poses), spread_mm,
                                        angle_spread, float(np.mean(errs)),
                                        float(np.max(skews)) if skews else 0.0,
                                        worst_mm)

    return out


def constellation_conditioning(solutions):
    """
    How well-conditioned the solved marker constellation is.

    Returns (aspect, verdict, extent_mm). Near-collinear constellations AMPLIFY
    systematic error rather than averaging it down, and their accuracy then depends on
    where the markers happen to fall in the lens. Measured: a 3.6:1 strip swung 9x in
    lateral accuracy with position, a 1:1 spread swung 2x.

    A user cannot judge this by eye, which is exactly why it is measured before a solve
    is accepted.
    """
    if len(solutions) < 3:
        return float("inf"), "TOO FEW", 0.0

    pts = np.array([s.position for s in solutions])
    centred = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)

    extent_mm = float(sv[0]) * 1000.0

    if sv[1] < 1e-9:
        return float("inf"), "COLLINEAR", extent_mm

    aspect = float(sv[0] / sv[1])
    verdict = "GOOD" if aspect < 2.0 else ("MARGINAL" if aspect < 3.0 else "TOO COLLINEAR")

    return aspect, verdict, extent_mm


def is_coplanar(solutions, tolerance_mm=10.0):
    """
    Whether the markers all lie in one plane.

    Coplanar constellations cannot resolve the planar tilt ambiguity - measured on
    hardware, a coplanar set still flipped. Depth spread is what breaks it, so this is
    worth reporting as a property of the layout rather than discovering as instability.
    """
    if len(solutions) < 4:
        return True, 0.0

    pts = np.array([s.position for s in solutions])
    centred = pts - pts.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False)

    out_of_plane_mm = float(sv[2]) * 1000.0
    return out_of_plane_mm < tolerance_mm, out_of_plane_mm
