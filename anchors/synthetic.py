"""
Synthetic marker observations.

The solver can be checked completely without a camera, a headset or a cockpit: place
markers at known poses, put a virtual camera at known poses, project the corners, and
see whether the solver recovers what went in.

This is the same trick as tracing.capture.synthetic_capture, and it is what makes the
anchoring work iterable at a desk instead of one headset session per change.
"""

import numpy as np

from tracing.geometry import Camera, camera_to_world_from_hmd, pose_to_matrix

from .solver import Observation, marker_object_points


def marker_pose(position, euler_deg=(0.0, 0.0, 0.0)):
    """A marker's ground-truth pose in world coordinates."""
    return pose_to_matrix(position, euler_deg)


def project_marker(camera, pose, size_mm):
    """
    A marker's four corners as pixels, or None if any corner is behind the camera or
    outside the image.

    Rejecting out-of-frame corners matters: cv2.undistortPoints and the PnP solvers
    misbehave well outside the frame, so a test fixture that placed a marker off-axis
    would fail in a way that looks like a solver bug.
    """
    corners = []

    for local in marker_object_points(size_mm):
        world = (pose @ np.append(local, 1.0))[:3]
        px = camera.project(world)

        if px is None or not camera.in_frame(*px):
            return None

        corners.append(px)

    return np.array(corners, np.float64)


def observe(markers, camera_poses, size_mm_of, image_size=(1600, 1200),
            K=None, dist=None, camera_offset=(-0.031, -0.047, -0.138),
            pixel_noise=0.0, seed=0):
    """
    Generate observations of `markers` from a sequence of head poses.

    markers      {marker_id: 4x4 marker-to-world}
    camera_poses list of 4x4 HMD-to-world, one per frame
    size_mm_of   {marker_id: physical size in mm}
    pixel_noise  standard deviation of per-corner noise, to model click/detector error

    Returns (observations, cameras) ready for solve_markers.
    """
    if K is None:
        K = np.array([[1072.26851867, 0.0, 788.49299729],
                      [0.0, 1072.31519651, 614.54444602],
                      [0.0, 0.0, 1.0]])
    if dist is None:
        dist = np.array([0.08313216691950971, -0.10744697901181298,
                         -0.00016821021003468, 0.00025331486744491, 0.0])

    rng = np.random.default_rng(seed)
    observations = []
    cameras = {}

    for frame, hmd in enumerate(camera_poses):
        cam = Camera(K, dist, camera_to_world_from_hmd(hmd, camera_offset),
                     image_size=image_size)
        cameras[frame] = cam

        for marker_id, pose in markers.items():
            size = size_mm_of[marker_id]
            corners = project_marker(cam, pose, size)

            if corners is None:
                continue

            if pixel_noise > 0.0:
                corners = corners + rng.normal(0.0, pixel_noise, corners.shape)

            observations.append(Observation(marker_id, corners,
                                            cam.camera_to_world, size, frame))

    return observations, cameras


def plate_markers(origin, euler_deg, spread_mm, ids, size_mm):
    """
    Four markers at the corners of a square plate - the WinCtrl display panel layout.

    Returns {id: pose}. Marker frames inherit the plate's orientation, which is what a
    real plate does: the markers are printed or displayed on one rigid surface.
    """
    from tracing.geometry import euler_xyz_to_matrix

    r = euler_xyz_to_matrix(*euler_deg)
    o = np.asarray(origin, float)
    h = spread_mm / 2000.0

    offsets = [(-h, -h), (h, -h), (h, h), (-h, h)]
    out = {}

    for marker_id, (u, v) in zip(ids, offsets):
        pose = np.eye(4)
        pose[:3, :3] = r
        pose[:3, 3] = o + r[:, 0] * u + r[:, 1] * v
        out[marker_id] = pose

    return out


def arc_of_head_poses(centre, radius, count, height=1.15, yaw_span_deg=40.0,
                      pitch_deg=-15.0):
    """
    A short sweep of head poses looking at `centre`, as a user would while calibrating.

    Multiple viewpoints are what let a solve average down per-view error, so a fixture
    with one viewpoint would flatter the solver.
    """
    centre = np.asarray(centre, float)
    poses = []

    for i in range(count):
        t = 0.0 if count == 1 else (i / (count - 1) - 0.5)
        yaw = t * yaw_span_deg
        pos = centre + np.array([np.sin(np.radians(yaw)) * radius, 0.0,
                                 np.cos(np.radians(yaw)) * radius])
        pos[1] = height
        poses.append(pose_to_matrix(pos, (pitch_deg, yaw, 0.0)))

    return poses
