"""
Pure geometry for the cutout tracing tool.

Tracing a cutout outline means: the user clicks a point in a camera image, and we work
out where that click lands on the cutout's plane. That is entirely deterministic maths -
no camera, no headset - so it lives here on its own and is covered by unit tests.

Frames
------
camera : OpenCV convention. X right, Y down, Z forward (into the scene).
world  : OpenXR/stage convention. X right, Y up, Z back (so -Z is forward).
plane  : the cutout's own frame. X right, Y up, Z out of the plane toward the viewer.
         Outline points are (x, y) in this frame, in metres, origin at the cutout pose.

Rotation convention
-------------------
Intrinsic Euler XYZ in DEGREES, composing as R = Rz @ Ry @ Rx. This MUST match
GetQuadToWorldTransform in passthrough_renderer.h - if the two disagree, traced outlines
land rotated and nothing about the failure points at the cause. test_geometry.py pins
the convention down with explicit fixtures.
"""

import numpy as np


# --------------------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------------------

def rotation_x(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rotation_y(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rotation_z(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def euler_xyz_to_matrix(rx_deg, ry_deg, rz_deg):
    """Intrinsic XYZ in degrees -> 3x3. Composes as Rz @ Ry @ Rx, matching the C++ side."""
    return rotation_z(rz_deg) @ rotation_y(ry_deg) @ rotation_x(rx_deg)


def matrix_to_euler_xyz(r):
    """
    3x3 rotation -> intrinsic Euler XYZ in DEGREES, the inverse of euler_xyz_to_matrix.

    This is what turns a solved marker pose back into the RotX/RotY/RotZ the config
    stores, so it MUST invert the same composition the renderer applies (Rz @ Ry @ Rx).
    An error here does not fail - it writes a cutout that is rotated, and the user would
    read that as bad tracking rather than a conversion bug.

    Near gimbal lock (pitch at +/-90 degrees) the X and Z angles are not separable; the
    convention there is to put the whole rotation into X.
    """
    r = np.asarray(r, float).reshape(3, 3)

    sy = -r[2, 0]
    sy = float(np.clip(sy, -1.0, 1.0))
    ry = np.arcsin(sy)

    if abs(sy) > 1.0 - 1e-9:                      # gimbal lock
        rx = np.arctan2(-r[1, 2], r[1, 1])
        rz = 0.0
    else:
        rx = np.arctan2(r[2, 1], r[2, 2])
        rz = np.arctan2(r[1, 0], r[0, 0])

    return tuple(float(np.degrees(v)) for v in (rx, ry, rz))


def pose_to_matrix(position, euler_deg):
    """4x4 local-to-world for a cutout pose."""
    m = np.eye(4)
    m[:3, :3] = euler_xyz_to_matrix(*euler_deg)
    m[:3, 3] = position
    return m


# --------------------------------------------------------------------------------------
# planes
# --------------------------------------------------------------------------------------

class Plane:
    """A cutout's plane: an origin and an orthonormal basis, all in world coordinates."""

    def __init__(self, position, euler_deg):
        self.position = np.asarray(position, float)
        self.euler_deg = tuple(float(v) for v in euler_deg)
        r = euler_xyz_to_matrix(*self.euler_deg)
        self.x_axis = r[:, 0]
        self.y_axis = r[:, 1]
        self.normal = r[:, 2]

    def to_world(self, u, v):
        """Plane coordinates (metres) -> world point."""
        return self.position + self.x_axis * u + self.y_axis * v

    def to_plane(self, point_world):
        """World point -> (u, v) in plane coordinates. Any out-of-plane part is dropped."""
        d = np.asarray(point_world, float) - self.position
        return float(d @ self.x_axis), float(d @ self.y_axis)

    def distance_to(self, point_world):
        """Signed distance from the plane, positive along the normal."""
        return float((np.asarray(point_world, float) - self.position) @ self.normal)


def intersect_ray_plane(ray_origin, ray_dir, plane, min_t=1e-6):
    """
    Where a ray meets a plane, or None.

    Returns None when the ray is parallel to the plane, or when the intersection is
    behind the ray origin - a click that would land behind the camera must not silently
    produce a point.
    """
    ray_origin = np.asarray(ray_origin, float)
    ray_dir = np.asarray(ray_dir, float)

    denom = ray_dir @ plane.normal

    if abs(denom) < 1e-9:
        return None

    t = ((plane.position - ray_origin) @ plane.normal) / denom

    if t < min_t:
        return None

    return ray_origin + ray_dir * t


# --------------------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------------------

class Camera:
    """
    Pinhole camera with Brown-Conrady distortion, plus its pose in the world.

    camera_to_world maps camera-frame points to world. Because the camera frame is Y-down
    and Z-forward while the world is Y-up and -Z-forward, that matrix carries the axis
    flip as well as the pose; callers build it with camera_to_world_from_hmd().
    """

    def __init__(self, K, dist, camera_to_world, image_size=(1600, 1200)):
        self.K = np.asarray(K, float).reshape(3, 3)
        self.dist = np.asarray(dist, float).ravel()
        self.camera_to_world = np.asarray(camera_to_world, float).reshape(4, 4)
        self.image_size = (int(image_size[0]), int(image_size[1]))

    def in_frame(self, px, py):
        """
        Whether a pixel is actually inside the image.

        Worth having explicitly: cv2.undistortPoints solves iteratively and diverges well
        outside the frame, so a point far off-axis back-projects to nonsense rather than
        failing. Real clicks are always in frame; tests and callers should check.
        """
        return 0 <= px < self.image_size[0] and 0 <= py < self.image_size[1]

    @property
    def position(self):
        return self.camera_to_world[:3, 3]

    def pixel_to_ray(self, px, py):
        """Pixel -> unit ray direction in WORLD space, with distortion removed."""
        import cv2

        pts = np.array([[[float(px), float(py)]]], dtype=np.float64)
        norm = cv2.undistortPoints(pts, self.K, self.dist).reshape(2)

        # OpenCV normalised coords are already X right, Y down, Z = 1 forward.
        d_cam = np.array([norm[0], norm[1], 1.0])
        d_world = self.camera_to_world[:3, :3] @ d_cam

        n = np.linalg.norm(d_world)
        if n < 1e-12:
            raise ValueError("degenerate ray direction")

        return d_world / n

    def project(self, point_world):
        """
        World point -> pixel, the inverse of pixel_to_ray. Exists so the tests can assert
        a round trip; the tool itself never needs it.

        Returns None for points behind the camera.
        """
        import cv2

        world_to_camera = np.linalg.inv(self.camera_to_world)
        p_cam = world_to_camera @ np.append(np.asarray(point_world, float), 1.0)

        if p_cam[2] <= 1e-9:
            return None

        img, _ = cv2.projectPoints(
            p_cam[:3].reshape(1, 1, 3),
            np.zeros(3), np.zeros(3), self.K, self.dist)

        return tuple(img.reshape(2))


def camera_to_world_from_hmd(hmd_to_world, camera_offset, camera_euler_deg=(0.0, 0.0, 0.0)):
    """
    Build camera_to_world from the headset pose and the calibrated camera offset.

    The axis flip lives here: OpenCV camera space is Y-down / Z-forward, the world is
    Y-up / -Z-forward, so the camera basis is the HMD basis with Y and Z negated. Getting
    this wrong mirrors traced outlines vertically, which looks like a tracing mistake
    rather than a frame error - hence it is in one place, and tested.
    """
    hmd_to_world = np.asarray(hmd_to_world, float).reshape(4, 4)

    flip = np.diag([1.0, -1.0, -1.0])
    r_cam = hmd_to_world[:3, :3] @ euler_xyz_to_matrix(*camera_euler_deg) @ flip

    m = np.eye(4)
    m[:3, :3] = r_cam
    m[:3, 3] = hmd_to_world[:3, 3] + hmd_to_world[:3, :3] @ np.asarray(camera_offset, float)
    return m


# --------------------------------------------------------------------------------------
# the operation the tool actually performs
# --------------------------------------------------------------------------------------

def backproject_click(camera, plane, px, py):
    """
    A click at (px, py) in the camera image -> (u, v) on the cutout's plane, in metres.

    Returns None when the click does not land on the plane at all - looking past the
    panel, or the plane edge-on. The caller must handle that rather than recording a
    bogus point.
    """
    ray_dir = camera.pixel_to_ray(px, py)
    hit = intersect_ray_plane(camera.position, ray_dir, plane)

    if hit is None:
        return None

    return plane.to_plane(hit)


def backproject_outline(camera, plane, pixels):
    """
    Back-project a whole traced outline. Returns (points, dropped) where dropped counts
    clicks that missed the plane, so the caller can tell the user rather than silently
    producing a shorter outline.
    """
    points = []
    dropped = 0

    for px, py in pixels:
        uv = backproject_click(camera, plane, px, py)

        if uv is None:
            dropped += 1
            continue

        points.append(uv)

    return points, dropped


# --------------------------------------------------------------------------------------
# outline hygiene
# --------------------------------------------------------------------------------------

def point_to_segment_distance(p, a, b):
    """Shortest distance from point p to the segment ab."""
    p = np.asarray(p, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    ab = b - a
    denom = ab @ ab

    if denom < 1e-18:
        return float(np.linalg.norm(p - a))

    t = float(np.clip(((p - a) @ ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + ab * t)))


def point_to_outline_distance(p, outline):
    """
    Shortest distance from p to a closed outline's EDGES.

    Distance to the nearest vertex is the wrong measure for how faithful a simplified
    outline is: a point halfway along a straight edge is far from both endpoints while
    lying exactly on the edge. Simplification error has to be measured against the edges.
    """
    n = len(outline)

    if n == 0:
        return float("inf")
    if n == 1:
        return float(np.linalg.norm(np.asarray(p, float) - np.asarray(outline[0], float)))

    return min(point_to_segment_distance(p, outline[i], outline[(i + 1) % n])
               for i in range(n))


def polygon_signed_area(points):
    """Positive for counter-clockwise. Matches PolygonSignedArea in mesh.cpp."""
    n = len(points)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i - 1) % n
        area += points[j][0] * points[i][1] - points[i][0] * points[j][1]

    return area * 0.5


def is_simple_polygon(points):
    """
    True when no two non-adjacent edges cross.

    Worth checking before saving: ear clipping rejects a self-intersecting outline, and
    the C++ then falls back to a rectangle. Catching it while the user is still drawing
    lets them fix it, instead of wondering why their cutout became a rectangle.
    """
    n = len(points)
    if n < 3:
        return False

    def seg_intersect(a, b, c, d):
        def orient(p, q, r):
            v = (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])
            return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

        o1, o2 = orient(a, b, c), orient(a, b, d)
        o3, o4 = orient(c, d, a), orient(c, d, b)
        return o1 != o2 and o3 != o4

    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            c, d = points[j], points[(j + 1) % n]
            if seg_intersect(a, b, c, d):
                return False

    return True


def simplify_outline(points, tolerance_m=0.002):
    """
    Ramer-Douglas-Peucker on a closed outline.

    A hand-traced outline carries far more points than the shape needs, and the config
    caps a cutout at 32. Dropping points that sit within `tolerance_m` of the line they
    lie on keeps the shape while making room. 2 mm is well under the ~2 cm alignment
    budget, so simplification cannot be what makes a cutout visibly wrong.
    """
    if len(points) <= 3:
        return list(points)

    pts = [np.asarray(p, float) for p in points]

    def rdp(seq):
        if len(seq) < 3:
            return list(seq)

        start, end = seq[0], seq[-1]
        line = end - start
        length = np.linalg.norm(line)

        if length < 1e-12:
            dists = [np.linalg.norm(p - start) for p in seq]
        else:
            unit = line / length
            # 2-D cross product written out: numpy 2 removed the 2-vector form of
            # np.cross, and this is the perpendicular distance from the chord.
            dists = [abs(unit[0] * (p - start)[1] - unit[1] * (p - start)[0]) for p in seq]

        idx = int(np.argmax(dists))

        if dists[idx] <= tolerance_m:
            return [start, end]

        return rdp(seq[:idx + 1])[:-1] + rdp(seq[idx:])

    # Split the closed loop at the two most distant points so neither is discarded.
    arr = np.array(pts)
    d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    i, j = min(i, j), max(i, j)

    first = rdp(pts[i:j + 1])
    second = rdp(pts[j:] + pts[:i + 1])

    out = first[:-1] + second[:-1]
    return [(float(p[0]), float(p[1])) for p in out]
