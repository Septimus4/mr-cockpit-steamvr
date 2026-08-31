"""
Unit tests for the tracing geometry.

The whole point is to be able to iterate on the tracing maths without a camera or a
headset. Anything that needs hardware is not tested here and is marked as such in the
docs; everything below is deterministic.

Run:  .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracing.geometry import (
    Camera, Plane, backproject_click, backproject_outline, camera_to_world_from_hmd,
    euler_xyz_to_matrix, intersect_ray_plane, is_simple_polygon, matrix_to_euler_xyz,
    polygon_signed_area,
    point_to_outline_distance, point_to_segment_distance, pose_to_matrix,
    rotation_x, rotation_y, rotation_z, simplify_outline,
)

# The real calibrated left camera, from config-final-calibrated.ini. Using the actual
# intrinsics rather than a toy pinhole means the tests exercise real distortion.
K_LEFT = np.array([[1072.26851867, 0.0, 788.49299729],
                   [0.0, 1072.31519651, 614.54444602],
                   [0.0, 0.0, 1.0]])
D_LEFT = np.array([0.08313216691950971, -0.10744697901181298,
                   -0.00016821021003468, 0.00025331486744491, 0.0])
D_ZERO = np.zeros(5)


def identity_camera(dist=D_ZERO, position=(0.0, 0.0, 0.0)):
    """Camera at `position`, looking along world -Z, Y-up world / Y-down camera."""
    return Camera(K_LEFT, dist, camera_to_world_from_hmd(
        np.eye(4), np.asarray(position, float)))


class TestRotationConvention(unittest.TestCase):
    """
    These pin down the convention shared with GetQuadToWorldTransform in C++. If the two
    ever diverge, traced outlines land rotated and nothing about the symptom points at
    the cause.
    """

    def test_identity(self):
        np.testing.assert_allclose(euler_xyz_to_matrix(0, 0, 0), np.eye(3), atol=1e-12)

    def test_composition_order_is_rz_ry_rx(self):
        rx, ry, rz = 11.0, -23.0, 47.0
        expected = rotation_z(rz) @ rotation_y(ry) @ rotation_x(rx)
        np.testing.assert_allclose(euler_xyz_to_matrix(rx, ry, rz), expected, atol=1e-12)

    def test_order_actually_matters(self):
        """Guards against a symmetric test that would pass under the wrong order."""
        a = euler_xyz_to_matrix(30, 40, 50)
        b = rotation_x(30) @ rotation_y(40) @ rotation_z(50)
        self.assertFalse(np.allclose(a, b, atol=1e-6))

    def test_ninety_degrees_about_y_maps_x_to_minus_z(self):
        v = euler_xyz_to_matrix(0, 90, 0) @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 0.0, -1.0], atol=1e-12)

    def test_rotations_are_orthonormal(self):
        m = euler_xyz_to_matrix(17, -66, 128)
        np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(m), 1.0, places=12)


class TestPlane(unittest.TestCase):

    def test_unrotated_plane_axes(self):
        p = Plane((0, 1.2, -0.7), (0, 0, 0))
        np.testing.assert_allclose(p.x_axis, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(p.y_axis, [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(p.normal, [0, 0, 1], atol=1e-12)

    def test_to_world_to_plane_round_trip(self):
        p = Plane((0.3, 1.1, -0.6), (-25.0, 12.0, 7.0))

        for u, v in [(0, 0), (0.1, -0.05), (-0.22, 0.31), (1.5, -2.0)]:
            w = p.to_world(u, v)
            uu, vv = p.to_plane(w)
            self.assertAlmostEqual(uu, u, places=9)
            self.assertAlmostEqual(vv, v, places=9)

    def test_origin_maps_to_zero(self):
        p = Plane((0.3, 1.1, -0.6), (-25.0, 12.0, 7.0))
        np.testing.assert_allclose(p.to_plane(p.position), (0.0, 0.0), atol=1e-12)

    def test_points_on_plane_have_zero_distance(self):
        p = Plane((0.0, 1.0, -0.8), (30.0, -15.0, 5.0))
        self.assertAlmostEqual(p.distance_to(p.to_world(0.2, -0.1)), 0.0, places=12)

    def test_distance_is_signed_along_normal(self):
        p = Plane((0, 0, 0), (0, 0, 0))
        self.assertAlmostEqual(p.distance_to((0, 0, 0.5)), 0.5, places=12)
        self.assertAlmostEqual(p.distance_to((0, 0, -0.5)), -0.5, places=12)


class TestRayPlane(unittest.TestCase):

    def test_straight_on_hit(self):
        p = Plane((0, 0, -1.0), (0, 0, 0))
        hit = intersect_ray_plane((0, 0, 0), (0, 0, -1), p)
        np.testing.assert_allclose(hit, [0, 0, -1], atol=1e-12)

    def test_parallel_ray_misses(self):
        p = Plane((0, 0, -1.0), (0, 0, 0))
        self.assertIsNone(intersect_ray_plane((0, 0, 0), (1, 0, 0), p))

    def test_plane_behind_ray_is_rejected(self):
        """A click that would land behind the camera must not produce a point."""
        p = Plane((0, 0, 1.0), (0, 0, 0))
        self.assertIsNone(intersect_ray_plane((0, 0, 0), (0, 0, -1), p))

    def test_oblique_hit(self):
        p = Plane((0, 0, -2.0), (0, 0, 0))
        d = np.array([1.0, 0.0, -1.0]) / np.sqrt(2)
        hit = intersect_ray_plane((0, 0, 0), d, p)
        np.testing.assert_allclose(hit, [2.0, 0.0, -2.0], atol=1e-9)


class TestCameraFrame(unittest.TestCase):

    def test_principal_point_looks_along_minus_z(self):
        """The camera's optical axis must be world -Z when the HMD is at identity."""
        cam = identity_camera()
        ray = cam.pixel_to_ray(K_LEFT[0, 2], K_LEFT[1, 2])
        np.testing.assert_allclose(ray, [0, 0, -1], atol=1e-6)

    def test_pixel_below_centre_points_down_in_world(self):
        """
        Camera Y is DOWN, world Y is UP. A pixel below the principal point must give a
        ray with negative world Y. Getting this flip wrong mirrors every traced outline.
        """
        cam = identity_camera()
        ray = cam.pixel_to_ray(K_LEFT[0, 2], K_LEFT[1, 2] + 200)
        self.assertLess(ray[1], -0.01)

    def test_pixel_right_of_centre_points_right_in_world(self):
        cam = identity_camera()
        ray = cam.pixel_to_ray(K_LEFT[0, 2] + 200, K_LEFT[1, 2])
        self.assertGreater(ray[0], 0.01)

    def test_camera_offset_is_applied_in_hmd_frame(self):
        hmd = pose_to_matrix((0.0, 1.5, 0.0), (0.0, 90.0, 0.0))
        cam = Camera(K_LEFT, D_ZERO, camera_to_world_from_hmd(hmd, (0.0, 0.0, -0.138)))
        # Facing world -X after a 90 deg yaw, so a forward offset moves the camera in -X.
        np.testing.assert_allclose(cam.position, [-0.138, 1.5, 0.0], atol=1e-9)

    def test_rays_are_unit_length(self):
        cam = identity_camera(dist=D_LEFT)
        for px, py in [(10, 10), (800, 600), (1590, 1190)]:
            self.assertAlmostEqual(np.linalg.norm(cam.pixel_to_ray(px, py)), 1.0, places=9)


class TestBackprojectionRoundTrip(unittest.TestCase):
    """
    The core property: a point on the cutout plane, projected into the image and clicked
    back, must return to where it started. This is what makes the tracing tool correct,
    and it exercises intrinsics, distortion, the axis flip, the pose and the plane at once.
    """

    def _round_trip(self, cam, plane, uv_points, places=6):
        for u, v in uv_points:
            world = plane.to_world(u, v)
            px = cam.project(world)
            self.assertIsNotNone(px, f"({u},{v}) projected behind the camera")
            self.assertTrue(cam.in_frame(*px),
                            f"({u},{v}) projects to {px}, outside the image - the test "
                            f"fixture is unrealistic, not the code")

            uv = backproject_click(cam, plane, px[0], px[1])
            self.assertIsNotNone(uv, f"({u},{v}) failed to back-project")

            self.assertAlmostEqual(uv[0], u, places=places)
            self.assertAlmostEqual(uv[1], v, places=places)

    def test_frontal_plane_no_distortion(self):
        cam = identity_camera()
        plane = Plane((0, 0, -0.7), (0, 0, 0))
        self._round_trip(cam, plane, [(0, 0), (0.05, 0.03), (-0.12, 0.08), (0.2, -0.15)])

    def test_frontal_plane_with_real_distortion(self):
        cam = identity_camera(dist=D_LEFT)
        plane = Plane((0, 0, -0.7), (0, 0, 0))
        self._round_trip(cam, plane, [(0, 0), (0.1, 0.06), (-0.18, -0.11), (0.22, 0.14)])

    def test_tilted_plane(self):
        """A panel tilted back and canted inward, as a real MIP unit is."""
        cam = identity_camera(dist=D_LEFT)
        plane = Plane((0.15, -0.10, -0.65), (-30.0, 18.0, 0.0))
        self._round_trip(cam, plane, [(0, 0), (0.06, 0.04), (-0.07, -0.05)])

    def test_offset_camera_and_moved_hmd(self):
        hmd = pose_to_matrix((0.2, 1.3, 0.1), (-8.0, 14.0, 3.0))
        cam = Camera(K_LEFT, D_LEFT,
                     camera_to_world_from_hmd(hmd, (-0.031, -0.047, -0.138)))
        plane = Plane((0.1, 1.1, -0.5), (-22.0, 10.0, -4.0))
        self._round_trip(cam, plane, [(0, 0), (0.05, 0.03), (-0.06, 0.02)], places=5)

    def test_console_sized_outline(self):
        """
        A long, steeply angled console, viewed from a head pose that is actually looking
        at it.

        An earlier version placed the console 69 degrees off-axis, far outside the
        camera's ~37 degree half-FOV, where cv2.undistortPoints diverges. The fixture was
        wrong, not the code - which is why _round_trip now asserts the point is in frame,
        so that mistake fails with a message naming the cause.
        """
        hmd = pose_to_matrix((0.0, 1.2, 0.0), (-15.0, 35.0, 0.0))
        cam = Camera(K_LEFT, D_LEFT,
                     camera_to_world_from_hmd(hmd, (-0.031, -0.047, -0.138)))
        plane = Plane((-0.30, 0.80, -0.45), (-55.0, 35.0, 10.0))
        self._round_trip(cam, plane,
                         [(0, 0), (0.10, 0.04), (-0.10, -0.04)], places=5)

    def test_click_missing_the_plane_returns_none(self):
        """Looking away from the panel must give nothing, not a bogus point."""
        cam = identity_camera()
        plane = Plane((0, 0, 1.0), (0, 0, 0))     # behind the camera
        self.assertIsNone(backproject_click(cam, plane, K_LEFT[0, 2], K_LEFT[1, 2]))

    def test_outline_reports_dropped_clicks(self):
        cam = identity_camera()
        plane = Plane((0, 0, -0.7), (0, 0, 0))
        good = cam.project(plane.to_world(0.05, 0.02))

        behind = Plane((0, 0, 1.0), (0, 0, 0))
        pts, dropped = backproject_outline(cam, behind, [good, good, good])
        self.assertEqual(pts, [])
        self.assertEqual(dropped, 3)

        pts, dropped = backproject_outline(cam, plane, [good, good])
        self.assertEqual(len(pts), 2)
        self.assertEqual(dropped, 0)


class TestPolygonHelpers(unittest.TestCase):

    def test_signed_area_sign_and_magnitude(self):
        ccw = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertAlmostEqual(polygon_signed_area(ccw), 1.0, places=12)
        self.assertAlmostEqual(polygon_signed_area(list(reversed(ccw))), -1.0, places=12)

    def test_signed_area_matches_cpp_convention(self):
        """Same formula as PolygonSignedArea in mesh.cpp, so windings agree."""
        pts = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
        self.assertAlmostEqual(polygon_signed_area(pts), 3.0, places=12)

    def test_simple_polygons_accepted(self):
        self.assertTrue(is_simple_polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))
        self.assertTrue(is_simple_polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]))

    def test_self_intersecting_polygon_rejected(self):
        """A bow tie. Ear clipping would reject it and fall back to a rectangle."""
        self.assertFalse(is_simple_polygon([(0, 0), (1, 1), (1, 0), (0, 1)]))

    def test_too_few_points_rejected(self):
        self.assertFalse(is_simple_polygon([(0, 0), (1, 1)]))


class TestDistances(unittest.TestCase):

    def test_distance_to_segment_endpoints(self):
        self.assertAlmostEqual(point_to_segment_distance((0, 0), (0, 0), (1, 0)), 0.0)
        self.assertAlmostEqual(point_to_segment_distance((1, 0), (0, 0), (1, 0)), 0.0)

    def test_perpendicular_distance(self):
        self.assertAlmostEqual(point_to_segment_distance((0.5, 2), (0, 0), (1, 0)), 2.0)

    def test_clamps_beyond_the_ends(self):
        """Distance to a SEGMENT, not to its infinite line."""
        self.assertAlmostEqual(point_to_segment_distance((3, 0), (0, 0), (1, 0)), 2.0)

    def test_degenerate_segment(self):
        self.assertAlmostEqual(point_to_segment_distance((3, 4), (0, 0), (0, 0)), 5.0)

    def test_outline_distance_uses_edges(self):
        """A point mid-edge is on the outline even though it is far from every vertex."""
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertAlmostEqual(point_to_outline_distance((5, 0), square), 0.0)

    def test_outline_distance_inside(self):
        square = [(0, 0), (10, 0), (10, 10), (0, 10)]
        self.assertAlmostEqual(point_to_outline_distance((5, 5), square), 5.0)


class TestSimplify(unittest.TestCase):

    def test_collinear_points_removed(self):
        pts = [(0, 0), (0.5, 0), (1, 0), (1, 1), (0, 1)]
        out = simplify_outline(pts, tolerance_m=0.002)
        self.assertLess(len(out), len(pts))
        self.assertGreaterEqual(len(out), 3)

    def test_corners_preserved(self):
        square = [(0, 0), (1, 0), (1, 1), (0, 1)]
        dense = []
        for i in range(4):
            a, b = np.array(square[i], float), np.array(square[(i + 1) % 4], float)
            for t in np.linspace(0, 1, 12, endpoint=False):
                dense.append(tuple(a + (b - a) * t))

        out = simplify_outline(dense, tolerance_m=0.01)
        self.assertLessEqual(len(out), 8)
        for corner in square:
            self.assertTrue(
                any(abs(p[0] - corner[0]) < 0.02 and abs(p[1] - corner[1]) < 0.02 for p in out),
                f"corner {corner} was lost")

    def test_shape_preserved_within_tolerance(self):
        """Simplification must not move the outline by more than its tolerance."""
        t = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        circle = [(float(np.cos(a) * 0.2), float(np.sin(a) * 0.2)) for a in t]

        out = simplify_outline(circle, tolerance_m=0.002)
        self.assertLess(len(out), 60)

        for p in circle:
            self.assertLess(point_to_outline_distance(p, out), 0.004)

    def test_short_outlines_untouched(self):
        pts = [(0, 0), (1, 0), (0.5, 1)]
        self.assertEqual(len(simplify_outline(pts)), 3)

    def test_fits_the_config_cap(self):
        """A hand-traced outline must reduce below MAX_QUAD_POLYGON_POINTS (32)."""
        t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
        blob = [(float((0.3 + 0.02 * np.cos(5 * a)) * np.cos(a)),
                 float((0.2 + 0.02 * np.sin(4 * a)) * np.sin(a))) for a in t]
        self.assertLessEqual(len(simplify_outline(blob, tolerance_m=0.004)), 32)


class TestEulerInverse(unittest.TestCase):
    """
    matrix_to_euler_xyz turns a solved pose back into the RotX/RotY/RotZ the config
    stores. It must invert euler_xyz_to_matrix exactly, because an error does not fail -
    it writes a rotated cutout, which reads as bad tracking rather than a conversion bug.
    """

    def test_round_trip_over_many_orientations(self):
        rng = np.random.default_rng(11)
        for _ in range(200):
            a = rng.uniform(-89.0, 89.0)       # avoid gimbal lock, tested separately
            b = rng.uniform(-89.0, 89.0)
            c = rng.uniform(-179.0, 179.0)

            m = euler_xyz_to_matrix(a, b, c)
            back = matrix_to_euler_xyz(m)

            np.testing.assert_allclose(euler_xyz_to_matrix(*back), m, atol=1e-9)

    def test_identity(self):
        np.testing.assert_allclose(matrix_to_euler_xyz(np.eye(3)), (0, 0, 0), atol=1e-9)

    def test_known_single_axis(self):
        np.testing.assert_allclose(matrix_to_euler_xyz(euler_xyz_to_matrix(30, 0, 0)),
                                   (30, 0, 0), atol=1e-9)
        np.testing.assert_allclose(matrix_to_euler_xyz(euler_xyz_to_matrix(0, -25, 0)),
                                   (0, -25, 0), atol=1e-9)
        np.testing.assert_allclose(matrix_to_euler_xyz(euler_xyz_to_matrix(0, 0, 47)),
                                   (0, 0, 47), atol=1e-9)

    def test_realistic_panel_pose(self):
        """A MIP unit tilted back and canted inward."""
        m = euler_xyz_to_matrix(-28.0, 12.0, -3.0)
        np.testing.assert_allclose(matrix_to_euler_xyz(m), (-28.0, 12.0, -3.0), atol=1e-9)

    def test_gimbal_lock_still_reproduces_the_rotation(self):
        """
        At pitch +/-90 the X and Z angles are not separable. The angles returned need not
        match what went in, but the ROTATION they describe must.
        """
        for b in (90.0, -90.0):
            m = euler_xyz_to_matrix(20.0, b, 35.0)
            np.testing.assert_allclose(euler_xyz_to_matrix(*matrix_to_euler_xyz(m)), m,
                                       atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
