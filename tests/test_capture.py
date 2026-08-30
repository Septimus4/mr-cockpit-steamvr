"""
Tests for captures, including an END-TO-END test of the whole tracing pipeline.

The end-to-end test is the important one. It renders a known outline on a known plane
into a synthetic camera frame, clicks its corners, and checks the recovered outline
matches what went in - exercising intrinsics, distortion, the camera axis flip, the pose,
the plane basis, simplification and the config format together, with no hardware.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracing.capture import Capture, hmd_matrix_to_numpy, synthetic_capture
from tracing.config_io import format_points, parse_points
from tracing.geometry import (
    Camera, Plane, backproject_outline, camera_to_world_from_hmd, is_simple_polygon,
    point_to_outline_distance, polygon_signed_area, pose_to_matrix, simplify_outline,
)

K_LEFT = np.array([[1072.26851867, 0.0, 788.49299729],
                   [0.0, 1072.31519651, 614.54444602],
                   [0.0, 0.0, 1.0]])
D_LEFT = np.array([0.08313216691950971, -0.10744697901181298,
                   -0.00016821021003468, 0.00025331486744491, 0.0])


class TestHmdMatrix(unittest.TestCase):

    def test_identity(self):
        m34 = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))
        np.testing.assert_allclose(hmd_matrix_to_numpy(m34), np.eye(4), atol=1e-12)

    def test_translation_is_the_last_column(self):
        m34 = ((1, 0, 0, 0.5), (0, 1, 0, 1.2), (0, 0, 1, -0.3))
        m = hmd_matrix_to_numpy(m34)
        np.testing.assert_allclose(m[:3, 3], [0.5, 1.2, -0.3], atol=1e-12)

    def test_no_axis_flip_is_applied(self):
        """
        OpenVR and OpenXR stage space share a convention, so this conversion must NOT
        flip anything. The camera's Y-down flip belongs in camera_to_world_from_hmd, and
        applying it twice would mirror every traced outline.
        """
        m34 = ((0, 0, 1, 0), (0, 1, 0, 0), (-1, 0, 0, 0))
        m = hmd_matrix_to_numpy(m34)
        np.testing.assert_allclose(m[:3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)

    def test_bottom_row_is_homogeneous(self):
        m = hmd_matrix_to_numpy(((1, 0, 0, 1), (0, 1, 0, 2), (0, 0, 1, 3)))
        np.testing.assert_allclose(m[3], [0, 0, 0, 1], atol=1e-12)


class TestCaptureIO(unittest.TestCase):

    def setUp(self):
        cam = Camera(K_LEFT, D_LEFT, camera_to_world_from_hmd(np.eye(4), (0, 0, 0)))
        plane = Plane((0, 0, -0.7), (0, 0, 0))
        self.capture, _ = synthetic_capture(
            cam, plane, [(-0.1, -0.07), (0.1, -0.07), (0.1, 0.07), (-0.1, 0.07)])

    def test_save_load_round_trip(self):
        fd, path = tempfile.mkstemp(suffix=".npz")
        os.close(fd)
        try:
            self.capture.save(path)
            loaded = Capture.load(path)

            np.testing.assert_array_equal(loaded.image, self.capture.image)
            np.testing.assert_allclose(loaded.K, self.capture.K)
            np.testing.assert_allclose(loaded.dist, self.capture.dist)
            np.testing.assert_allclose(loaded.camera_to_world, self.capture.camera_to_world)
        finally:
            os.unlink(path)

    def test_camera_size_matches_image(self):
        self.assertEqual(self.capture.camera().image_size, self.capture.size)


class TestEndToEndTracing(unittest.TestCase):
    """
    The full pipeline, with no hardware: known outline -> synthetic frame -> clicks ->
    back-projection -> simplify -> config format -> parse back -> compare.
    """

    def _pipeline(self, cam, plane, outline, tolerance=1e-3):
        capture, pixels = synthetic_capture(cam, plane, outline)

        for px, py in pixels:
            self.assertTrue(capture.camera().in_frame(px, py),
                            f"outline point projects to {px},{py}, outside the frame")

        recovered, dropped = backproject_outline(capture.camera(), plane, pixels)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(recovered), len(outline))

        value = format_points(recovered)
        final = parse_points(value)

        self.assertEqual(len(final), len(outline))
        for (ax, ay), (bx, by) in zip(outline, final):
            self.assertAlmostEqual(ax, bx, delta=tolerance)
            self.assertAlmostEqual(ay, by, delta=tolerance)

        return final

    def test_rectangle_frontal(self):
        cam = Camera(K_LEFT, D_LEFT, camera_to_world_from_hmd(np.eye(4), (0, 0, 0)))
        plane = Plane((0, 0, -0.7), (0, 0, 0))
        self._pipeline(cam, plane,
                       [(-0.10, -0.07), (0.10, -0.07), (0.10, 0.07), (-0.10, 0.07)])

    def test_concave_outline_on_a_tilted_panel(self):
        """An L-shaped panel on a MIP unit tilted back and canted inward."""
        hmd = pose_to_matrix((0.0, 1.15, 0.0), (-18.0, 8.0, 0.0))
        cam = Camera(K_LEFT, D_LEFT,
                     camera_to_world_from_hmd(hmd, (-0.031, -0.047, -0.138)))
        plane = Plane((0.05, 1.05, -0.62), (-28.0, 12.0, 0.0))

        outline = [(-0.09, -0.06), (0.09, -0.06), (0.09, 0.00),
                   (0.02, 0.00), (0.02, 0.06), (-0.09, 0.06)]

        final = self._pipeline(cam, plane, outline)
        self.assertTrue(is_simple_polygon(final))

    def test_winding_is_preserved(self):
        """
        Winding must survive the round trip. Ear clipping handles either, but a flip here
        would mean the Python and C++ disagreed about the outline's orientation.
        """
        cam = Camera(K_LEFT, D_LEFT, camera_to_world_from_hmd(np.eye(4), (0, 0, 0)))
        plane = Plane((0, 0, -0.7), (0, 0, 0))

        ccw = [(-0.10, -0.07), (0.10, -0.07), (0.10, 0.07), (-0.10, 0.07)]
        self.assertGreater(polygon_signed_area(ccw), 0)

        final = self._pipeline(cam, plane, ccw)
        self.assertGreater(polygon_signed_area(final), 0)

    def test_dense_trace_simplifies_within_the_config_cap(self):
        """
        A hand trace produces far more clicks than the 32-point cap. Simplification must
        bring it under while keeping the shape.
        """
        cam = Camera(K_LEFT, D_LEFT, camera_to_world_from_hmd(np.eye(4), (0, 0, 0)))
        plane = Plane((0, 0, -0.7), (0, 0, 0))

        corners = [(-0.12, -0.08), (0.12, -0.08), (0.12, 0.08), (-0.12, 0.08)]
        dense = []
        for i in range(4):
            a = np.array(corners[i], float)
            b = np.array(corners[(i + 1) % 4], float)
            for t in np.linspace(0, 1, 25, endpoint=False):
                dense.append(tuple(a + (b - a) * t))

        capture, pixels = synthetic_capture(cam, plane, dense)
        recovered, dropped = backproject_outline(capture.camera(), plane, pixels)
        self.assertEqual(dropped, 0)

        simplified = simplify_outline(recovered, tolerance_m=0.002)
        self.assertLessEqual(len(simplified), 32)
        self.assertGreaterEqual(len(simplified), 4)

        # Every original click stays close to the simplified OUTLINE. Measured against
        # the edges, not the vertices: a point mid-edge is far from both endpoints while
        # lying exactly on the edge.
        for p in recovered:
            self.assertLess(point_to_outline_distance(p, simplified), 0.005)

    def test_accuracy_is_far_inside_the_alignment_budget(self):
        """
        The plan allows ~2 cm of out-of-plane deviation before alignment suffers. Tracing
        error must be far below that, or the tool itself becomes the limiting factor.
        """
        hmd = pose_to_matrix((0.1, 1.2, 0.05), (-20.0, 15.0, 2.0))
        cam = Camera(K_LEFT, D_LEFT,
                     camera_to_world_from_hmd(hmd, (-0.031, -0.047, -0.138)))
        plane = Plane((0.0, 1.05, -0.60), (-25.0, 10.0, 0.0))

        outline = [(-0.08, -0.05), (0.08, -0.05), (0.08, 0.05), (-0.08, 0.05)]
        capture, pixels = synthetic_capture(cam, plane, outline)

        # a click is only accurate to about a pixel
        rng = np.random.default_rng(12345)
        jittered = [(px + rng.uniform(-1, 1), py + rng.uniform(-1, 1)) for px, py in pixels]

        recovered, _ = backproject_outline(capture.camera(), plane, jittered)

        worst = max(np.hypot(a[0] - b[0], a[1] - b[1])
                    for a, b in zip(outline, recovered))
        self.assertLess(worst, 0.002, f"one pixel of click error moved a point {worst*1000:.2f} mm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
