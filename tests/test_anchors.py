"""
Tests for the anchor solver.

The property that matters: markers at KNOWN poses, seen by a virtual camera at KNOWN
poses, must be recovered to where they started. That exercises the marker frame
convention, PnP, the camera axis flip, pose composition and the averaging together.

Everything here is deterministic - no camera, no headset, no cockpit.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from anchors.solver import (
    average_rotations, constellation_conditioning, is_coplanar, marker_object_points,
    solve_marker_pose, solve_markers,
)
from anchors.detect import (
    detect_markers, is_diagnostic_id, make_detector, plate_size_overrides, summarise,
)
from anchors.synthetic import (
    arc_of_head_poses, marker_pose, observe, plate_markers, render_frame,
)
from tracing.geometry import Camera, camera_to_world_from_hmd
from tracing.geometry import euler_xyz_to_matrix, pose_to_matrix


def angle_between(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1))))


class TestMarkerObjectPoints(unittest.TestCase):
    """
    Pins the corner convention. Getting it wrong yields a marker rotated by a multiple
    of 90 degrees, which reads as a mounting mistake rather than a code one.
    """

    def test_size_and_centring(self):
        pts = marker_object_points(30.0)
        self.assertEqual(pts.shape, (4, 3))
        np.testing.assert_allclose(pts.mean(axis=0), [0, 0, 0], atol=1e-12)

        edge = np.linalg.norm(pts[1] - pts[0])
        self.assertAlmostEqual(edge, 0.030, places=9)

    def test_detector_order_is_tl_tr_br_bl(self):
        """Y is UP in the marker frame, matching OpenCV's ArUco convention."""
        pts = marker_object_points(30.0)
        self.assertGreater(pts[0][1], 0)       # top-left
        self.assertGreater(pts[1][1], 0)       # top-right
        self.assertLess(pts[2][1], 0)          # bottom-right
        self.assertLess(pts[3][1], 0)          # bottom-left
        self.assertLess(pts[0][0], 0)          # left
        self.assertGreater(pts[1][0], 0)       # right

    def test_markers_are_flat(self):
        for p in marker_object_points(50.0):
            self.assertAlmostEqual(p[2], 0.0, places=12)


class TestSingleMarker(unittest.TestCase):

    def _round_trip(self, pose, head_pose, size_mm=32.8, pos_tol_mm=1.0, ang_tol_deg=1.0):
        obs, cams = observe({0: pose}, [head_pose], {0: size_mm})
        self.assertEqual(len(obs), 1, "the marker was not visible in the fixture")

        solved = solve_marker_pose(cams[0], obs[0].corners_px, size_mm)
        self.assertIsNotNone(solved)
        got, err = solved

        pos_err = np.linalg.norm(got[:3, 3] - pose[:3, 3]) * 1000.0
        ang_err = angle_between(got[:3, :3], pose[:3, :3])

        self.assertLess(pos_err, pos_tol_mm, f"position off by {pos_err:.2f} mm")
        self.assertLess(ang_err, ang_tol_deg, f"orientation off by {ang_err:.2f} deg")
        self.assertLess(err, 1.0, f"reprojection {err:.2f} px")

    def test_facing_the_camera_gives_an_exact_POSITION(self):
        """
        Dead-on is the worst case for ORIENTATION and the best for position. The angle
        tolerance is loose on purpose - see test_frontal_orientation_is_ambiguous.
        """
        self._round_trip(marker_pose((0.0, 1.15, -0.7)),
                         pose_to_matrix((0.0, 1.15, 0.0), (0.0, 0.0, 0.0)),
                         ang_tol_deg=20.0)

    def test_perpendicular_marker_is_ambiguous_but_tilt_resolves_it(self):
        """
        Documents a real limit, and its bound.

        A planar marker EXACTLY perpendicular to the view admits two poses that reproject
        within a fifth of a pixel of each other, so a single view cannot pin its tilt.
        Measured on hardware: 27-47% of frames sat on the far side of a 15 degree jump.

        The useful half is that a little tilt resolves it completely. Every real cockpit
        panel is tilted, so the case that breaks is the one that does not occur - which is
        why the plate tests below solve exactly from every view.
        """
        head = pose_to_matrix((0.0, 1.15, 0.0), (0.0, 0.0, 0.0))

        def orientation_error(marker_euler):
            pose = marker_pose((0.0, 1.15, -0.7), marker_euler)
            obs, cams = observe({0: pose}, [head], {0: 32.8})
            self.assertEqual(len(obs), 1, "fixture did not see the marker")
            got, _ = solve_marker_pose(cams[0], obs[0].corners_px, 32.8)
            return angle_between(got[:3, :3], pose[:3, :3])

        perpendicular = orientation_error((0.0, 0.0, 0.0))
        tilted = orientation_error((-20.0, 0.0, 0.0))

        self.assertGreater(perpendicular, 2.0,
                           "a perpendicular marker should be ambiguous - if this is now "
                           "exact, the model or the solver has changed materially")
        self.assertLess(tilted, 1.0,
                        f"a 20 degree tilt should resolve the ambiguity, got {tilted:.2f} deg")

    def test_tilted_marker(self):
        self._round_trip(marker_pose((0.0, 1.05, -0.65), (-25.0, 10.0, 0.0)),
                         pose_to_matrix((0.0, 1.15, 0.0), (-10.0, 0.0, 0.0)))

    def test_off_centre_in_the_frame(self):
        self._round_trip(marker_pose((0.12, 1.05, -0.6), (-20.0, -15.0, 0.0)),
                         pose_to_matrix((0.0, 1.15, 0.0), (-8.0, 0.0, 0.0)))

    def test_wrong_size_puts_the_error_into_RANGE(self):
        """
        The reason size is encoded in the marker id and cannot be configured. Solving a
        32.8 mm marker as 50 mm does not fail - it moves the marker further away, by the
        ratio of the sizes, and looks entirely plausible.
        """
        true_pose = marker_pose((0.0, 1.15, -0.7))
        head = pose_to_matrix((0.0, 1.15, 0.0), (0.0, 0.0, 0.0))
        obs, cams = observe({0: true_pose}, [head], {0: 32.8})

        wrong, _ = solve_marker_pose(cams[0], obs[0].corners_px, 50.0)

        true_range = np.linalg.norm(true_pose[:3, 3] - cams[0].position)
        wrong_range = np.linalg.norm(wrong[:3, 3] - cams[0].position)

        self.assertAlmostEqual(wrong_range / true_range, 50.0 / 32.8, delta=0.02)


class TestConstellation(unittest.TestCase):

    def _plate(self):
        return plate_markers((0.0, 1.05, -0.62), (-20.0, 0.0, 0.0), 69.0,
                             [0, 1, 2, 3], 32.8)

    def test_recovers_a_plate_from_several_views(self):
        """The WinCtrl panel geometry, measured on hardware: 69 mm spread, 32.8 mm."""
        markers = self._plate()
        heads = arc_of_head_poses((0.0, 0.0, -0.62), 0.62, 5)
        obs, cams = observe(markers, heads, {i: 32.8 for i in markers})

        self.assertGreater(len(obs), 8, "fixture saw too few markers")

        solutions = solve_markers(obs, cams)
        self.assertEqual(len(solutions), 4)

        for marker_id, sol in solutions.items():
            pos_err = np.linalg.norm(sol.position - markers[marker_id][:3, 3]) * 1000.0
            ang_err = angle_between(sol.pose[:3, :3], markers[marker_id][:3, :3])

            self.assertLess(pos_err, 2.0, f"marker {marker_id} off by {pos_err:.2f} mm")
            self.assertLess(ang_err, 1.0, f"marker {marker_id} off by {ang_err:.2f} deg")

    def test_pixel_noise_averages_down_over_views(self):
        """
        More viewpoints must help. If they did not, there would be no reason to sweep the
        head during calibration.
        """
        markers = self._plate()
        sizes = {i: 32.8 for i in markers}

        def worst_error(n_views):
            heads = arc_of_head_poses((0.0, 0.0, -0.62), 0.62, n_views)
            obs, cams = observe(markers, heads, sizes, pixel_noise=0.5, seed=7)
            sols = solve_markers(obs, cams)
            return max(np.linalg.norm(s.position - markers[i][:3, 3]) * 1000.0
                       for i, s in sols.items())

        self.assertLess(worst_error(9), worst_error(1) + 1e-9)

    def test_reports_spread_as_confidence(self):
        markers = self._plate()
        heads = arc_of_head_poses((0.0, 0.0, -0.62), 0.62, 5)
        obs, cams = observe(markers, heads, {i: 32.8 for i in markers}, pixel_noise=0.3, seed=3)

        for sol in solve_markers(obs, cams).values():
            self.assertGreater(sol.observations, 1)
            self.assertGreaterEqual(sol.position_spread_mm, 0.0)
            self.assertLess(sol.reprojection_px, 2.0)


class TestConditioning(unittest.TestCase):
    """
    The check that refuses a layout which would amplify systematic error. Measured on
    hardware: a 3.6:1 strip swung 9x in accuracy with position in frame; a 1:1 spread
    swung 2x.
    """

    class _Fake:
        def __init__(self, p):
            self.position = np.asarray(p, float)

    def test_square_is_good(self):
        pts = [(0, 0, 0), (0.07, 0, 0), (0.07, 0.07, 0), (0, 0.07, 0)]
        aspect, verdict, extent = constellation_conditioning([self._Fake(p) for p in pts])
        self.assertLess(aspect, 2.0)
        self.assertEqual(verdict, "GOOD")
        self.assertGreater(extent, 40.0)

    def test_strip_is_refused(self):
        """A row of markers - the layout the hardware measurements condemned."""
        pts = [(0, 0, 0), (0.04, 0, 0), (0.08, 0, 0), (0.12, 0.005, 0)]
        aspect, verdict, _ = constellation_conditioning([self._Fake(p) for p in pts])
        self.assertGreater(aspect, 3.0)
        self.assertEqual(verdict, "TOO COLLINEAR")

    def test_too_few_markers(self):
        aspect, verdict, _ = constellation_conditioning([self._Fake((0, 0, 0))])
        self.assertEqual(verdict, "TOO FEW")

    def test_perfectly_collinear(self):
        pts = [(0, 0, 0), (0.05, 0, 0), (0.10, 0, 0)]
        _, verdict, _ = constellation_conditioning([self._Fake(p) for p in pts])
        self.assertEqual(verdict, "COLLINEAR")

    def test_coplanar_is_detected(self):
        """
        Coplanar sets cannot resolve the planar tilt ambiguity - confirmed on hardware,
        where a coplanar constellation still flipped. Worth reporting rather than
        discovering as instability.
        """
        flat = [self._Fake(p) for p in
                [(0, 0, 0), (0.07, 0, 0), (0.07, 0.07, 0), (0, 0.07, 0)]]
        coplanar, out_of_plane = is_coplanar(flat)
        self.assertTrue(coplanar)
        self.assertLess(out_of_plane, 1.0)

        deep = [self._Fake(p) for p in
                [(0, 0, 0), (0.07, 0, 0.05), (0.07, 0.07, -0.04), (0, 0.07, 0.06)]]
        coplanar, out_of_plane = is_coplanar(deep)
        self.assertFalse(coplanar)
        self.assertGreater(out_of_plane, 10.0)


class TestRotationAveraging(unittest.TestCase):

    def test_identical_rotations(self):
        r = euler_xyz_to_matrix(10, -20, 30)
        np.testing.assert_allclose(average_rotations([r, r, r]), r, atol=1e-9)

    def test_result_is_a_rotation(self):
        mats = [euler_xyz_to_matrix(10 + i, -20, 30) for i in range(5)]
        avg = average_rotations(mats)
        np.testing.assert_allclose(avg @ avg.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(avg), 1.0, places=9)

    def test_a_flipped_minority_does_not_drag_the_result(self):
        """
        The planar ambiguity flips a minority of solves by a large angle. A plain mean
        would be pulled between the two clusters and land at neither.
        """
        good = euler_xyz_to_matrix(5, 0, 0)
        flipped = euler_xyz_to_matrix(5, 60, 0)
        mats = [good] * 5 + [flipped] * 2

        avg = average_rotations(mats)
        self.assertLess(angle_between(avg, good), 3.0)


K_LEFT = np.array([[1072.26851867, 0.0, 788.49299729],
                   [0.0, 1072.31519651, 614.54444602],
                   [0.0, 0.0, 1.0]])
D_LEFT = np.array([0.08313216691950971, -0.10744697901181298,
                   -0.00016821021003468, 0.00025331486744491, 0.0])


def camera_at(head_pose):
    return Camera(K_LEFT, D_LEFT,
                  camera_to_world_from_hmd(head_pose, (-0.031, -0.047, -0.138)))


class TestDetectionEndToEnd(unittest.TestCase):
    """
    Renders REAL ArUco markers into a synthetic frame, runs the REAL detector over it,
    and solves. Projecting corners by hand tests the solver but assumes the detector
    returns them in the order we expect - and a wrong order does not fail, it rotates
    every marker by a multiple of 90 degrees, which reads as a mounting error.
    """

    def _scene(self):
        markers = plate_markers((0.0, 1.05, -0.62), (-20.0, 0.0, 0.0), 69.0,
                                [0, 1, 2, 3], 32.8)
        head = arc_of_head_poses((0.0, 0.0, -0.62), 0.62, 1)[0]
        return markers, camera_at(head)

    def test_detector_finds_every_marker(self):
        markers, cam = self._scene()
        frame, drawn = render_frame(cam, markers, {i: 32.8 for i in markers})

        self.assertEqual(sorted(drawn), [0, 1, 2, 3], "the fixture did not draw them all")

        obs, rejected = detect_markers(frame, cam)
        self.assertEqual(sorted(o.marker_id for o in obs), [0, 1, 2, 3])
        self.assertEqual(rejected, {})

    def test_sizes_come_from_the_id_not_the_caller(self):
        """ids 0-3 are the 30 mm sticker class, so 22.4 mm - regardless of what was drawn."""
        markers, cam = self._scene()
        frame, _ = render_frame(cam, markers, {i: 32.8 for i in markers})

        obs, _ = detect_markers(frame, cam)
        for o in obs:
            self.assertAlmostEqual(o.size_mm, 22.4, places=3)

    def test_poses_recovered_through_the_real_detector(self):
        """
        The end-to-end property. Sizes must match what was drawn for the poses to come
        back, so this passes the true size explicitly rather than taking it from the id.
        """
        markers, cam = self._scene()
        frame, _ = render_frame(cam, markers, {i: 32.8 for i in markers})

        detector = make_detector()
        obs, _ = detect_markers(frame, cam, detector=detector)

        for o in obs:
            o.size_mm = 32.8            # what the fixture actually drew

        solutions = solve_markers(obs, {0: cam})
        self.assertEqual(len(solutions), 4)

        for marker_id, sol in solutions.items():
            pos_err = np.linalg.norm(sol.position - markers[marker_id][:3, 3]) * 1000.0
            ang_err = angle_between(sol.pose[:3, :3], markers[marker_id][:3, :3])

            # Looser than the projected-corner tests: the detector works from rendered
            # pixels, so it carries rasterisation error the ideal projection does not.
            self.assertLess(pos_err, 6.0, f"marker {marker_id} off by {pos_err:.2f} mm")
            self.assertLess(ang_err, 6.0, f"marker {marker_id} off by {ang_err:.2f} deg")

    def test_no_markers_is_not_an_error(self):
        _, cam = self._scene()
        blank = np.full((1200, 1600), 60, np.uint8)

        obs, rejected = detect_markers(blank, cam)
        self.assertEqual(obs, [])
        self.assertEqual(rejected, {})


class TestIdGuards(unittest.TestCase):
    """
    A marker that is SEEN but not usable must be reported, never silently dropped -
    "I stuck it on and nothing happened" is the failure this avoids.
    """

    def test_diagnostic_range(self):
        self.assertTrue(is_diagnostic_id(32))
        self.assertTrue(is_diagnostic_id(43))
        self.assertFalse(is_diagnostic_id(31))
        self.assertFalse(is_diagnostic_id(44))
        self.assertFalse(is_diagnostic_id(0))

    def test_diagnostic_markers_are_rejected_with_a_reason(self):
        """
        A test sheet in view carries ids whose size the id cannot be trusted for. Solving
        one would put a marker of unknown scale into the constellation.
        """
        head = arc_of_head_poses((0.0, 0.0, -0.62), 0.62, 1)[0]
        cam = camera_at(head)
        markers = plate_markers((0.0, 1.05, -0.62), (-20.0, 0.0, 0.0), 69.0,
                                [32, 33, 34, 35], 29.8)
        frame, _ = render_frame(cam, markers, {i: 29.8 for i in markers})

        obs, rejected = detect_markers(frame, cam)

        self.assertEqual(obs, [], "a diagnostic marker was accepted")
        self.assertTrue(rejected, "a diagnostic marker was dropped without a reason")
        for why in rejected.values():
            self.assertIn("diagnostic", why)

    def test_unallocated_id_is_rejected(self):
        head = arc_of_head_poses((0.0, 0.0, -0.62), 0.62, 1)[0]
        cam = camera_at(head)
        markers = plate_markers((0.0, 1.05, -0.62), (-20.0, 0.0, 0.0), 69.0,
                                [44, 45, 46, 47], 30.0)
        frame, _ = render_frame(cam, markers, {i: 30.0 for i in markers})

        obs, rejected = detect_markers(frame, cam)

        self.assertEqual(obs, [])
        for why in rejected.values():
            self.assertIn("size class", why)

    def test_summary_mentions_what_was_skipped(self):
        text = summarise([], {32: "diagnostic id - a test sheet is in view"})
        self.assertIn("SKIPPED 32", text)


class TestSizeOverrides(unittest.TestCase):
    """
    The id -> size map guarantees a PRINTED marker's size cannot be configured wrong. A
    DISPLAY plate breaks that guarantee - it renders whatever size fits the panel.

    This is not a small error. Measured on real cockpit data: ids 0-11 displayed at
    32.8 mm but solved as the map's 22.4 mm came out 28% too close, and because each view
    then placed the marker at the wrong distance along a DIFFERENT ray, the per-view
    estimates scattered by 38 mm instead of simply being nearer.
    """

    def _scene(self, drawn_size):
        markers = plate_markers((0.0, 1.05, -0.62), (-20.0, 0.0, 0.0), 69.0,
                                [0, 1, 2, 3], drawn_size)
        head = arc_of_head_poses((0.0, 0.0, -0.62), 0.62, 1)[0]
        cam = camera_at(head)
        frame, _ = render_frame(cam, markers, {i: drawn_size for i in markers})
        return markers, cam, frame

    def test_override_is_used_in_preference_to_the_id(self):
        _, cam, frame = self._scene(32.8)

        obs, _ = detect_markers(frame, cam, size_overrides={0: 32.8, 1: 32.8, 2: 32.8, 3: 32.8})
        self.assertTrue(obs)
        for o in obs:
            self.assertAlmostEqual(o.size_mm, 32.8, places=3)

    def test_without_the_override_the_scale_is_wrong(self):
        """Guards the regression: this is what the real capture actually did."""
        markers, cam, frame = self._scene(32.8)

        obs, _ = detect_markers(frame, cam)          # id map says 22.4 mm
        sols = solve_markers(obs, {0: cam})

        p = np.array([sols[i].position for i in (0, 1, 2, 3)])
        side = np.mean([np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]) * 1000.0

        self.assertLess(side, 60.0,
                        "without the override the plate should come out too small; if it "
                        "does not, the id map or the plate size has changed")

    def test_with_the_override_the_scale_is_right(self):
        markers, cam, frame = self._scene(32.8)

        obs, _ = detect_markers(frame, cam, size_overrides={i: 32.8 for i in range(4)})
        sols = solve_markers(obs, {0: cam})

        p = np.array([sols[i].position for i in (0, 1, 2, 3)])
        side = np.mean([np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]) * 1000.0

        self.assertAlmostEqual(side, 69.0, delta=3.0)

    def test_overrides_are_read_from_display_plates_only(self):
        """Sticker plates must NOT override - their size really does come from the id."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "plate-disp.json"), "w") as f:
                json.dump({"kind": "display", "ids": [0, 1], "marker_mm": 32.8}, f)
            with open(os.path.join(d, "plate-stick.json"), "w") as f:
                json.dump({"kind": "sticker", "ids": [4, 5], "marker_mm": 22.4}, f)

            out = plate_size_overrides(d)

        self.assertEqual(sorted(out), [0, 1])
        self.assertAlmostEqual(out[0], 32.8, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
