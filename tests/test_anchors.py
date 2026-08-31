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

from anchors.camera_rig import (
    CAMERA_OFFSET, CAMERA_OFFSET_LEGACY, apply_offset_delta, offset_delta,
)
from anchors.place import (
    _segments_cross, banded_outline, bridge_hole, cover_all, cutout_extent,
    fit_plane_frame, fit_rigid, flattening_cost_mm, group_by_plate,
    orient_frame_towards, outline_with_holes, panel_rects_in_plane,
    place_from_markers, place_from_plate, plate_local_points, rect_to_points,
    shaped_cutout,
)
from anchors.solver import (
    Observation, average_rotations, constellation_conditioning, is_coplanar,
    marker_object_points, solve_marker_pose, solve_markers,
)
from anchors.detect import (
    detect_markers, is_diagnostic_id, make_detector, plate_size_overrides, summarise,
)
from anchors.synthetic import (
    arc_of_head_poses, marker_pose, observe, plate_markers, render_frame,
)
from tracing.geometry import Camera, camera_to_world_from_hmd
from tracing.geometry import euler_xyz_to_matrix, matrix_to_euler_xyz, pose_to_matrix


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


class TestCornerRefinement(unittest.TestCase):
    """
    OpenCV defaults to CORNER_REFINE_NONE, which locates corners to about a pixel. Range
    is inferred from a marker's apparent size, so a pixel of error on a 70-pixel marker
    is 1.4% of range - about 9 mm at half a metre.

    That was the residual measured on real cockpit data once marker size, time skew,
    camera rotation, camera offset and pose flips had all been ruled out, and it is worth
    a test because the default is silent: detection succeeds either way, just less
    precisely.
    """

    def test_subpixel_refinement_is_on_by_default(self):
        import cv2
        det = make_detector()
        params = det.getDetectorParameters()
        self.assertEqual(params.cornerRefinementMethod, cv2.aruco.CORNER_REFINE_SUBPIX)

    def test_refinement_measurably_improves_accuracy(self):
        """Not just that the flag is set - that it actually buys something."""
        markers = plate_markers((0.0, 1.05, -0.62), (-20.0, 0.0, 0.0), 69.0,
                                [0, 1, 2, 3], 32.8)
        sizes = {i: 32.8 for i in markers}

        def mean_error(refine):
            det = make_detector(refine=refine)
            errs = []
            for head in arc_of_head_poses((0.0, 0.0, -0.62), 0.62, 5):
                cam = camera_at(head)
                frame, _ = render_frame(cam, markers, sizes)
                obs, _ = detect_markers(frame, cam, detector=det, size_overrides=sizes)
                for o in obs:
                    r = solve_marker_pose(cam, o.corners_px, o.size_mm)
                    if r is not None:
                        errs.append(np.linalg.norm(r[0][:3, 3] -
                                                   markers[o.marker_id][:3, 3]) * 1000.0)
            return float(np.mean(errs))

        coarse = mean_error(False)
        refined = mean_error(True)

        self.assertLess(refined, coarse * 0.7,
                        f"refinement should cut error substantially: "
                        f"{coarse:.2f} mm -> {refined:.2f} mm")


class TestPlacement(unittest.TestCase):
    """
    Turning solved markers into a cutout pose. This is where anchoring becomes the
    feature: a cutout that places itself instead of eighteen hand-typed numbers.
    """

    class _Sol:
        def __init__(self, p, spread=1.0):
            self.position = np.asarray(p, float)
            self.position_spread_mm = spread

    def _plate_solutions(self, origin=(0.0, 1.05, -0.62), euler=(-20.0, 0.0, 0.0),
                         spread_mm=69.0):
        markers = plate_markers(origin, euler, spread_mm, [0, 1, 2, 3], 32.8)
        return {i: self._Sol(m[:3, 3]) for i, m in markers.items()}, markers

    def test_recovers_the_plate_plane(self):
        sols, markers = self._plate_solutions()
        got = place_from_markers("panel", sols, [0, 1, 2, 3], viewpoint=(0.0, 1.15, 0.0))

        self.assertIsNotNone(got)

        # centre of the four markers
        expect = np.mean([m[:3, 3] for m in markers.values()], axis=0)
        np.testing.assert_allclose(got.position, expect, atol=1e-6)

        self.assertAlmostEqual(got.width, 0.069, delta=0.002)
        self.assertAlmostEqual(got.height, 0.069, delta=0.002)
        self.assertLess(got.flatness_mm, 0.01, "a flat plate should fit a plane exactly")

    def test_orientation_matches_the_plate(self):
        """
        The recovered rotation must describe the same plane the markers lie in. The Euler
        angles need not match the fixture's - the in-plane axes are arbitrary - but the
        NORMAL must.
        """
        sols, _ = self._plate_solutions(euler=(-25.0, 15.0, 0.0))
        got = place_from_markers("panel", sols, [0, 1, 2, 3], viewpoint=(0.0, 1.15, 0.0))

        expect_normal = euler_xyz_to_matrix(-25.0, 15.0, 0.0)[:, 2]
        got_normal = euler_xyz_to_matrix(*got.euler_deg)[:, 2]

        self.assertGreater(abs(float(got_normal @ expect_normal)), 0.999)

    def test_normal_faces_the_viewer(self):
        """
        SVD has no notion of which side is the front, so without this a cutout would face
        away half the time. The quad is double-sided so it would still draw - the stored
        rotation would simply be meaningless to read.
        """
        sols, _ = self._plate_solutions()
        viewpoint = np.array([0.0, 1.15, 0.0])
        got = place_from_markers("panel", sols, [0, 1, 2, 3], viewpoint=viewpoint)

        normal = euler_xyz_to_matrix(*got.euler_deg)[:, 2]
        self.assertGreater(float(normal @ (viewpoint - got.position)), 0.0)

    def test_margin_extends_the_cutout(self):
        sols, _ = self._plate_solutions()
        bare = place_from_markers("p", sols, [0, 1, 2, 3], (0.0, 1.15, 0.0))
        wide = place_from_markers("p", sols, [0, 1, 2, 3], (0.0, 1.15, 0.0), margin_mm=40.0)

        self.assertAlmostEqual(wide.width - bare.width, 0.040, places=6)
        np.testing.assert_allclose(wide.position, bare.position, atol=1e-9)

    def test_three_markers_is_enough_two_is_not(self):
        sols, _ = self._plate_solutions()

        self.assertIsNotNone(place_from_markers("p", sols, [0, 1, 2], (0.0, 1.15, 0.0)))
        self.assertIsNone(place_from_markers("p", sols, [0, 1], (0.0, 1.15, 0.0)),
                          "two markers give an arbitrary orientation, not an error")

    def test_flatness_reports_a_bent_plate(self):
        """A marker knocked off the panel plane should show up as flatness, not silently."""
        sols, _ = self._plate_solutions()
        sols[2].position = sols[2].position + np.array([0.0, 0.0, 0.02])

        got = place_from_markers("p", sols, [0, 1, 2, 3], (0.0, 1.15, 0.0))
        self.assertGreater(got.flatness_mm, 2.0)

    def test_grouping_separates_loose_markers(self):
        sols = {i: self._Sol((0, 0, 0)) for i in [0, 1, 2, 3, 12, 13]}
        plates = [{"name": "L", "ids": [0, 1, 2, 3]}]

        grouped, loose = group_by_plate(sols, plates)

        self.assertEqual(grouped, [("L", [0, 1, 2, 3])])
        self.assertEqual(loose, [12, 13],
                         "loose stickers anchor the cockpit frame, they are not a panel")

    def test_fit_plane_frame_is_right_handed(self):
        pts = [(0, 0, 0), (0.07, 0, 0), (0.07, 0.07, 0), (0, 0.07, 0)]
        _, r, flat = fit_plane_frame(pts)

        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=9)
        self.assertLess(flat, 1e-9)


PLATE_C = {
    "name": "winctrl-C",
    "kind": "display",
    "usable_mm": {"x": 0.94, "y": 41.25, "w": 118.12, "h": 117.81},
    "marker_mm": 32.812,
    "ids": [4, 5, 6, 7],
    "centres_mm": [[25.346, 65.656], [94.654, 65.656],
                   [94.654, 134.654], [25.346, 134.654]],
}


class TestPlateFit(unittest.TestCase):
    """
    Fitting a cutout to a plate whose layout is KNOWN.

    This is the better path: the size comes from the measured panel, the centre is the
    panel's centre, and the residual is a real check rather than a self-consistency
    figure. It is also what runs on William's three WinCtrl panels.
    """

    class _Sol:
        def __init__(self, p, spread=1.0):
            self.position = np.asarray(p, float)
            self.position_spread_mm = spread

    def _solutions_for(self, pose):
        """Place plate C at `pose` exactly, as a perfect solve would."""
        local = plate_local_points(PLATE_C)
        return {i: self._Sol(pose[:3, :3] @ v + pose[:3, 3]) for i, v in local.items()}

    def test_local_layout_is_centred_and_y_up(self):
        local = plate_local_points(PLATE_C)

        centre = np.mean(list(local.values()), axis=0)
        np.testing.assert_allclose(centre, [0, 0, 0], atol=1e-9)

        # Marker 4 is the TOP-left in panel coordinates (y down), so in the cutout frame
        # it must be ABOVE centre. A sign slip here mirrors the cutout.
        self.assertGreater(local[4][1], 0.0)
        self.assertLess(local[4][0], 0.0)

        span = max(v[0] for v in local.values()) - min(v[0] for v in local.values())
        self.assertAlmostEqual(span, 0.069308, places=5)

    def test_recovers_an_exact_pose(self):
        pose = pose_to_matrix((0.02, 1.031, -0.588), (-27.4, 3.1, -0.8))
        got = place_from_plate(PLATE_C, self._solutions_for(pose))

        np.testing.assert_allclose(got.position, pose[:3, 3], atol=1e-9)

        r = euler_xyz_to_matrix(*got.euler_deg)
        np.testing.assert_allclose(r, pose[:3, :3], atol=1e-7)

        self.assertLess(got.flatness_mm, 1e-6, "an exact layout must fit exactly")

    def test_size_comes_from_the_measured_panel_not_the_markers(self):
        """
        The markers span 69 mm; the panel is 118 mm. Sizing from the marker bounding box
        would produce a cutout barely half the panel.
        """
        pose = pose_to_matrix((0.0, 1.0, -0.6), (-20.0, 0.0, 0.0))
        got = place_from_plate(PLATE_C, self._solutions_for(pose))

        self.assertAlmostEqual(got.width, 0.11812, places=5)
        self.assertAlmostEqual(got.height, 0.11781, places=5)

    def test_residual_reports_a_bad_solve(self):
        """
        A marker solved 1 cm out of place must show up as residual. Without this the
        cutout would simply be placed slightly wrong, with nothing to say so.
        """
        pose = pose_to_matrix((0.0, 1.0, -0.6), (-20.0, 0.0, 0.0))
        sols = self._solutions_for(pose)
        sols[6].position = sols[6].position + np.array([0.01, 0.0, 0.0])

        got = place_from_plate(PLATE_C, sols)
        self.assertGreater(got.flatness_mm, 2.0)

    def test_survives_a_missing_marker(self):
        """One marker occluded by a hand or a throttle must not lose the panel."""
        pose = pose_to_matrix((0.1, 0.95, -0.55), (-25.0, 10.0, 2.0))
        sols = self._solutions_for(pose)
        del sols[7]

        got = place_from_plate(PLATE_C, sols)

        self.assertEqual(got.marker_ids, [4, 5, 6])
        np.testing.assert_allclose(got.position, pose[:3, 3], atol=1e-9)

    def test_two_markers_is_not_enough(self):
        pose = pose_to_matrix((0.0, 1.0, -0.6), (0.0, 0.0, 0.0))
        sols = self._solutions_for(pose)
        del sols[6], sols[7]

        self.assertIsNone(place_from_plate(PLATE_C, sols))

    def test_real_plate_files_parse(self):
        """The shipped plate JSONs must satisfy what place_from_plate needs."""
        import glob
        import json

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths = glob.glob(os.path.join(root, "PRINT-THESE", "plates", "plate-*.json"))

        display = []

        for path in paths:
            with open(path) as f:
                g = json.load(f)
            if g.get("kind") == "display":
                display.append(g)

        self.assertTrue(display, "no display plates to check")

        for g in display:
            local = plate_local_points(g)
            self.assertEqual(len(local), len(g["ids"]), g["name"])
            # Half a millimetre, not exact zero: marker centres are placed on PIXELS
            # (0.156 mm each) and stored rounded, so a perfectly centred layout still
            # lands a fraction of a pixel off. The check is for a wrong-frame bug, which
            # would be off by tens of millimetres.
            np.testing.assert_allclose(np.mean(list(local.values()), axis=0),
                                       [0, 0, 0], atol=5e-4, err_msg=g["name"])


class TestFitRigid(unittest.TestCase):

    def test_recovers_a_known_transform(self):
        src = np.array([[0.0, 0.0, 0.0], [0.07, 0.0, 0.0],
                        [0.07, 0.07, 0.0], [0.0, 0.07, 0.0]])
        r_true = euler_xyz_to_matrix(-31.0, 17.0, 5.0)
        t_true = np.array([0.2, 1.1, -0.6])

        r, t, rms = fit_rigid(src, src @ r_true.T + t_true)

        np.testing.assert_allclose(r, r_true, atol=1e-9)
        np.testing.assert_allclose(t, t_true, atol=1e-9)
        self.assertLess(rms, 1e-12)

    def test_never_returns_a_reflection(self):
        """A reflection can fit points beautifully and is not a pose."""
        rng = np.random.default_rng(11)

        for _ in range(50):
            src = rng.normal(size=(5, 3)) * 0.05
            dst = rng.normal(size=(5, 3)) * 0.05
            r, _, _ = fit_rigid(src, dst)

            self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=9)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)

    def test_does_not_absorb_scale_error(self):
        """
        A plate solved 5% too large must show as residual, not be silently swallowed.
        Scale is measured, so letting it float would hide a range error.
        """
        src = np.array([[0.0, 0.0, 0.0], [0.07, 0.0, 0.0],
                        [0.07, 0.07, 0.0], [0.0, 0.07, 0.0]])

        _, _, rms = fit_rigid(src, src * 1.05)
        self.assertGreater(rms * 1000.0, 1.0)


class TestUnitExtent(unittest.TestCase):
    """
    The cutout must cover the BUTTONS, not the screen.

    The markers can only measure the screen, because that is what draws them. For the
    WinCtrl MFDs that is 118 x 118 mm inside a 167 x 185 mm unit - less than half the
    area that actually matters. Sizing to the screen is the wrong answer by construction.
    """

    class _Sol:
        def __init__(self, p, spread=1.0):
            self.position = np.asarray(p, float)
            self.position_spread_mm = spread

    def _unit_plate(self, **unit):
        plate = dict(PLATE_C)
        plate["unit_mm"] = dict({"w": 167.0, "h": 185.0}, **unit)
        return plate

    def _solutions_for(self, plate, pose):
        local = plate_local_points(plate)
        return {i: self._Sol(pose[:3, :3] @ v + pose[:3, 3]) for i, v in local.items()}

    def test_falls_back_to_the_screen_when_no_unit_is_declared(self):
        w, h, dx, dy = cutout_extent(PLATE_C)

        self.assertAlmostEqual(w, 118.12, places=3)
        self.assertAlmostEqual(h, 117.81, places=3)
        self.assertEqual((dx, dy), (0.0, 0.0))

    def test_unit_extent_covers_the_buttons(self):
        plate = self._unit_plate()
        pose = pose_to_matrix((0.0, 0.9, -0.5), (-11.0, 0.0, 1.0))

        got = place_from_plate(plate, self._solutions_for(plate, pose))

        self.assertAlmostEqual(got.width, 0.167, places=5)
        self.assertAlmostEqual(got.height, 0.185, places=5)

        # More than double the screen's area - the buttons are most of the unit.
        screen = 0.11812 * 0.11781
        self.assertGreater(got.width * got.height / screen, 2.0)

    def test_offset_moves_the_cutout_in_its_own_plane(self):
        """
        A screen aperture that is not centred in its housing needs dx/dy, and the offset
        lives in the CUTOUT's plane - applying it in world axes would slide the cutout off
        the panel as soon as the panel is tilted, which every cockpit panel is.
        """
        plate = self._unit_plate(dy=20.0)
        pose = pose_to_matrix((0.0, 0.9, -0.5), (-40.0, 0.0, 0.0))

        centred = place_from_plate(self._unit_plate(), self._solutions_for(plate, pose))
        offset = place_from_plate(plate, self._solutions_for(plate, pose))

        moved = offset.position - centred.position

        self.assertAlmostEqual(float(np.linalg.norm(moved)), 0.020, places=6)

        # It moved along the panel's own up axis, not the world's.
        up = pose[:3, :3] @ np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(float(moved @ up), 0.020, places=6)
        self.assertLess(abs(moved[1]), 0.020, "a tilted panel's up is not the world's up")

    def test_margin_still_applies_on_top_of_the_unit(self):
        plate = self._unit_plate()
        pose = pose_to_matrix((0.0, 0.9, -0.5), (0.0, 0.0, 0.0))
        sols = self._solutions_for(plate, pose)

        got = place_from_plate(plate, sols, margin_mm=30.0)

        self.assertAlmostEqual(got.width, 0.197, places=5)
        self.assertAlmostEqual(got.height, 0.215, places=5)

    def test_shipped_plates_declare_their_unit(self):
        """The WinCtrl plates must size to the MFD housing, or the buttons stay hidden."""
        import glob
        import json

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        for path in glob.glob(os.path.join(root, "PRINT-THESE", "plates",
                                           "plate-winctrl-*.json")):
            with open(path) as f:
                g = json.load(f)

            w, h, _, _ = cutout_extent(g)

            self.assertGreater(w, g["usable_mm"]["w"], g["name"])
            self.assertGreater(h, g["usable_mm"]["h"], g["name"])


class TestCoverAll(unittest.TestCase):
    """
    One cutout over the whole assembly. The centre console carries no markers at all, so
    this is the only way to reach it before stickers arrive.
    """

    class _Sol:
        def __init__(self, p, spread=1.0):
            self.position = np.asarray(p, float)
            self.position_spread_mm = spread

    def _three_panels(self, tilts=(-11.0, -8.0, -26.0)):
        sols = {}
        base = 0

        for n, (tilt, origin) in enumerate(zip(tilts, [(-0.16, 0.89, -0.49),
                                                       (0.21, 0.90, -0.48),
                                                       (0.03, 0.69, -0.45)])):
            pose = pose_to_matrix(origin, (tilt, 0.0, 0.0))
            for i, v in plate_local_points(PLATE_C).items():
                sols[base + (i - 4)] = self._Sol(pose[:3, :3] @ v + pose[:3, 3])
            base += 4

        return sols

    def test_spans_everything_and_prices_the_flattening(self):
        sols = self._three_panels()
        placed = [object()]                       # only its count matters here

        got = cover_all(placed, sols, 530.0, 430.0, viewpoint=(0.0, 1.1, -0.2))

        self.assertIsNotNone(got)
        self.assertAlmostEqual(got.width, 0.530, places=5)
        self.assertAlmostEqual(got.height, 0.430, places=5)
        self.assertEqual(len(got.marker_ids), 12)

        # Panels at different tilts cannot be exactly coplanar, and the number must be
        # reported rather than assumed away.
        self.assertGreater(got.flatness_mm, 1.0)
        self.assertLess(got.flatness_mm, 60.0)

    def test_coplanar_panels_cost_nothing(self):
        got = cover_all([object()], self._three_panels(tilts=(-11.0, -11.0, -11.0)),
                        530.0, 430.0, viewpoint=(0.0, 1.1, -0.2))

        self.assertLess(got.flatness_mm, 30.0,
                        "panels at one tilt should flatten cheaply")

    def test_normal_faces_the_viewer(self):
        viewpoint = np.array([0.0, 1.1, -0.2])
        got = cover_all([object()], self._three_panels(), 530.0, 430.0, viewpoint)

        normal = euler_xyz_to_matrix(*got.euler_deg)[:, 2]
        self.assertGreater(float(normal @ (viewpoint - got.position)), 0.0)

    def test_needs_three_markers(self):
        sols = {0: self._Sol((0, 0, 0)), 1: self._Sol((0.1, 0, 0))}
        self.assertIsNone(cover_all([object()], sols, 530.0, 430.0, (0, 1.1, 0)))

    def test_flattening_cost_follows_the_depth_relation(self):
        """
        shift = baseline x deviation / distance^2. The same relation that governs the
        whole project's depth budget - halving the distance quadruples the error.
        """
        near = flattening_cost_mm(24.0, 0.4)
        far = flattening_cost_mm(24.0, 0.8)

        self.assertAlmostEqual(near / far, 4.0, places=6)
        self.assertAlmostEqual(flattening_cost_mm(24.0, 0.4),
                               0.15 * 0.024 / 0.16 * 1000.0, places=6)
        self.assertEqual(flattening_cost_mm(24.0, 0.0), 0.0)


def _polygon_area(points):
    n = len(points)
    return abs(sum(points[i][0] * points[(i + 1) % n][1] -
                   points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0


class TestBandedOutline(unittest.TestCase):
    """
    A pit is not a rectangle. Three panels across the top with one below the centre is a
    T, and a bounding box around that spends a third of its area on cockpit side wall -
    passthrough there covers the game instead of revealing a control.
    """

    TOP = [(-0.25, -0.09, 0.02, 0.20), (-0.08, 0.08, 0.02, 0.20),
           (0.09, 0.25, 0.02, 0.20)]
    STEM = [(-0.08, 0.08, -0.19, -0.01)]

    def test_makes_a_T(self):
        out = banded_outline(self.TOP + self.STEM)

        self.assertEqual(len(out), 8, "a two-row T is eight corners")

        xs = [p[0] for p in out]
        ys = [p[1] for p in out]
        self.assertAlmostEqual(min(xs), -0.25, places=6)
        self.assertAlmostEqual(max(xs), 0.25, places=6)
        self.assertAlmostEqual(max(ys), 0.20, places=6)
        self.assertAlmostEqual(min(ys), -0.19, places=6)

        # Only the stem reaches the bottom, so the bottom edge is the stem's width.
        bottom = sorted(p[0] for p in out if abs(p[1] - min(ys)) < 1e-9)
        self.assertAlmostEqual(bottom[1] - bottom[0], 0.16, places=6)

    def test_uses_much_less_than_the_bounding_box(self):
        out = banded_outline(self.TOP + self.STEM)
        box = (0.50) * (0.39)

        self.assertLess(_polygon_area(out) / box, 0.75)
        self.assertGreater(_polygon_area(out) / box, 0.5)

    def test_covers_every_rectangle(self):
        """Whatever the shape, no panel may end up outside the outline."""
        out = banded_outline(self.TOP + self.STEM)
        xs = [p[0] for p in out]
        ys = [p[1] for p in out]

        for xmin, xmax, ymin, ymax in self.TOP + self.STEM:
            self.assertGreaterEqual(xmin, min(xs) - 1e-9)
            self.assertLessEqual(xmax, max(xs) + 1e-9)
            self.assertGreaterEqual(ymin, min(ys) - 1e-9)
            self.assertLessEqual(ymax, max(ys) + 1e-9)

    def test_one_row_is_a_rectangle(self):
        out = banded_outline(self.TOP)

        self.assertEqual(len(out), 4, "collinear corners must be dropped")
        self.assertAlmostEqual(_polygon_area(out), 0.50 * 0.18, places=9)

    def test_gaps_between_rows_are_closed(self):
        """
        Two bands with a gap would be two disconnected pieces, and an outline is ONE
        closed loop - it cannot express that at all.
        """
        out = banded_outline([(-0.2, 0.2, 0.1, 0.3), (-0.05, 0.05, -0.3, -0.1)])
        ys = sorted({round(p[1], 6) for p in out})

        self.assertEqual(len(ys), 3, "the two bands must meet at one shared edge")
        self.assertAlmostEqual(ys[1], -0.0, places=6)

    def test_fits_the_config_point_cap(self):
        from tracing.config_io import MAX_POINTS

        rows = [(-0.3 + 0.05 * i, 0.3 - 0.05 * i, 0.3 - 0.1 * i, 0.4 - 0.1 * i)
                for i in range(6)]

        self.assertLessEqual(len(banded_outline(rows)), MAX_POINTS)

    def test_empty_input(self):
        self.assertEqual(banded_outline([]), [])


class TestShapedCutout(unittest.TestCase):

    class _Sol:
        def __init__(self, p, spread=1.0):
            self.position = np.asarray(p, float)
            self.position_spread_mm = spread

    def _pit(self):
        """Two panels across the top, one below - the real WinCtrl layout."""
        sols = {}
        placed = []
        base = 0

        for origin in [(-0.16, 0.89, -0.49), (0.21, 0.90, -0.48), (0.03, 0.69, -0.45)]:
            pose = pose_to_matrix(origin, (-10.0, 0.0, 0.0))
            for i, v in plate_local_points(PLATE_C).items():
                sols[base + (i - 4)] = self._Sol(pose[:3, :3] @ v + pose[:3, 3])

            plate = dict(PLATE_C)
            plate["unit_mm"] = {"w": 167.0, "h": 185.0}
            placed.append(place_from_plate(
                plate, {i: sols[base + (i - 4)] for i in PLATE_C["ids"]}))
            base += 4

        return placed, sols

    def test_produces_an_outline_that_beats_its_box(self):
        placed, sols = self._pit()
        got = shaped_cutout(placed, sols, viewpoint=(0.0, 1.1, -0.2))

        self.assertIsNotNone(got)
        self.assertGreaterEqual(len(got.points), 6)

        box = got.width * got.height
        self.assertLess(_polygon_area(got.points) / box, 0.9)

    def test_outline_lies_in_the_cutout_plane(self):
        """
        The points are 2D in the cutout's own frame. Transforming them by the placement's
        pose must land them on the panels - anything else means the frame is wrong, and a
        wrong frame draws a plausible shape in the wrong place.
        """
        placed, sols = self._pit()
        got = shaped_cutout(placed, sols, viewpoint=(0.0, 1.1, -0.2))

        r = euler_xyz_to_matrix(*got.euler_deg)
        world = [got.position + r @ np.array([x, y, 0.0]) for x, y in got.points]

        for p in np.array([s.position for s in sols.values()]):
            # every marker must sit within the outline's world-space bounding box
            local = r.T @ (p - got.position)
            self.assertLessEqual(abs(local[0]), got.width / 2 + 1e-6)
            self.assertLessEqual(abs(local[1]), got.height / 2 + 1e-6)

        self.assertEqual(len(world), len(got.points))

    def test_needs_markers(self):
        self.assertIsNone(shaped_cutout([], {}, viewpoint=(0, 1.1, 0)))

    def test_panel_rects_are_axis_aligned_in_the_plane(self):
        placed, sols = self._pit()
        pts = np.array([s.position for s in sols.values()])
        origin, r, _ = fit_plane_frame(pts)

        rects = panel_rects_in_plane(placed, origin, r)

        self.assertEqual(len(rects), 3)
        for xmin, xmax, ymin, ymax in rects:
            self.assertGreater(xmax - xmin, 0.15, "a 167 mm unit, roughly")
            self.assertGreater(ymax - ymin, 0.15)


class TestCameraRig(unittest.TestCase):
    """
    The camera offset is where the camera sits relative to the headset's TRACKING ORIGIN -
    a virtual point inside the headset no ruler can reach. It is the least trustworthy
    number in the chain, so changing it must be safe for captures already taken.
    """

    def test_delta_against_a_legacy_capture(self):
        np.testing.assert_allclose(
            offset_delta(CAMERA_OFFSET_LEGACY),
            np.array(CAMERA_OFFSET) - np.array(CAMERA_OFFSET_LEGACY))

        np.testing.assert_allclose(offset_delta(CAMERA_OFFSET), np.zeros(3), atol=0)

    def test_missing_offset_is_assumed_legacy(self):
        """Captures taken before the field existed all used the legacy value."""
        np.testing.assert_allclose(offset_delta(None), offset_delta(CAMERA_OFFSET_LEGACY))

    def test_shift_is_in_the_headset_frame_not_the_world(self):
        """
        The offset is defined in the headset's frame. Applying the delta in world axes is
        right only for a head that never turned - which is exactly what a desk capture
        looks like, so it would pass an eyeball test and fail in a real cockpit sweep.
        """
        from tracing.geometry import camera_to_world_from_hmd, euler_xyz_to_matrix

        hmd = np.eye(4)
        hmd[:3, :3] = euler_xyz_to_matrix(0.0, 90.0, 0.0)     # looking along -X
        hmd[:3, 3] = [0.0, 1.1, 0.0]

        old = Camera(np.eye(3), np.zeros(5),
                     camera_to_world_from_hmd(hmd, CAMERA_OFFSET_LEGACY),
                     image_size=(100, 100))
        expected = camera_to_world_from_hmd(hmd, CAMERA_OFFSET)

        shifted, delta = apply_offset_delta({0: old}, CAMERA_OFFSET_LEGACY)

        np.testing.assert_allclose(shifted[0].camera_to_world[:3, 3],
                                   expected[:3, 3], atol=1e-12)

        # And the naive world-axis version would have been wrong here.
        naive = old.camera_to_world[:3, 3] + delta
        self.assertGreater(float(np.linalg.norm(naive - expected[:3, 3])), 1e-3)

    def test_no_change_returns_the_same_cameras(self):
        cams = {0: Camera(np.eye(3), np.zeros(5), np.eye(4), image_size=(10, 10))}
        out, delta = apply_offset_delta(cams, CAMERA_OFFSET)

        self.assertIs(out, cams)
        np.testing.assert_allclose(delta, np.zeros(3))


class TestBridgedHoles(unittest.TestCase):
    """
    Cutting the MFD screens out of the outline, so the SIM draws the display and
    passthrough covers only the buttons around it.

    There is no second contour in the config format or in the C++ ear clipper, so one loop
    has to describe outer and holes both. Bridging does it: a zero-width slit to the hole,
    the hole walked the opposite way, and back along the same slit.

    Area is the decisive check on both sides of the port. A hole whose winding is wrong
    still triangulates - it just ADDS area, and the screen stays covered with camera video
    while the outline looks broadly right.
    """

    def _area(self, points):
        n = len(points)
        return abs(sum(points[i][0] * points[(i + 1) % n][1] -
                       points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0

    def test_hole_removes_exactly_its_area(self):
        outer = rect_to_points(-0.2, 0.2, -0.2, 0.2)
        hole = rect_to_points(-0.1, 0.1, -0.1, 0.1)

        loop = bridge_hole(outer, hole)

        self.assertEqual(len(loop), 10, "four outer, four hole, two bridge duplicates")
        self.assertAlmostEqual(self._area(loop), 0.16 - 0.04, places=9)

    def test_winding_is_corrected_either_way(self):
        """A caller should not have to know which way round to hand in a hole."""
        outer = rect_to_points(-0.2, 0.2, -0.2, 0.2)
        hole = rect_to_points(-0.1, 0.1, -0.1, 0.1)

        forward = self._area(bridge_hole(outer, hole))
        reversed_ = self._area(bridge_hole(outer, hole[::-1]))

        self.assertAlmostEqual(forward, reversed_, places=9)
        self.assertAlmostEqual(forward, 0.12, places=9)

    def test_three_screens(self):
        outer = rect_to_points(-0.3, 0.3, -0.2, 0.2)
        holes = [rect_to_points(c - 0.05, c + 0.05, -0.05, 0.05)
                 for c in (-0.18, 0.0, 0.18)]

        loop, dropped = outline_with_holes(outer, holes)

        self.assertEqual(dropped, 0)
        self.assertAlmostEqual(self._area(loop), 0.6 * 0.4 - 3 * 0.01, places=9)

    def test_holes_beyond_the_point_cap_are_dropped_not_truncated(self):
        """
        A truncated loop is not a polygon at all - it would draw as garbage, or fall back
        to the rectangle with nothing to say why. Dropping a hole is a visible, reportable
        loss instead.
        """
        from tracing.config_io import MAX_POINTS

        outer = rect_to_points(-0.5, 0.5, -0.3, 0.3)
        holes = [rect_to_points(c - 0.02, c + 0.02, -0.02, 0.02)
                 for c in np.linspace(-0.4, 0.4, 8)]

        loop, dropped = outline_with_holes(outer, holes)

        self.assertLessEqual(len(loop), MAX_POINTS)
        self.assertGreater(dropped, 0)
        self.assertEqual(len(loop), 4 + (8 - dropped) * 6)

    def test_a_hole_needs_three_points(self):
        outer = rect_to_points(-0.2, 0.2, -0.2, 0.2)

        self.assertEqual(bridge_hole(outer, [(0.0, 0.0), (0.1, 0.0)]), outer)

    def test_matches_the_cpp_fixture(self):
        """
        These exact points are compiled into rectus tests/test_mesh.cpp. The Python suite
        validates a PORT of the ear clipper, so pinning shared fixtures is what catches a
        transcription error between the two.
        """
        expected = [(-0.20, -0.20), (-0.10, -0.10), (-0.10, 0.10), (0.10, 0.10),
                    (0.10, -0.10), (-0.10, -0.10), (-0.20, -0.20), (0.20, -0.20),
                    (0.20, 0.20), (-0.20, 0.20)]

        got = bridge_hole(rect_to_points(-0.2, 0.2, -0.2, 0.2),
                          rect_to_points(-0.1, 0.1, -0.1, 0.1))

        for (ax, ay), (bx, by) in zip(got, expected):
            self.assertAlmostEqual(ax, bx, places=9)
            self.assertAlmostEqual(ay, by, places=9)


class TestExcludeScreens(unittest.TestCase):

    class _Sol:
        def __init__(self, p, spread=1.0):
            self.position = np.asarray(p, float)
            self.position_spread_mm = spread

    def _pit(self):
        sols = {}
        placed = []
        plates = {}
        base = 0

        for n, origin in enumerate([(-0.16, 0.89, -0.49), (0.21, 0.90, -0.48),
                                    (0.03, 0.69, -0.45)]):
            pose = pose_to_matrix(origin, (-10.0, 0.0, 0.0))
            for i, v in plate_local_points(PLATE_C).items():
                sols[base + (i - 4)] = self._Sol(pose[:3, :3] @ v + pose[:3, 3])

            plate = dict(PLATE_C)
            plate["name"] = f"p{n}"
            plate["unit_mm"] = {"w": 167.0, "h": 185.0}
            plates[plate["name"]] = plate

            got = place_from_plate(plate, {i: sols[base + (i - 4)] for i in PLATE_C["ids"]})
            got.name = plate["name"]
            placed.append(got)
            base += 4

        return placed, sols, plates

    def _area(self, points):
        n = len(points)
        return abs(sum(points[i][0] * points[(i + 1) % n][1] -
                       points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0

    def test_screens_are_removed_from_the_outline(self):
        placed, sols, plates = self._pit()

        solid = shaped_cutout(placed, sols, viewpoint=(0.0, 1.1, -0.2))
        holed = shaped_cutout(placed, sols, viewpoint=(0.0, 1.1, -0.2),
                              exclude_screens=plates, screen_shrink_mm=0.0)

        removed = self._area(solid.points) - self._area(holed.points)
        one_screen = 0.11812 * 0.11781

        self.assertAlmostEqual(removed, 3 * one_screen, delta=3 * one_screen * 0.05)
        self.assertEqual(holed.dropped_holes, 0)

    def test_shrink_leaves_error_on_the_bezel(self):
        """
        A hole pulled in means alignment error eats bezel, not screen. Which way to err is
        not arbitrary: a little game over the bezel is invisible, a little camera over a
        rendered display is not.
        """
        placed, sols, plates = self._pit()

        tight = shaped_cutout(placed, sols, (0.0, 1.1, -0.2),
                              exclude_screens=plates, screen_shrink_mm=0.0)
        shrunk = shaped_cutout(placed, sols, (0.0, 1.1, -0.2),
                               exclude_screens=plates, screen_shrink_mm=5.0)

        self.assertGreater(self._area(shrunk.points), self._area(tight.points),
                           "shrinking the holes must leave MORE passthrough")

    def test_outline_stays_inside_the_config_cap(self):
        from tracing.config_io import MAX_POINTS

        placed, sols, plates = self._pit()
        holed = shaped_cutout(placed, sols, (0.0, 1.1, -0.2), exclude_screens=plates)

        self.assertLessEqual(len(holed.points), MAX_POINTS)
        self.assertEqual(len(holed.points), 26, "8 outer plus three holes at 6 each")


class TestSegmentCrossing(unittest.TestCase):
    """
    Whether two segments cross PROPERLY. Touching is not crossing.

    This looks like a triviality and is not. A bridged outline is full of touching
    segments by construction - every bridge shares an endpoint with the contour it leaves
    from - so a test that counts touching as crossing calls every candidate bridge
    blocked. The hole is then silently dropped and the MFD screen ends up covered with
    camera video.

    It cost real time: an earlier version of this rule reported 18 self-intersections in a
    perfectly valid cockpit outline, and sent the search for the bug in the wrong
    direction entirely.
    """

    def test_a_real_crossing(self):
        self.assertTrue(_segments_cross((0, 0), (1, 1), (0, 1), (1, 0)))

    def test_shared_endpoint_is_not_a_crossing(self):
        self.assertFalse(_segments_cross((0, 0), (1, 0), (1, 0), (1, 1)))
        self.assertFalse(_segments_cross((0, 0), (1, 0), (0, 0), (0, 1)))

    def test_endpoint_touching_mid_segment_is_not_a_crossing(self):
        """A T-junction. The bridge endpoint lands ON a contour edge and stops there."""
        self.assertFalse(_segments_cross((0, 0), (2, 0), (1, 0), (1, 1)))

    def test_collinear_overlap_is_not_a_crossing(self):
        """The out and back halves of a zero-width slit are exactly this."""
        self.assertFalse(_segments_cross((0, 0), (2, 0), (0, 0), (2, 0)))
        self.assertFalse(_segments_cross((0, 0), (2, 0), (2, 0), (0, 0)))

    def test_disjoint_segments(self):
        self.assertFalse(_segments_cross((0, 0), (1, 0), (0, 5), (1, 5)))


def _proper_crossings(points):
    """Every pair of non-adjacent edges that properly cross."""
    n = len(points)
    out = []

    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue

            if _segments_cross(points[i], points[(i + 1) % n],
                               points[j], points[(j + 1) % n]):
                out.append((i, j))

    return out


class TestRealCockpitOutline(unittest.TestCase):
    """
    The outline the ACTUAL cockpit produces, from the real capture.

    Synthetic fixtures are symmetric and forgiving; this one is neither, and it is the
    shape that actually gets loaded. A self-intersecting outline is rejected by the ear
    clipper and the layer falls back to a RECTANGLE - which covers every screen with
    camera video and looks merely like the cutout being the wrong size.

    The same 26 points are compiled into rectus tests/real_outline.inl and checked against
    the shipping MeshCreatePolygon, so this and that must agree.
    """

    def setUp(self):
        import glob
        import json

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        capture = os.path.join(root, "captures", "anchors.npz")

        if not os.path.exists(capture):
            self.skipTest("no capture to replay")

        from anchors.camera_rig import apply_offset_delta

        z = np.load(capture, allow_pickle=True)
        cameras = {int(k): v for k, v in zip(z["frames"], z["cameras"])}
        cameras, _ = apply_offset_delta(
            cameras, z["camera_offset"] if "camera_offset" in z.files else None)

        self.solutions = solve_markers(list(z["observations"]), cameras)

        self.plates = {}
        for path in sorted(glob.glob(os.path.join(root, "PRINT-THESE", "plates",
                                                  "plate-*.json"))):
            with open(path) as f:
                g = json.load(f)
            if g.get("kind") == "display":
                self.plates[g["name"]] = g

        self.placed = []
        for g in self.plates.values():
            c = place_from_plate(g, self.solutions)
            if c is not None:
                c.name = g["name"]
                self.placed.append(c)

    def test_outline_does_not_self_intersect(self):
        got = shaped_cutout(self.placed, self.solutions, (0.0, 1.1, -0.2),
                            exclude_screens=self.plates, screen_shrink_mm=3.0)

        self.assertEqual(_proper_crossings(got.points), [],
                         "a self-intersecting outline draws as a rectangle over the screens")

    def test_the_solid_T_does_not_self_intersect(self):
        got = shaped_cutout(self.placed, self.solutions, (0.0, 1.1, -0.2))

        self.assertEqual(_proper_crossings(got.points), [])
        self.assertEqual(len(got.points), 8)

    def test_no_screen_is_silently_dropped(self):
        got = shaped_cutout(self.placed, self.solutions, (0.0, 1.1, -0.2),
                            exclude_screens=self.plates, screen_shrink_mm=3.0)

        self.assertEqual(got.dropped_holes, 0)
        self.assertEqual(len(got.points), 26, "8 for the T plus three screens at 6 each")

    def test_holes_remove_the_screen_area(self):
        solid = shaped_cutout(self.placed, self.solutions, (0.0, 1.1, -0.2))
        holed = shaped_cutout(self.placed, self.solutions, (0.0, 1.1, -0.2),
                              exclude_screens=self.plates, screen_shrink_mm=3.0)

        def area(points):
            n = len(points)
            return abs(sum(points[i][0] * points[(i + 1) % n][1] -
                           points[(i + 1) % n][0] * points[i][1] for i in range(n))) / 2.0

        removed = area(solid.points) - area(holed.points)

        self.assertAlmostEqual(removed, 3 * 0.01236, delta=0.002)

    def test_matches_the_cpp_fixture(self):
        """
        rectus tests/real_outline.inl is generated from this and run through the SHIPPING
        MeshCreatePolygon. If they drift apart, the C++ test stops testing what runs.
        """
        inl = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "rectus", "src", "tests", "real_outline.inl")

        if not os.path.exists(inl):
            self.skipTest("rectus checkout not present")

        import re

        with open(inl) as f:
            text = f.read()

        pairs = re.findall(r"([-+][0-9.]+)f,\s*([-+][0-9.]+)f", text)
        got = shaped_cutout(self.placed, self.solutions, (0.0, 1.1, -0.2),
                            exclude_screens=self.plates, screen_shrink_mm=3.0)

        self.assertEqual(len(pairs), len(got.points),
                         "regenerate real_outline.inl - it has drifted from the Python")

        for (sx, sy), (x, y) in zip(pairs, got.points):
            self.assertAlmostEqual(float(sx), x, places=5)
            self.assertAlmostEqual(float(sy), y, places=5)


class TestRangeScale(unittest.TestCase):
    """
    Scaling range by scaling the assumed marker size.

    A cutout that looks BOTH too small and too far away is one error, not two: a fixed
    physical size placed too far subtends a smaller angle. Range comes from apparent size
    (range = focal * size / pixels), so the correction belongs on the size, where it moves
    depth and leaves every direction alone.

    Nudging PosZ instead is the trap - world Z is not the direction from the eye to a
    panel, so it drags the cutout off the panel while appearing to help.
    """

    def _observations(self, size_mm=32.812):
        from anchors.synthetic import arc_of_head_poses, observe, plate_markers

        markers = plate_markers((0.0, 1.0, -0.6), (-20.0, 0.0, 0.0), 69.0, [0, 1, 2, 3],
                                size_mm)
        heads = arc_of_head_poses((0.0, 1.0, -0.6), 0.5, 5)

        return observe(markers, heads, {i: size_mm for i in markers})

    def test_scaling_size_scales_range_proportionally(self):
        obs, cams = self._observations()
        base = solve_markers(obs, cams)

        scaled_obs = [Observation(o.marker_id, o.corners_px, o.camera_to_world,
                                  o.size_mm * 0.9, o.frame) for o in obs]
        scaled = solve_markers(scaled_obs, cams)

        eye = np.array([c.camera_to_world[:3, 3] for c in cams.values()]).mean(axis=0)

        for marker_id in base:
            r0 = np.linalg.norm(base[marker_id].position - eye)
            r1 = np.linalg.norm(scaled[marker_id].position - eye)
            self.assertAlmostEqual(r1 / r0, 0.9, delta=0.02)

    def test_scaling_preserves_direction(self):
        """The angles are the well-measured part. Only depth may move."""
        obs, cams = self._observations()
        base = solve_markers(obs, cams)

        scaled_obs = [Observation(o.marker_id, o.corners_px, o.camera_to_world,
                                  o.size_mm * 0.85, o.frame) for o in obs]
        scaled = solve_markers(scaled_obs, cams)

        eye = np.array([c.camera_to_world[:3, 3] for c in cams.values()]).mean(axis=0)

        for marker_id in base:
            a = base[marker_id].position - eye
            b = scaled[marker_id].position - eye
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
            self.assertGreater(cos, 0.9995, "direction to the marker must not move")

    def test_a_scale_of_one_changes_nothing(self):
        obs, cams = self._observations()
        base = solve_markers(obs, cams)

        same = solve_markers([Observation(o.marker_id, o.corners_px, o.camera_to_world,
                                          o.size_mm * 1.0, o.frame) for o in obs], cams)

        for marker_id in base:
            np.testing.assert_allclose(same[marker_id].position,
                                       base[marker_id].position, atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
