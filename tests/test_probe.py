"""
Tests for measuring cockpit geometry with a tracked controller.

This path exists to delete most of the calibration chain: a controller tip is already in
stage coordinates, so touching a bezel corner gives its position with no camera, no
marker, no lens model and no detector bias.

What remains to get wrong is the TIP OFFSET. An error there displaces every measured point
by the same amount, which looks like perfect tracking of the wrong object - the most
expensive kind of failure, because nothing about it looks broken. Most of these tests are
about that.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anchors.probe import (
    compare_fingerprints, fit_pivot, level_frame, outline_from_touches,
    pivot_conditioning, pivot_uncertainty, plane_from_touches, tip_position,
)
from tracing.geometry import euler_xyz_to_matrix


def _pose(euler_deg, position):
    m = np.eye(4)
    m[:3, :3] = euler_xyz_to_matrix(*euler_deg)
    m[:3, 3] = position
    return m


def _pivot_poses(tip_offset, centre, angles, noise=0.0, seed=0, axes=(0, 1, 2)):
    """Controller poses whose tip all sit on `centre`, rotated over `angles`."""
    rng = np.random.default_rng(seed)
    poses = []

    for a in angles:
        euler = [a if i in axes else 0.0 for i in range(3)]
        r = euler_xyz_to_matrix(*euler)
        position = np.asarray(centre, float) - r @ np.asarray(tip_offset, float)

        if noise:
            position = position + rng.normal(scale=noise, size=3)

        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = position
        poses.append(m)

    return poses


class TestPivotCalibration(unittest.TestCase):

    TIP = np.array([0.0, -0.015, -0.098])        # a plausible Index-style tip
    CENTRE = np.array([0.12, 0.94, -0.55])

    def test_recovers_a_known_tip_offset(self):
        poses = _pivot_poses(self.TIP, self.CENTRE, np.linspace(-45, 45, 12))
        tip, centre, residual = fit_pivot(poses)

        np.testing.assert_allclose(tip, self.TIP, atol=1e-9)
        np.testing.assert_allclose(centre, self.CENTRE, atol=1e-9)
        self.assertLess(residual, 1e-12)

    def test_survives_realistic_hand_wobble(self):
        """
        A person cannot hold a tip perfectly still. Half a millimetre of wobble must not
        move the answer by more than the alignment budget cares about.
        """
        poses = _pivot_poses(self.TIP, self.CENTRE, np.linspace(-50, 50, 20),
                             noise=0.0005, seed=3)
        tip, _, residual = fit_pivot(poses)

        self.assertLess(float(np.linalg.norm(tip - self.TIP)) * 1000.0, 1.5)
        self.assertGreater(residual, 0.0, "noise must show up in the residual")

    def test_residual_reports_a_bad_pivot(self):
        """A tip that slid during the pivot must be reported, not averaged away."""
        poses = _pivot_poses(self.TIP, self.CENTRE, np.linspace(-45, 45, 12),
                             noise=0.01, seed=7)
        _, _, residual = fit_pivot(poses)

        self.assertGreater(residual * 1000.0, 2.0)

    def test_too_few_poses(self):
        tip, centre, residual = fit_pivot([np.eye(4), np.eye(4)])

        self.assertIsNone(tip)
        self.assertIsNone(centre)
        self.assertEqual(residual, float("inf"))

    def test_a_single_axis_pivot_is_flagged(self):
        """
        Rotating about one axis only fits perfectly and still leaves the offset free along
        that axis. The residual looks excellent and every later point is wrong by a
        constant - so the CONDITIONING has to be checked, not the residual.
        """
        flat = _pivot_poses(self.TIP, self.CENTRE, np.linspace(-40, 40, 10), axes=(0,))
        _, verdict = pivot_conditioning(flat)

        self.assertEqual(verdict, "TOO FLAT")

        good = _pivot_poses(self.TIP, self.CENTRE, np.linspace(-40, 40, 10), axes=(0, 1, 2))
        spread, verdict = pivot_conditioning(good)

        self.assertEqual(verdict, "GOOD", f"spread was {spread:.1f} deg")

    def test_conditioning_needs_samples(self):
        self.assertEqual(pivot_conditioning([np.eye(4)])[1], "TOO FEW")


class TestTipPosition(unittest.TestCase):

    def test_offset_is_applied_in_the_controller_frame(self):
        """
        In world axes it would only be right for a controller held one way - and a person
        touching the underside of a console holds it very differently.
        """
        tip = np.array([0.0, 0.0, -0.1])
        pose = _pose((0.0, 90.0, 0.0), (1.0, 1.0, 1.0))

        got = tip_position(pose, tip)

        # yaw 90 turns -Z into -X
        np.testing.assert_allclose(got, [0.9, 1.0, 1.0], atol=1e-9)

    def test_identity_pose(self):
        np.testing.assert_allclose(tip_position(np.eye(4), [0.1, 0.2, 0.3]),
                                   [0.1, 0.2, 0.3], atol=1e-12)


class TestOutlineFromTouches(unittest.TestCase):
    """Turning touched points into a cutout."""

    def _panel(self, w=0.167, h=0.185, euler=(-11.0, 0.0, 0.0),
               origin=(-0.16, 0.89, -0.49)):
        r = euler_xyz_to_matrix(*euler)
        o = np.asarray(origin, float)
        corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]

        return [o + r @ np.array([x, y, 0.0]) for x, y in corners], o, r

    def test_four_corners_give_the_panel_back(self):
        pts, origin, r = self._panel()
        got = outline_from_touches("left MFD", pts, viewpoint=(0.0, 1.1, 0.0))

        np.testing.assert_allclose(got.position, origin, atol=1e-9)
        self.assertAlmostEqual(got.width, 0.167, places=6)
        self.assertAlmostEqual(got.height, 0.185, places=6)

        normal = euler_xyz_to_matrix(*got.euler_deg)[:, 2]
        self.assertGreater(abs(float(normal @ r[:, 2])), 0.9999)

    def test_a_rectangle_stays_a_rectangle(self):
        """
        Four corners describe a rectangle exactly, and leaving it as Width/Height keeps
        the menu's sliders working on it - a traced outline ignores them.
        """
        pts, _, _ = self._panel()
        got = outline_from_touches("p", pts, viewpoint=(0.0, 1.1, 0.0))

        self.assertEqual(got.points, [])

    def test_a_traced_shape_keeps_its_outline(self):
        pts, _, _ = self._panel()
        r = euler_xyz_to_matrix(-11.0, 0.0, 0.0)
        o = np.array([-0.16, 0.89, -0.49])
        extra = o + r @ np.array([0.0, 0.13, 0.0])          # a notch above the panel

        got = outline_from_touches("console", pts + [extra], viewpoint=(0.0, 1.1, 0.0))

        self.assertEqual(len(got.points), 5)

    def test_touch_order_is_the_outline_order(self):
        """
        Walking the edge is how a person describes a shape. Re-sorting the points would
        turn a deliberate concave outline into its convex hull, which is exactly the shape
        a cockpit console is not.
        """
        r = euler_xyz_to_matrix(-11.0, 0.0, 0.0)
        o = np.array([0.0, 0.9, -0.5])
        local = [(-0.10, -0.06), (0.10, -0.06), (0.10, 0.06),
                 (0.0, 0.0), (-0.10, 0.06)]                 # an arrowhead notch
        pts = [o + r @ np.array([x, y, 0.0]) for x, y in local]

        got = outline_from_touches("c", pts, viewpoint=(0.0, 1.1, 0.0))

        self.assertEqual(len(got.points), 5)
        for (gx, gy), (lx, ly) in zip(got.points, local):
            self.assertAlmostEqual(gx, lx, places=6)
            self.assertAlmostEqual(gy, ly, places=6)

    def test_out_of_plane_touches_are_reported(self):
        """A finger that slipped off the bezel must show up, not silently tilt the plane."""
        pts, _, _ = self._panel()
        pts = list(pts)
        pts[2] = pts[2] + np.array([0.0, 0.0, 0.012])

        got = outline_from_touches("p", pts, viewpoint=(0.0, 1.1, 0.0))
        self.assertGreater(got.flatness_mm, 2.0)

    def test_normal_faces_the_viewer(self):
        pts, _, _ = self._panel()
        viewpoint = np.array([0.0, 1.1, 0.0])
        got = outline_from_touches("p", pts, viewpoint=viewpoint)

        normal = euler_xyz_to_matrix(*got.euler_deg)[:, 2]
        self.assertGreater(float(normal @ (viewpoint - got.position)), 0.0)

    def test_two_points_is_not_a_shape(self):
        self.assertIsNone(outline_from_touches("p", [(0, 0, 0), (0.1, 0, 0)],
                                               viewpoint=(0, 1.1, 0)))

    def test_width_is_the_horizontal_axis(self):
        """
        SVD picks the LONGEST spread as X, so a 167 x 185 panel would come back with its
        Width and Height swapped and its rotation 90 degrees out. Harmless to the maths,
        actively confusing to anyone reading the config or reaching for the Height slider.
        """
        pts, _, _ = self._panel(w=0.167, h=0.185)
        got = outline_from_touches("p", pts, viewpoint=(0.0, 1.1, 0.0))

        self.assertAlmostEqual(got.width, 0.167, places=6)
        self.assertAlmostEqual(got.height, 0.185, places=6)

    def test_levelled_frame_keeps_y_nearest_world_up(self):
        r = euler_xyz_to_matrix(-25.0, 40.0, 15.0)
        levelled = level_frame(r)

        np.testing.assert_allclose(levelled @ levelled.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(float(np.linalg.det(levelled)), 1.0, places=9)
        np.testing.assert_allclose(levelled[:, 2], r[:, 2], atol=1e-9,
                                   err_msg="the plane itself must not move")
        self.assertGreater(float(levelled[:, 1] @ [0, 1, 0]), float(r[:, 1] @ [0, 1, 0]) - 1e-9)

    def test_a_flat_panel_keeps_its_axes(self):
        """A panel facing straight up has no meaningful 'up' in its own plane."""
        r = np.eye(3)[:, [0, 2, 1]] * np.array([1.0, 1.0, -1.0])
        r = np.column_stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(float)

        np.testing.assert_allclose(level_frame(r), r, atol=1e-12)

    def test_plane_fit_reports_flatness(self):
        pts, _, _ = self._panel()
        _, _, worst = plane_from_touches(pts, (0.0, 1.1, 0.0))

        self.assertLess(worst, 1e-9)


class TestTipErrorIsSystematic(unittest.TestCase):
    """
    The failure mode worth naming: a wrong tip offset does not add noise, it adds a
    CONSTANT. Every point moves together, so the panel comes out the right size and shape,
    in the wrong place - and nothing in the residuals says so.
    """

    def test_a_wrong_tip_preserves_shape_but_moves_the_panel(self):
        tip = np.array([0.0, -0.015, -0.098])
        wrong = tip + np.array([0.0, 0.0, 0.02])           # 20 mm short

        r = euler_xyz_to_matrix(-11.0, 0.0, 0.0)
        o = np.array([-0.16, 0.89, -0.49])
        corners = [(-0.0835, -0.0925), (0.0835, -0.0925), (0.0835, 0.0925), (-0.0835, 0.0925)]

        # the same controller orientation at each corner, as a careful user would
        poses = []
        for x, y in corners:
            world = o + r @ np.array([x, y, 0.0])
            m = np.eye(4)
            m[:3, :3] = r
            m[:3, 3] = world - r @ tip
            poses.append(m)

        good = outline_from_touches("p", [tip_position(m, tip) for m in poses], (0, 1.1, 0))
        bad = outline_from_touches("p", [tip_position(m, wrong) for m in poses], (0, 1.1, 0))

        self.assertAlmostEqual(bad.width, good.width, places=9, msg="shape must be intact")
        self.assertAlmostEqual(bad.height, good.height, places=9)

        moved = float(np.linalg.norm(bad.position - good.position))
        self.assertAlmostEqual(moved, 0.02, places=9)

        self.assertAlmostEqual(bad.flatness_mm, good.flatness_mm, places=6,
                               msg="and no residual gives it away - hence pivot_conditioning")


class TestPivotUncertainty(unittest.TestCase):
    """
    How well the pivot actually pinned the tip down, per axis.

    This is the real guard, and the wobble is not. A pivot turned about ONE axis fits the
    data perfectly and keeps a small wobble, while leaving the offset free along that axis
    - so every point measured afterwards is wrong by the same constant, with nothing to
    show for it.
    """

    TIP = np.array([0.0, -0.018, 0.031])
    CENTRE = np.array([0.1, 0.9, -0.5])

    def _poses(self, axes, n=200, noise=0.002, span=40.0, seed=1):
        rng = np.random.default_rng(seed)
        out = []

        for _ in range(n):
            euler = [rng.uniform(-span, span) if i in axes else 0.0 for i in range(3)]
            r = euler_xyz_to_matrix(*euler)

            m = np.eye(4)
            m[:3, :3] = r
            m[:3, 3] = self.CENTRE - r @ self.TIP + rng.normal(scale=noise, size=3)
            out.append(m)

        return out

    def test_a_good_pivot_is_sub_millimetre(self):
        poses = self._poses((0, 1, 2))
        _, _, residual = fit_pivot(poses)

        self.assertLess(float(np.max(pivot_uncertainty(poses, residual))) * 1000.0, 1.0)

    def test_a_single_axis_pivot_is_unbounded(self):
        """
        The case the wobble check cannot see. Rotating about X leaves the tip's X
        component confounded with the pivot centre's, so no amount of data determines it.
        """
        poses = self._poses((0,))
        _, _, residual = fit_pivot(poses)

        self.assertLess(residual * 1000.0, 4.0, "the wobble looks fine, which is the trap")
        self.assertFalse(np.all(np.isfinite(pivot_uncertainty(poses, residual))))

    def test_narrow_rotation_is_worse_than_wide(self):
        wide = self._poses((0, 1, 2), span=40.0)
        narrow = self._poses((0, 1, 2), span=12.0)

        _, _, rw = fit_pivot(wide)
        _, _, rn = fit_pivot(narrow)

        self.assertGreater(float(np.max(pivot_uncertainty(narrow, rn))),
                           float(np.max(pivot_uncertainty(wide, rw))))

    def test_it_scales_with_wobble(self):
        """Twice the hand tremor, twice the uncertainty - nothing else changed."""
        steady = self._poses((0, 1, 2), noise=0.001, seed=4)
        shaky = self._poses((0, 1, 2), noise=0.002, seed=4)

        _, _, rs = fit_pivot(steady)
        _, _, rk = fit_pivot(shaky)

        ratio = (float(np.max(pivot_uncertainty(shaky, rk)))
                 / float(np.max(pivot_uncertainty(steady, rs))))

        self.assertAlmostEqual(ratio, 2.0, delta=0.25)

    def test_too_few_poses_is_unbounded(self):
        self.assertFalse(np.all(np.isfinite(pivot_uncertainty([np.eye(4)] * 2, 0.001))))


class TestStageFingerprint(unittest.TestCase):
    """
    Detecting that the stage origin has been re-established.

    Base stations are bolted to the room, so their positions IN STAGE COORDINATES describe
    the origin, not the hardware. When those numbers change, the room did not move - the
    origin did, and every cutout stored against the old one is wrong by that shift plus
    however much the yaw turned.

    Nothing in a stored config reveals this. A cutout simply stops being where the panel
    is, which reads as the measurement having been bad.
    """

    BEFORE = {"LHB-1": [0.836, 1.409, 0.389], "LHB-2": [0.166, 1.376, 0.825]}

    def test_an_unchanged_origin_is_quiet(self):
        moved, worst, _ = compare_fingerprints(self.BEFORE, dict(self.BEFORE))

        self.assertFalse(moved)
        self.assertAlmostEqual(worst, 0.0, places=9)

    def test_tracking_noise_is_not_a_recentre(self):
        jittered = {k: [v[0] + 0.003, v[1], v[2] - 0.002] for k, v in self.BEFORE.items()}

        self.assertFalse(compare_fingerprints(self.BEFORE, jittered)[0])

    def test_a_recentre_is_caught(self):
        shifted = {k: [v[0] + 0.13, v[1], v[2] + 0.72] for k, v in self.BEFORE.items()}
        moved, worst, detail = compare_fingerprints(self.BEFORE, shifted)

        self.assertTrue(moved)
        self.assertAlmostEqual(worst, 732.0, delta=1.0)
        self.assertIn("LHB-1", detail)

    def test_a_base_station_switched_off_is_not_evidence(self):
        """One device missing means less information, not a moved origin."""
        partial = {"LHB-1": self.BEFORE["LHB-1"]}

        self.assertFalse(compare_fingerprints(self.BEFORE, partial)[0])

    def test_no_common_reference(self):
        moved, worst, detail = compare_fingerprints(self.BEFORE, {"LHB-9": [0, 0, 0]})

        self.assertFalse(moved)
        self.assertEqual(worst, 0.0)
        self.assertIn("no common reference", detail)

    def test_nothing_recorded(self):
        self.assertFalse(compare_fingerprints({}, self.BEFORE)[0])

    def test_the_tolerance_is_adjustable(self):
        nudged = {k: [v[0] + 0.05, v[1], v[2]] for k, v in self.BEFORE.items()}

        self.assertFalse(compare_fingerprints(self.BEFORE, nudged, tolerance_mm=100.0)[0])
        self.assertTrue(compare_fingerprints(self.BEFORE, nudged, tolerance_mm=20.0)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
