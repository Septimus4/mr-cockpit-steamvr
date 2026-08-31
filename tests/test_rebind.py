"""
Tests for carrying cutouts across a change of stage frame.

The failure this prevents is one that already happened: cutouts measured against one stage
origin, the origin re-established, and every cutout metres away with nothing in the config
to say why. It reads as the measurement having been bad.

The maths is small. What matters is that it composes the rotation on the correct side and
refuses when it cannot know - both mistakes produce a plausible-looking wrong answer rather
than an error.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anchors.rebind import (
    binding_is_plausible, frame_shift, move_pose, reference_from, shift_summary,
)
from tracing.geometry import euler_xyz_to_matrix


class _Sol:
    def __init__(self, position):
        self.position = np.asarray(position, float)


def _markers():
    """Four markers spread over a pit, not collinear and not coplanar."""
    return {
        0: _Sol([-0.16, 0.89, -0.49]),
        1: _Sol([0.21, 0.90, -0.48]),
        2: _Sol([0.03, 0.69, -0.45]),
        3: _Sol([0.02, 1.03, -0.52]),
    }


def _recentre(yaw_deg, offset):
    r = euler_xyz_to_matrix(0.0, yaw_deg, 0.0)
    return r, np.asarray(offset, float)


class TestFrameShift(unittest.TestCase):

    def test_recovers_a_known_recentre(self):
        """The exact case measured on hardware: a 60 degree yaw and a 0.7 m slide."""
        r, t = _recentre(60.0, [0.13, 0.0, 0.72])
        before = reference_from(_markers())
        after = {i: r @ s.position + t for i, s in _markers().items()}

        got = frame_shift(before, after)
        self.assertIsNotNone(got)

        rotation, translation, rms, ids = got

        np.testing.assert_allclose(rotation, r, atol=1e-9)
        np.testing.assert_allclose(translation, t, atol=1e-9)
        self.assertLess(rms, 1e-12)
        self.assertEqual(ids, [0, 1, 2, 3])

    def test_an_unmoved_frame_is_the_identity(self):
        before = reference_from(_markers())
        rotation, translation, rms, _ = frame_shift(before, {i: np.array(v) for i, v in before.items()})

        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(translation, np.zeros(3), atol=1e-9)

    def test_a_knocked_marker_shows_up_in_the_rms(self):
        """
        Scale is deliberately not fitted, so a marker that has moved relative to the others
        cannot be absorbed into a plausible fit - it has to come out as residual.
        """
        r, t = _recentre(20.0, [0.1, 0.0, 0.1])
        before = reference_from(_markers())
        after = {i: r @ s.position + t for i, s in _markers().items()}
        after[2] = after[2] + np.array([0.03, 0.0, 0.0])

        _, _, rms, _ = frame_shift(before, after)
        self.assertGreater(rms * 1000.0, 5.0)

    def test_too_few_shared_markers(self):
        before = reference_from(_markers())
        self.assertIsNone(frame_shift(before, {0: np.zeros(3), 1: np.ones(3)}))

    def test_only_shared_markers_are_used(self):
        """A marker present now but not then is not evidence about how the frame moved."""
        r, t = _recentre(15.0, [0.0, 0.0, 0.2])
        before = reference_from(_markers())
        after = {i: r @ s.position + t for i, s in _markers().items()}
        after[9] = np.array([5.0, 5.0, 5.0])

        rotation, translation, rms, ids = frame_shift(before, after)

        self.assertEqual(ids, [0, 1, 2, 3])
        np.testing.assert_allclose(rotation, r, atol=1e-9)
        self.assertLess(rms, 1e-12)

    def test_string_keys_from_json_still_match(self):
        """A remembered reference comes back from JSON with string keys."""
        r, t = _recentre(30.0, [0.2, 0.0, 0.1])
        before = {str(i): v for i, v in reference_from(_markers()).items()}
        after = {i: r @ s.position + t for i, s in _markers().items()}

        got = frame_shift({int(k): v for k, v in before.items()}, after)
        self.assertIsNotNone(got)


class TestMovePose(unittest.TestCase):

    def test_a_cutout_follows_its_panel(self):
        """
        The decisive property: a cutout carried through the frame change must end up in the
        same place RELATIVE TO THE MARKERS as it started.
        """
        r, t = _recentre(60.0, [0.13, 0.0, 0.72])

        markers = _markers()
        before = reference_from(markers)
        after = {i: r @ s.position + t for i, s in markers.items()}

        position = np.array([-0.16, 0.89, -0.49])
        euler = (-11.0, 2.0, 1.0)

        # where the cutout sits relative to marker 0, before
        offset_before = position - markers[0].position

        rotation, translation, _, _ = frame_shift(before, after)
        moved, moved_euler = move_pose(position, euler, rotation, translation)

        offset_after = moved - after[0]

        # the same offset, expressed in the new frame
        np.testing.assert_allclose(offset_after, rotation @ offset_before, atol=1e-9)

        # and the panel's normal turned with the world, not about itself
        np.testing.assert_allclose(euler_xyz_to_matrix(*moved_euler),
                                   rotation @ euler_xyz_to_matrix(*euler), atol=1e-9)

    def test_the_rotation_composes_on_the_left(self):
        """
        Composing on the right would spin the panel about its own normal instead of
        turning it with the world - which looks almost right, and is not.
        """
        r = euler_xyz_to_matrix(0.0, 90.0, 0.0)
        _, euler = move_pose([0.0, 0.0, 0.0], (0.0, 0.0, 0.0), r, np.zeros(3))

        np.testing.assert_allclose(euler_xyz_to_matrix(*euler), r, atol=1e-9)

    def test_the_identity_changes_nothing(self):
        position, euler = move_pose([0.1, 0.2, 0.3], (10.0, 20.0, 30.0),
                                    np.eye(3), np.zeros(3))

        np.testing.assert_allclose(position, [0.1, 0.2, 0.3], atol=1e-12)
        np.testing.assert_allclose(euler_xyz_to_matrix(*euler),
                                   euler_xyz_to_matrix(10.0, 20.0, 30.0), atol=1e-9)


class TestShiftSummary(unittest.TestCase):
    """
    A recentre is a yaw and a slide. Reporting those two makes it obvious what happened,
    where a 3x3 matrix and a vector do not.
    """

    def test_reports_yaw_and_distance(self):
        r, t = _recentre(60.0, [0.13, 0.0, 0.72])
        angle, yaw, distance = shift_summary(r, t)

        self.assertAlmostEqual(angle, 60.0, places=4)
        self.assertAlmostEqual(abs(yaw), 60.0, places=4)
        self.assertAlmostEqual(distance, 0.7317, places=3)

    def test_an_unmoved_frame_reports_nothing(self):
        angle, yaw, distance = shift_summary(np.eye(3), np.zeros(3))

        self.assertAlmostEqual(angle, 0.0, places=6)
        self.assertAlmostEqual(distance, 0.0, places=9)


class TestBindingIsPlausible(unittest.TestCase):
    """
    Catching a binding made from two different stage frames.

    A cutout sits on the panel its markers are stuck to, so the nearest marker should be
    within a panel-width. This is not hypothetical - the first binding attempted on this
    rig paired markers swept before a recentre with cutouts touched after one, and every
    cutout was 0.6-0.8 m from its nearest marker. Binding them would have recorded a frame
    change that never happened, and restore would then have applied it.
    """

    MARKERS = [[-0.16, 0.89, -0.49], [0.21, 0.90, -0.48], [0.03, 0.69, -0.45]]

    def test_a_same_frame_binding_passes(self):
        cutouts = [[-0.16, 0.89, -0.49], [0.21, 0.90, -0.48]]
        ok, worst, each = binding_is_plausible(self.MARKERS, cutouts)

        self.assertTrue(ok)
        self.assertLess(worst, 1.0)
        self.assertEqual(len(each), 2)

    def test_a_cutout_a_panel_width_away_still_passes(self):
        """Markers sit on the screen; a cutout centre is a bezel away, not a metre."""
        cutouts = [[-0.16, 0.89 + 0.09, -0.49]]

        self.assertTrue(binding_is_plausible(self.MARKERS, cutouts)[0])

    def test_the_real_mismatch_is_caught(self):
        """The measured case: 0.6-0.8 m, from a recentre between the two measurements."""
        cutouts = [[0.536, 0.912, -0.097], [0.427, 0.931, 0.057],
                   [0.334, 0.914, 0.219], [0.412, 0.712, 0.050]]

        ok, worst, each = binding_is_plausible(self.MARKERS, cutouts)

        self.assertFalse(ok)
        self.assertGreater(worst, 500.0)
        self.assertEqual(len(each), 4)

    def test_one_bad_cutout_fails_the_whole_binding(self):
        """A frame change moves everything, so one outlier means the set is unusable."""
        cutouts = [[-0.16, 0.89, -0.49], [3.0, 0.9, 3.0]]

        self.assertFalse(binding_is_plausible(self.MARKERS, cutouts)[0])

    def test_nothing_to_compare(self):
        self.assertFalse(binding_is_plausible([], [[0, 0, 0]])[0])
        self.assertFalse(binding_is_plausible(self.MARKERS, [])[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
