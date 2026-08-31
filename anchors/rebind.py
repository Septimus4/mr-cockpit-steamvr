"""
Keeping cutouts where they belong when the stage frame moves underneath them.

Cutouts are stored in stage coordinates, and that frame is not permanent. A SteamVR
recentre or a re-run room setup re-establishes it with a translation AND a yaw, and every
stored cutout is then wrong by that much - metres, in the case measured here - with nothing
in the config to say so. The cutout simply stops being where the panel is, which reads as
the measurement having been bad.

Touch calibration cannot fix that. It is exact and it is static: the controller measures
where a panel is NOW, in whatever frame happens to be current.

Markers can. They are stuck to the panels, so wherever the frame goes they go with it.
Solving them once alongside the cutouts records a REFERENCE; solving them again later says
how the frame has moved, and the cutouts can be carried across without touching anything.

This is the useful division of labour between the two methods, and the reason to keep both:

    controller   measures the shape, exactly, including buttons a marker cannot see
    markers      notice the frame has moved, and put the shape back
"""

import numpy as np

from tracing.geometry import euler_xyz_to_matrix, matrix_to_euler_xyz

from .place import fit_rigid


def reference_from(solutions):
    """Marker positions to remember, as {id: [x, y, z]}."""
    return {int(i): [float(v) for v in s.position] for i, s in solutions.items()}


def frame_shift(reference, current, min_markers=3):
    """
    The rigid motion carrying the REMEMBERED marker positions onto the current ones.

    Returns (rotation, translation, rms_metres, ids_used), or None if too few markers are
    shared. That transform is the frame change itself, so applying it to a stored cutout
    pose puts the cutout back on its panel.

    Rigid, with no scale: the markers are stuck to panels that did not resize. Letting
    scale float would absorb a genuine disagreement - a marker that has been knocked loose,
    say - into a plausible-looking fit, and the rms is what would otherwise have reported
    it.

    Three markers is the minimum for an orientation, but three nearly-collinear ones give a
    poorly determined one; the rms is the guard, since a bad constellation cannot fit well.
    """
    shared = sorted(set(int(i) for i in reference) & set(int(i) for i in current))

    if len(shared) < min_markers:
        return None

    a = np.array([reference[i] if i in reference else reference[str(i)] for i in shared],
                 dtype=float)
    b = np.array([current[i] for i in shared], dtype=float)

    r, t, rms = fit_rigid(a, b)

    return r, t, rms, shared


def move_pose(position, euler_deg, rotation, translation):
    """
    Carry one cutout pose through a frame change.

    The rotation composes on the LEFT: the cutout has not turned within the world, the
    world has turned under it. Composing on the right would rotate the panel about its own
    normal instead, which looks almost right and is not.
    """
    r = np.asarray(rotation, float).reshape(3, 3)
    t = np.asarray(translation, float).reshape(3)

    new_position = r @ np.asarray(position, float) + t
    new_rotation = r @ euler_xyz_to_matrix(*euler_deg)

    return new_position, matrix_to_euler_xyz(new_rotation)


def shift_summary(rotation, translation):
    """
    How far the frame moved, in terms a person can check against what they remember doing.

    A recentre is a yaw and a slide; reporting those two separately makes it obvious which
    happened, where a 3x3 matrix and a vector do not.
    """
    r = np.asarray(rotation, float).reshape(3, 3)

    angle = float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))
    yaw = float(np.degrees(np.arctan2(r[0, 2], r[2, 2])))
    distance = float(np.linalg.norm(translation))

    return angle, yaw, distance


def binding_is_plausible(marker_positions, cutout_positions, limit_m=0.5):
    """
    Whether the markers and the cutouts were measured in the SAME stage frame.

    A cutout sits on the panel its markers are stuck to, so the nearest marker should be
    within a panel-width. Metres away means the two were measured against different
    origins - and binding them would record a frame change that never happened, then
    "restore" the cutouts by applying it.

    This is not hypothetical: the first binding attempted here paired markers swept before
    a recentre with cutouts touched after one, and every cutout was 0.6-0.8 m from its
    nearest marker.

    Returns (plausible, worst_mm, per_cutout_mm).
    """
    if not marker_positions or not cutout_positions:
        return False, float("inf"), []

    markers = np.array([np.asarray(v, float) for v in marker_positions])
    distances = [float(np.min(np.linalg.norm(markers - np.asarray(p, float), axis=1)))
                 for p in cutout_positions]

    worst = max(distances) if distances else float("inf")

    return worst <= limit_m, worst * 1000.0, [d * 1000.0 for d in distances]
