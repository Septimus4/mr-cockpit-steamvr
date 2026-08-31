"""
Finding markers in camera frames and turning them into observations.

Sizes are never passed in by the caller: they come from the marker's id, via
scripts/marker_ids.py. That is the whole reason size is encoded in the id - a wrong size
does not fail, it puts the error straight into range, and a marker sitting 67% too far
away looks like a plausible pose rather than a fault.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from marker_ids import DIAG_BASE, DIAG_SIZES, size_of

from .camera_rig import CORNER_BIAS_PX
from .solver import Observation

DIAG_LAST = DIAG_BASE + 3 * len(DIAG_SIZES) - 1


def is_diagnostic_id(marker_id):
    """
    Diagnostic ids belong to the test sheets and must never be mounted in a cockpit.

    Seeing one means a test sheet is in view, and its size is whatever that sheet used -
    not something the id can be trusted for. Silently solving it would put a marker of
    unknown scale into the constellation.
    """
    return DIAG_BASE <= marker_id <= DIAG_LAST


def dilate_corners(corners_px, bias_px=None):
    """
    Push a detected marker's corners back out to where its edges really are.

    The detector finds each edge about a pixel INSIDE the true one - see CORNER_BIAS_PX.
    Scaling the corners away from their centroid by (side + 2*bias)/side moves every edge
    out by `bias`, which is exact for a square and close enough for the mild perspective a
    cockpit panel is seen under.

    Corrected here rather than by shrinking the assumed marker size, because the size is
    known exactly and the bias is not proportional to it: the same pixel error is a bigger
    fraction of a marker that is further away.
    """
    bias = CORNER_BIAS_PX if bias_px is None else bias_px

    c = np.asarray(corners_px, float).reshape(4, 2)

    if abs(bias) < 1e-9:
        return c

    centre = c.mean(axis=0)
    side = float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))

    if side < 1e-6:
        return c

    return centre + (c - centre) * (1.0 + 2.0 * bias / side)


def make_detector(dictionary_name="DICT_4X4_50", refine=True):
    """
    A detector with SUBPIXEL CORNER REFINEMENT on.

    OpenCV defaults to CORNER_REFINE_NONE, which locates corners to about a pixel. That
    is the single largest error source in the solve: range is inferred from a marker's
    apparent size, so a pixel of corner error on a 70-pixel marker is 1.4% of range -
    about 9 mm at half a metre, which is exactly the residual measured on real cockpit
    data after size, time skew, camera rotation, camera offset and pose flips had all
    been ruled out.
    """
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    params = cv2.aruco.DetectorParameters()

    if refine:
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.01

    return cv2.aruco.ArucoDetector(d, params)


def plate_size_overrides(plates_dir=None):
    """
    Marker sizes declared by DISPLAY plates, as {id: size_mm}.

    The id -> size map exists so a printed marker's size cannot be configured wrong. A
    DISPLAY plate breaks that guarantee: it renders whatever size fits the panel, which is
    not what the id says. Ignoring this is not a small error - solving a 32.8 mm marker as
    22.4 mm shrinks every range by a third, and because each view then places the marker
    at the wrong distance along a DIFFERENT ray, the estimates scatter instead of simply
    being closer. Measured on real data: 28% scale error and 38 mm of per-view spread.
    """
    import glob
    import json

    if plates_dir is None:
        plates_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "PRINT-THESE", "plates")

    out = {}

    for path in sorted(glob.glob(os.path.join(plates_dir, "plate-*.json"))):
        try:
            with open(path) as f:
                g = json.load(f)
        except (OSError, ValueError):
            continue

        if g.get("kind") != "display":
            continue

        # The DRAWN size, which is known exactly: a whole number of rendered pixels at a
        # measured pitch. The detector's inward bias is corrected on the CORNERS instead -
        # see dilate_corners - because a pixel of bias is not a fixed fraction of a size.
        for marker_id in g.get("ids", []):
            out[int(marker_id)] = float(g["marker_mm"])

    return out


def detect_markers(image, camera, frame=0, detector=None, allow_diagnostic=False,
                   size_overrides=None, corner_bias_px=None):
    """
    Find markers in one frame and return (observations, rejected).

    `size_overrides` maps an id to its true physical size, for markers whose size the id
    cannot describe - display panels, which render whatever fits. See
    plate_size_overrides.

    `corner_bias_px` overrides the calibrated detector bias. Pass 0 for SYNTHETIC frames:
    the bias is a physical property of a real camera looking at a real glowing panel, and
    applying it to a rendered image would inject the very error it exists to remove.

    `rejected` maps a marker id to why it was skipped, so a marker that is seen but not
    used can be reported rather than silently dropped - "I stuck it on and nothing
    happened" is the failure this avoids.
    """
    import cv2

    detector = detector or make_detector()

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    corners, ids, _ = detector.detectMarkers(image)

    observations = []
    rejected = {}

    if ids is None:
        return observations, rejected

    for c, i in zip(corners, ids.flatten()):
        marker_id = int(i)

        if is_diagnostic_id(marker_id) and not allow_diagnostic:
            rejected[marker_id] = "diagnostic id - a test sheet is in view"
            continue

        size_mm = (size_overrides or {}).get(marker_id)

        if size_mm is None:
            size_mm = size_of(marker_id)

        if size_mm is None:
            rejected[marker_id] = "id has no size class, so its scale is unknown"
            continue

        observations.append(Observation(marker_id,
                                        dilate_corners(c.reshape(4, 2), corner_bias_px),
                                        camera.camera_to_world, size_mm, frame))

    return observations, rejected


def summarise(observations, rejected=None):
    """A short human-readable account of what one frame contributed."""
    by_id = {}
    for o in observations:
        by_id.setdefault(o.marker_id, 0)
        by_id[o.marker_id] += 1

    lines = [f"{len(observations)} observation(s) of {len(by_id)} marker(s)"]

    if by_id:
        lines.append("  ids " + ", ".join(str(i) for i in sorted(by_id)))

    for marker_id, why in sorted((rejected or {}).items()):
        lines.append(f"  SKIPPED {marker_id}: {why}")

    return "\n".join(lines)


def refresh_observations(observations, size_overrides, capture_bias_px=0.0):
    """
    Re-apply the CURRENT calibration to observations recorded earlier.

    Two things can have changed since a capture: the marker sizes, and the corner bias.
    Both are applied as a DELTA against what the capture was taken with, so replaying is
    idempotent - dilating corners that were already dilated would double the correction,
    and the result would look like a new error rather than a repeated one.

    A capture with no recorded bias predates the correction and therefore has raw corners.

    Returns (observations, {marker_id: (old_mm, new_mm)}, bias_delta_px).
    """
    from .solver import Observation

    delta = CORNER_BIAS_PX - float(capture_bias_px or 0.0)
    out = []
    changed = {}

    for o in observations:
        size = (size_overrides or {}).get(o.marker_id, o.size_mm)
        corners = o.corners_px

        if abs(delta) > 1e-9:
            corners = dilate_corners(corners, delta)

        if abs(size - o.size_mm) > 1e-9:
            changed[o.marker_id] = (o.size_mm, size)

        if abs(size - o.size_mm) < 1e-9 and abs(delta) < 1e-9:
            out.append(o)
        else:
            out.append(Observation(o.marker_id, corners, o.camera_to_world, size, o.frame))

    return out, changed, delta
