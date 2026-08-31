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

        for marker_id in g.get("ids", []):
            out[int(marker_id)] = float(g["marker_mm"])

    return out


def detect_markers(image, camera, frame=0, detector=None, allow_diagnostic=False,
                   size_overrides=None):
    """
    Find markers in one frame and return (observations, rejected).

    `size_overrides` maps an id to its true physical size, for markers whose size the id
    cannot describe - display panels, which render whatever fits. See
    plate_size_overrides.

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

        observations.append(Observation(marker_id, c.reshape(4, 2),
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
