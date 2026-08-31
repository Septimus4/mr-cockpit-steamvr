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


def make_detector(dictionary_name="DICT_4X4_50"):
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def detect_markers(image, camera, frame=0, detector=None, allow_diagnostic=False):
    """
    Find markers in one frame and return (observations, rejected).

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
