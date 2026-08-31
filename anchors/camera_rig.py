"""
Where the camera is, and what it sees through.

One place for the rig constants, because they are shared by the tool that captures and
the tool that places, and a disagreement between those two is invisible: every solved
marker simply lands somewhere slightly wrong, consistently, which reads as bad tracking.

The intrinsics come from camera-calibration.exe against a chessboard. The OFFSET does not
- it is where the camera sits relative to the headset's tracking origin, and that origin
is a virtual point inside the headset that no ruler can reach. It is therefore the least
trustworthy number in the whole chain, and the one to suspect first when cutouts land
beside their panels rather than on them.
"""

import numpy as np

K_LEFT = np.array([[1072.26851867, 0.0, 788.49299729],
                   [0.0, 1072.31519651, 614.54444602],
                   [0.0, 0.0, 1.0]])

D_LEFT = np.array([0.08313216691950971, -0.10744697901181298,
                   -0.00016821021003468, 0.00025331486744491, 0.0])

# The value every capture before 2026-08-31 was taken with. Kept so those captures can
# still be replayed correctly - they bake the offset into their stored camera poses, so
# the only way to re-place them under a new offset is to know the old one.
CAMERA_OFFSET_LEGACY = (-0.031, -0.047, -0.138)

# X corrected from -0.031 on 2026-08-31. Three independent in-headset alignments - two
# per-panel cutouts and one whole-assembly cutout - all needed the same -40 to -45 mm
# nudge in X, and applying -0.040 reproduces the hand-tuned position to 0.1 mm.
#
# The direction is a clue to the cause: poses are solved from the ELP's LEFT lens, which
# sits off the camera body's centre. Measuring to the body rather than the lens would
# produce exactly this.
CAMERA_OFFSET = (-0.071, -0.047, -0.138)

# How far INSIDE the true edge the detector places a marker's corners, in camera pixels.
#
# Measured 2026-09-01 at 1.0 px per edge. Subpixel refinement on a bright emissive panel
# pulls the corner in: the white-to-black transition is spread over a couple of pixels by
# the lens and the panel's own glow, and the refined corner lands short of the true edge.
#
# This is deliberately NOT modelled as a smaller marker. The marker's size is KNOWN
# exactly - 210 rendered pixels at a measured pitch - and pretending otherwise would be
# wrong in a way that changes with distance: a fixed pixel bias is a bigger fraction of a
# marker that is further away, while a fixed size error is not. Correcting the corners
# keeps the ground truth intact and scales correctly with range.
#
# Uncorrected it put every panel 3.4% too far away, which reads in the headset as a cutout
# that is at once too small and too distant.
CORNER_BIAS_PX = 1.0


def offset_delta(capture_offset):
    """
    How far to shift stored camera poses when the rig calibration has changed since.

    A capture bakes the offset into its camera-to-world matrices, so replaying it under a
    new offset means shifting every camera by the difference. Without this, re-placing an
    old capture would silently use the old calibration.
    """
    old = np.asarray(capture_offset if capture_offset is not None
                     else CAMERA_OFFSET_LEGACY, float)

    return np.asarray(CAMERA_OFFSET, float) - old


def apply_offset_delta(cameras, capture_offset):
    """
    Re-express stored cameras under the current rig calibration.

    The offset is defined in the HEADSET's frame, so the shift must be rotated into the
    world by the HEADSET's orientation - not the camera's, and not left in world axes.

    The camera basis is the headset basis with Y and Z negated (OpenCV is Y-down), so the
    headset rotation is recovered by applying that flip again; it is its own inverse. For
    a capture taken facing one direction all three choices agree, which is exactly why
    this is worth getting right rather than testing by eye.
    """
    from tracing.geometry import Camera

    delta = offset_delta(capture_offset)

    if not np.any(delta):
        return cameras, delta

    flip = np.diag([1.0, -1.0, -1.0])
    out = {}

    for key, c in cameras.items():
        m = np.array(c.camera_to_world, float, copy=True)
        hmd_rotation = m[:3, :3] @ flip
        m[:3, 3] = m[:3, 3] + hmd_rotation @ delta
        out[key] = Camera(c.K, c.dist, m, image_size=c.image_size)

    return out, delta
