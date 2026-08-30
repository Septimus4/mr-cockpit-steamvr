"""
Camera captures for the tracing tool.

A capture is one camera frame plus everything needed to back-project a click on it: the
intrinsics, the distortion, and the camera's pose in the world at the moment the frame
was taken. Bundling the pose with the image is the point - trace it later, on the desk,
without the headset on.

The pure parts (pose conversion, save/load, synthetic captures) are unit tested. Only
grab_capture() needs hardware.
"""

import os

import numpy as np

from .geometry import Camera, Plane, camera_to_world_from_hmd


def hmd_matrix_to_numpy(m34):
    """
    OpenVR's HmdMatrix34_t -> 4x4 numpy.

    OpenVR uses the same right-handed, Y-up, -Z-forward convention as OpenXR stage space,
    so this is a pure reshape with no axis flip. The camera's own Y-down flip is applied
    later, in camera_to_world_from_hmd, and belongs only there.
    """
    m = np.eye(4)

    for r in range(3):
        for c in range(4):
            m[r, c] = m34[r][c]

    return m


class Capture:
    """One camera frame with the pose it was taken from."""

    def __init__(self, image, K, dist, camera_to_world, note=""):
        self.image = np.asarray(image)
        self.K = np.asarray(K, float).reshape(3, 3)
        self.dist = np.asarray(dist, float).ravel()
        self.camera_to_world = np.asarray(camera_to_world, float).reshape(4, 4)
        self.note = note

    @property
    def size(self):
        return (self.image.shape[1], self.image.shape[0])

    def camera(self):
        return Camera(self.K, self.dist, self.camera_to_world, image_size=self.size)

    def save(self, path):
        np.savez_compressed(
            path, image=self.image, K=self.K, dist=self.dist,
            camera_to_world=self.camera_to_world, note=np.array(self.note))
        return path

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        return Capture(z["image"], z["K"], z["dist"], z["camera_to_world"],
                       note=str(z["note"]) if "note" in z else "")


def synthetic_capture(camera, plane, outline_uv, image_size=(1600, 1200),
                      fill=200, background=40):
    """
    Render a known outline on a known plane into a fake camera frame.

    This exists so the whole tracing pipeline can be tested end to end - project a shape
    the test already knows, click its corners, and check the recovered outline matches -
    without a camera, a headset or SteamVR. It is the difference between being able to
    iterate on this tool and having to put the hardware on for every change.

    Returns (capture, pixel_corners).
    """
    import cv2

    img = np.full((image_size[1], image_size[0]), background, np.uint8)
    pixels = []

    for u, v in outline_uv:
        px = camera.project(plane.to_world(u, v))

        if px is None:
            raise ValueError(f"outline point ({u}, {v}) is behind the camera")

        pixels.append(px)

    poly = np.array([[int(round(x)), int(round(y))] for x, y in pixels], np.int32)
    cv2.fillPoly(img, [poly], fill)
    cv2.polylines(img, [poly], True, 255, 2)

    cap = Capture(img, camera.K, camera.dist, camera.camera_to_world,
                  note="synthetic")
    return cap, pixels


def grab_capture(camera_index=None, K=None, dist=None, camera_offset=(-0.031, -0.047, -0.138),
                 use_openvr=True):
    """
    Grab a live frame and the current headset pose. NEEDS HARDWARE.

    Without SteamVR the pose is unknown, and a capture with a guessed pose would produce
    an outline that looks plausible and is wrong. So this refuses rather than assuming,
    unless the caller explicitly opts out with use_openvr=False for a bench test.
    """
    import cv2
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

    hmd_to_world = np.eye(4)

    if use_openvr:
        import openvr

        vr = openvr.init(openvr.VRApplication_Background)
        try:
            poses = vr.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0,
                openvr.k_unMaxTrackedDeviceCount)
            hmd = poses[openvr.k_unTrackedDeviceIndex_Hmd]

            if not hmd.bPoseIsValid:
                raise RuntimeError(
                    "headset pose is not valid - is the Beyond tracking? A capture with "
                    "an unknown pose would produce a plausible but wrong outline.")

            hmd_to_world = hmd_matrix_to_numpy(hmd.mDeviceToAbsoluteTracking)
        finally:
            openvr.shutdown()

    from cam import open_elp

    cap = open_elp() if camera_index is None else cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("failed to read a frame from the camera")
    finally:
        cap.release()

    # Left eye of the side-by-side pair, which is what the calibration describes.
    left = frame[:, :frame.shape[1] // 2]
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    return Capture(gray, K, dist, camera_to_world_from_hmd(hmd_to_world, camera_offset),
                   note="live")
