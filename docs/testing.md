# Test suite

Two suites, both fast and **neither needing hardware**. That is the point: this is code
where a sign error produces a plausible-looking wrong answer, and checking it by putting
a headset on is slow enough that it would not get done.

## Python — the tracing tool

    cd mr-cockpit-steamvr
    .venv\Scripts\python.exe -m unittest discover -s tests

242 tests, ~1.7 s.

| file | area |
|------|------|
| `tests/test_geometry.py` | rotation convention, planes, ray-plane intersection, camera frame, back-projection round trip, polygon helpers, simplification |
| `tests/test_config_io.py` | the ini format shared with the C++ parser, reading poses, writing outlines without disturbing the file |
| `tests/test_capture.py` | OpenVR pose conversion, capture save/load, and the END-TO-END pipeline |
| `tests/test_probe.py` | controller tip calibration, and turning touched points into cutouts |
| `tests/test_anchors.py` | marker detection, PnP solving, robust averaging, constellation conditioning, and turning solved markers into cutout poses |

## C++ — the mesh that actually ships

    cd rectus\src
    tests
un_tests.bat

62 checks, builds `mesh.cpp` standalone against the project's own headers.

**These exist because the Python suite validates a PORT of the ear-clipping algorithm.**
That catches design errors but cannot catch a transcription error between the two, and
`MeshCreatePolygon` is the one piece of new C++ where a subtle mistake produces a
wrong-shaped cutout rather than a crash.

The decisive check is the same in both: triangle areas must sum EXACTLY to the polygon
area. Too small means gaps, too large means overlaps or triangles outside a concave
outline - which is precisely what a naive triangle fan produces.

Also covered: quad vertex/triangle counts, that subdivision changes density but not area,
that a sized quad stays centred on the origin (the transform is pose-only, so an
off-centre mesh would offset the cutout), and that a traced rectangle covers the same area
as the equivalent Width/Height rectangle.

## The tests that matter most

**Back-projection round trip.** A point on the cutout plane, projected into the image and
clicked back, must return to where it started. One assertion exercises intrinsics,
distortion, the camera axis flip, the headset pose and the plane basis together. Run
against the real calibrated ELP intrinsics, not a toy pinhole.

**End-to-end with a synthetic capture.** `synthetic_capture()` renders a known outline on
a known plane into a fake camera frame. The test then clicks its corners, back-projects,
simplifies, formats to the ini and parses back, and compares to what went in. This is
what makes the tracing tool iterable without a cockpit.

**Convention pinning.** Several tests exist only to fail loudly if a convention drifts:

- Euler composition is `Rz @ Ry @ Rx`, matching `GetQuadToWorldTransform` in C++
- `hmd_matrix_to_numpy` applies NO axis flip (OpenVR and OpenXR agree); the camera's
  Y-down flip belongs only in `camera_to_world_from_hmd`, and applying it twice would
  mirror every traced outline
- the ini format matches `Config_QuadShape::ParseConfig`, including its "fewer than three
  points means use the rectangle" rule and its 32-point cap
- `matrix_to_euler_xyz` is the exact inverse of `euler_xyz_to_matrix`, round-tripped over
  200 random orientations. An error here does not fail - it writes a cutout that is
  rotated, which reads as bad tracking rather than a conversion bug
- a plate's marker centres run Y DOWN (screen coordinates) while the cutout frame runs
  Y up. A sign slip mirrors the cutout

**Placement against a known layout.** `place_from_plate` fits a plate's measured marker
layout onto its solved positions. The tests check that an exact pose comes back exactly,
that the size comes from the measured panel rather than the marker bounding box (118 mm
against 69 mm), that a marker displaced by 1 cm shows up as residual instead of being
absorbed, and that `fit_rigid` never returns a reflection - which fits points beautifully
and is not a pose.

**Config writes that cannot silently no-op.** `config.ini` is owned by the layer and the
settings menu, so a key appended at the end of the file would land in whatever section
came last and never be read. The tests assert new keys land inside `[Quads]`, that
unrelated sections survive untouched, and that a repeated write is byte-identical.

A mismatch in any of these is silent in normal use, which is precisely why they are tested.

**Accuracy against the budget.** One test jitters each click by up to a pixel and asserts
the recovered point moves under 2 mm - well inside the ~2 cm out-of-plane budget from
PLAN.md, so the tracing tool is not the limiting factor.

## Bugs these tests caught while being written

- `np.cross` on 2-vectors, removed in numpy 2, in the simplifier
- a test fixture placing a console 69 degrees off-axis where `cv2.undistortPoints`
  diverges - the fixture was wrong, and `_round_trip` now asserts the point is in frame
  so that mistake names its own cause
- simplification fidelity measured to the nearest VERTEX rather than the nearest EDGE, so
  a point mid-edge looked 19 mm off while lying exactly on the outline
- a C++ area assertion at 1e-9 on float32 data, where the representable precision is
  about 1e-7 - the test was wrong, not the mesh

## Not covered, and why

Grabbing a live frame, the OpenVR session, and the OpenCV window all need hardware or a
display. They are kept as thin as possible - `scripts/trace_cutout.py` is only a window, a
mouse callback and file paths - so that everything which could be silently wrong lives in
the tested modules instead.
