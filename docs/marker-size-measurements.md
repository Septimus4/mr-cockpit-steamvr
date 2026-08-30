# Marker size measurements, ELP-3DGS1200P01 (left camera, f = 1072 px)

Method: `scripts/marker_test.py`, on-screen test sheet, Lenovo L24q-35 at 0.207 mm/px.
Distance back-solved from measured pixel size: **~73 cm** - further than any F/A-18
instrument, so these are worst-case numbers.

| marker | run A all-3 | run B all-3 | run C all-3 | worst id | px/module |
|--------|-------------|-------------|-------------|----------|-----------|
| 19.9 mm | 5.2% | 5.7% | - | 8.1% | 5.0 |
| 22.4 mm | 64.6% | 70.3% | - | 74.9% | 5.8 |
| 29.8 mm | 63.3% | 71.9% | - | 73.4% | 6.9 |
| 37.3 mm | 84.6% | 87.7% | - | 88.5% | 8.6 |

## Findings

**Detection cliff at ~5.5 px/module.** 5.0 px/module detects 5.7% of frames; 5.8
detects 70%. Not a gentle rolloff - ArUco bit extraction fails outright below a
sampling density. Steeper than the usual ">6 px/module" rule of thumb.

**Detection caps at ~88%,** even at 8.6 px/module. Hand-held head motion plus exposure
time costs ~12% of frames regardless of marker size; making markers bigger does not
fix it. This is why the design anchors quads to Lighthouse tracking and uses markers
only for periodic correction (`AnchorMode = 1`), rather than tracking per-frame.

**Predicted px/module** from f = 1072, marker mm S, distance d mm: `1072*S/(6*d)`.
Measured values land on prediction, so the model can be trusted for placement planning.

| marker | 30 cm | 40 cm | 50 cm | 60 cm | 73 cm |
|--------|-------|-------|-------|-------|-------|
| 22.4 mm (30 mm sticker) | 13.4 | 10.0 | 8.0 | 6.7 | 5.8 measured |
| 37.3 mm (50 mm sticker) | 22.4 | 16.7 | 13.4 | 11.1 | 8.6 measured |

Crossover: the 30 mm sticker drops under the cliff past ~65 cm. Use 50 mm beyond that.

## Caveats

- All readings are from an **emissive screen**. Matte vinyl at an oblique angle in dim
  cockpit light will read worse. Treat as optimistic.
- Detection rate is a proxy. **Pose jitter** is what determines anchoring quality and
  is measured separately by `scripts/jitter_test2.py`.
- A first run was invalidated by a Windows preview window showing a second copy of the
  sheet; ids are now allocated so duplicates are detected and reported, not averaged in.

# Pose jitter, first run (359 frames, ~71 cm, camera propped)

Run with the original `jitter_test.py`, since superseded by `jitter_test2.py` and moved
to `archive/superseded-scripts/`. Its flip metric was flawed - see the correction below -
which is why it is archived rather than kept runnable.

| marker | sigX mm | sigY mm | sigZ mm | flips % |
|--------|---------|---------|---------|---------|
| 19.9 mm | 0.01 | 0.12 | 0.40 | 44.5 |
| 22.4 mm | 0.02 | 0.00 | 0.78 | 46.7 |
| 29.8 mm | 0.16 | 0.47 | 2.69 | 26.7 |
| 37.3 mm | 0.01 | 0.14 | 0.79 | 27.2 |

**Position jitter is excellent** - sub-millimetre laterally, under 3 mm in depth even
for the smallest marker at 71 cm. Position is not the problem.

**Single-marker ORIENTATION is bistable.** 27-47% of frames sit on the far side of a
15 degree jump: the planar pose ambiguity. A flat marker viewed near-frontally admits
two poses that reproject almost identically, and the solver alternates between them.
It improves only weakly with marker size (44% -> 27%), confirming it is geometric
rather than resolution-limited. Larger stickers do not fix it.

The `sigAng` column of that run (2.8-11.5 deg) is NOT the noise floor - MAD loses
robustness past ~50% contamination, so at these flip rates it was measuring the flip
itself. `jitter_test2.py` separates the branches and measures within the dominant one.

## Consequence for the design

Never derive orientation from a single marker. Solve one pose jointly from the corners
of all visible markers, so the constellation's baseline breaks the ambiguity. Practical
rules, pending the constellation measurement:

- minimum 3 markers visible for an orientation fix; below that, position only
- spread them as widely as the panel allows - baseline is what kills the ambiguity
- prefer markers at differing depths where the cockpit permits; a strictly coplanar
  set is the weak case

# CORRECTION: on-screen scale factor

The 100 mm reference on `screen-size-test.png` measured **90 mm** on the L24q-35.
Monitor EDID reports 53 x 30 cm (a genuine 24" panel, 0.207 mm/px), and the generated
files verify correct - all 300 dpi sheets measure 100.25 mm and the PDF MediaBoxes are
right. The image viewer was therefore scaling by 0.90; the file and the panel are fine.

Actual sizes tested were 10% smaller than labelled:

| labelled | actual | px/module measured |
|----------|--------|--------------------|
| 19.9 mm | 17.9 mm | 5.0 |
| 22.4 mm | 20.2 mm | 5.8 |
| 29.8 mm | 26.8 mm | 6.9 |
| 37.3 mm | 33.6 mm | 8.6 |

True test distance ~64 cm (not 71). Jitter-test distances scale the same way, and its
sigZ values are 10% high; the position/orientation conclusions are unchanged, and the
pose ambiguity finding is scale-invariant.

**Unaffected:** the ~5.5 px/module detection cliff. px/module is measured directly off
the image and never depended on the assumed physical size.

**Corrected working ranges** from `px/module = f*S/(6*d)`, f = 1072, validated against
the measurements above to within 3-9%:

| sticker | marker | comfortable (8 px/mod) | cliff (5.5 px/mod) |
|---------|--------|------------------------|--------------------|
| 30 mm | 22.4 mm | 50 cm | 73 cm |
| 50 mm | 37.3 mm | 83 cm | 121 cm |

This supersedes the earlier "30 mm drops out past 65 cm", which was pessimistic because
it attributed the 5.8 px/module reading to a 22.4 mm marker that was really 20.2 mm.

**PRINTING IS UNAFFECTED.** The scale error was in on-screen display only. Always verify
the 100 mm reference with a real ruler after printing, before sticking anything down.

# Corrected-scale runs (display verified 1:1 via scripts/show_1to1.py)

## Detection (`marker_test.py`, 1595 frames, ~63 cm, DELIBERATELY steep angles + motion)

| marker | all-3 | worst id | px/marker | px/module |
|--------|-------|----------|-----------|-----------|
| 19.9 mm | 87.9% | 93.6% | 33.8 | 5.6 |
| 22.4 mm | 91.4% | 94.8% | 37.0 | 6.2 |
| 29.8 mm | 99.2% | 99.2% | 48.5 | 8.1 |
| 37.3 mm | 99.9% | 99.9% | 59.7 | 10.0 |

**RETRACTION:** the earlier claim that detection caps near 88% "regardless of marker
size, due to motion blur" was WRONG. It was the 10% scale error. At true scale, 29.8 mm
and above detect at 99%+ under harsher motion and angles than a cockpit imposes.
Detection is not a limiting factor above ~8 px/module.

## Pose jitter (`jitter_test2.py`, 481 frames, camera propped)

Single marker:

| marker | dominant branch | sigAng in-branch | ambig e1/e2 |
|--------|-----------------|------------------|-------------|
| 19.9 mm | 76.4% | 7.361 deg | 0.83 |
| 22.4 mm | 98.3% | 1.513 deg | 0.84 |
| 29.8 mm | 83.8% | 3.042 deg | 0.64 |
| 37.3 mm | 94.6% | 1.019 deg | 0.86 |

Constellation (3 coplanar markers, one joint solve):

| marker | sigX mm | sigY mm | sigZ mm | sigAng | flips |
|--------|---------|---------|---------|--------|-------|
| 19.9 mm | 0.27 | 0.77 | 1.36 | 1.114 | 13.4% |
| 22.4 mm | 0.48 | 0.58 | 1.42 | 1.018 | 1.9% |
| 29.8 mm | 0.55 | 0.30 | 3.15 | 0.591 | 3.7% |
| 37.3 mm | 0.35 | 0.49 | 4.90 | 1.273 | 27.7% |

**The constellation reduces angular noise 3-6x** (1.0-7.4 deg -> 0.59-1.27 deg) but does
NOT eliminate the bistability, because these markers are strictly coplanar - a flat
screen is the weak geometry case. `ambig` at 0.64-0.86 confirms the runner-up pose
still fits nearly as well.

Caveat: the 27.7% on the widest baseline is suspect - that flip count used an arbitrary
middle frame as reference. Metric since replaced with the robust branch split, and raw
poses are now saved to `archive/jitter_raw.npz` for re-analysis.

## Consequences

- **AnchorMode = 1 (bake once) is comfortably viable.** ~1 deg per-frame angular jitter
  averaged over a few hundred frames is ~0.06 deg. Position jitter is already sub-mm.
- **AnchorMode = 2 (continuous) needs non-coplanar markers** plus hard low-pass
  filtering. Coplanar constellations are not sufficient on their own.
- Placement: spread markers widely AND across differing depths. Depth spread is what
  breaks the ambiguity; lateral spread only averages it down.
- Open: whether oblique viewing angle alone breaks it. Test pending.

# THE DOMINANT ERROR IS LAYOUT, NOT MARKER SIZE

Run 3 (509 frames) decomposed the constellation rotation jitter per axis:

| marker | sigma about x (long axis) | about y | about z (viewing axis) |
|--------|---------------------------|---------|------------------------|
| 19.9 mm | 2.099 deg | 0.804 | 0.243 |
| 22.4 mm | 1.158 deg | 0.968 | 0.301 |
| 29.8 mm | 1.353 deg | 0.968 | 0.224 |
| 37.3 mm | 4.452 deg | 1.465 | 0.209 |

Roll about the viewing axis is pinned at ~0.25 deg. Rotation about the axis the markers
are SPREAD ALONG is 5-20x worse.

Cause: every constellation on the size-test sheet is a ROW - a 3.2-3.7:1 strip. Tilting
a thin strip about its long axis barely changes the image, so that rotation is weakly
observable, and it couples into depth. That is also the whole explanation for the
37.3 mm anomaly: worst sigma-x (4.45 deg) and worst sigZ (13.36 mm) are one effect, not
two. Geometry model verified exact against the sheet (0.146 mm uniform), so this is not
a modelling bug.

This is why the constellation solve gained only 3-6x rather than eliminating the
problem: markers spread in ONE dimension are still near-degenerate.

## Placement rule

- **Never place markers in a line.** Spread them in 2D across the panel.
- Aim for a constellation aspect ratio near 1:1; a 3.5:1 strip costs ~5x in the weak axis.
- Depth variation helps further, but 2D spread is the first-order fix and is free.
- Roll is essentially free at any layout - the constraint is on the two tilt axes.

Verified with `make_layout_test.py` + `layout_test.py`, now in
`archive/superseded-scripts/` - the experiment is complete and its result is recorded
below.

# CONFIRMED: 2D marker layout, verified by position swap

`layout_test.py` compares a 3.6:1 ROW against a 1:1 SQUARE - same marker size (22.4 mm),
same 80 mm lateral span, both solved from the SAME frames. Run twice with the two
layouts swapped between screen positions, to separate layout from lens position.

| layout | position | sigX mm | sigY mm | sigZ mm | rot x | rot y | rot z |
|--------|----------|---------|---------|---------|-------|-------|-------|
| ROW | A (top) | 0.29 | 0.47 | 1.58 | 0.715 | 0.940 | 0.165 |
| ROW | B (mid) | 2.63 | 2.65 | 9.06 | 3.478 | 3.208 | 0.182 |
| SQUARE | B (mid) | 0.21 | 0.14 | 3.18 | 1.027 | 0.415 | 0.052 |
| SQUARE | A (top) | 0.42 | 0.23 | 1.27 | 0.865 | 0.689 | 0.116 |

**The row swings 9.1x in sigX and 4.9x in rot-x with position. The square swings 2.0x
and 1.2x.** In the swapped run the square wins every metric by 4-11x.

The first run looked ambiguous only because the row happened to sit in a favourable
position; that is exactly what the swap was designed to expose.

## Mechanism

A thin strip is ill-conditioned, so it AMPLIFIES small systematic corner errors, and
which errors apply depends on where the markers fall in the lens. A 2D spread is
well-conditioned and suppresses them. The dominant term is therefore systematic error
amplified by poor conditioning - not random per-frame noise.

This also supersedes the earlier per-axis claim ("x = the long axis of the strip"),
which was an unverified label; the row's own numbers contradicted it. The correct
statement is about conditioning, not about a specific axis.

## Placement rule (final)

- **Never place markers in a line.** Spread them in 2D; aim for aspect near 1:1.
- 4 markers in a square beat 3 in a row by 4-11x, and cost one extra sticker.
- Robustness matters more than best case: head motion moves markers around the frame,
  and a row's accuracy swings ~9x depending on where they land. A square stays within 2x.
- Roll is well constrained by any layout (0.05-0.18 deg). The gain is in the tilt axes
  and in position.
