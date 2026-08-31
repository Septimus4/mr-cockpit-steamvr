# M07 — anchored placement

Turning solved markers into cutouts, so the passthrough hole lands on the panel instead
of on eighteen numbers typed by hand.

M06 answered *where are the markers*. M07 answers *therefore where is the panel*, and
writes that into the layer's config.

## The chain

```mermaid
flowchart LR
    A["camera frame<br/>+ headset pose"] --> B["detect_markers<br/>anchors/detect.py"]
    B --> C["solve_markers<br/>anchors/solver.py"]
    C --> D["place_from_plate<br/>anchors/place.py"]
    E["plate JSON<br/>measured layout"] --> D
    D --> F["write_quad<br/>tracing/config_io.py"]
    F --> G["config.ini"]
    G --> H["the layer<br/>draws the cutout"]

    style D fill:#2d5016,color:#fff
    style F fill:#2d5016,color:#fff
```

Green is what M07 added. Everything to its left already existed.

## Two ways to place a cutout

### Known layout — `place_from_plate`

A display plate's JSON records every marker centre in panel millimetres, next to the
usable rectangle. The layout is therefore **known**, not inferred, and the placement is a
rigid fit (Kabsch) of that known layout onto the solved world positions.

This is strictly better than fitting a plane through the markers:

| | plane fit | rigid fit to known layout |
|---|---|---|
| size | markers' bounding box — **69 mm** | measured panel — **118 mm** |
| centre | centre of the markers | centre of the panel |
| residual | self-consistency only | disagreement with real geometry |

The residual is the part that matters. A plane fitted through four points is always
flat-ish; a *known* layout fitted onto them cannot hide a bad solve, because the distances
between markers are fixed. **Scale is deliberately not fitted** — the panel's size is
measured, so letting scale float would quietly absorb a range error that should be visible.

### Unknown layout — `place_from_markers`

For markers whose arrangement is not recorded — loose stickers on an irregular console.
Fits a best-fit plane (smallest singular value, not three chosen points), then orients it
towards the viewer, because SVD has no notion of which side is the front.

## Frames

Two sign conventions meet here, and both are the kind of mistake that does not fail:

- **Panel coordinates run Y down**, like the screen the plate is drawn on. The cutout
  frame runs Y up. `plate_local_points` flips it. A slip here mirrors the cutout, which
  reads as a tracking fault.
- **Rotation is stored as intrinsic Euler XYZ in degrees**, composing `Rz @ Ry @ Rx`, to
  match `Config_Quad`. `matrix_to_euler_xyz` is the exact inverse of
  `euler_xyz_to_matrix` and is tested by round-tripping 200 random orientations.

## Writing the config

`config.ini` is owned by the layer **and** by the settings menu. `write_keys` therefore
rewrites only the lines it changes, and appends missing keys **inside `[Quads]`** — a key
appended at the end of file would land in whatever section came last and be silently
ignored.

Two limits come straight from `shared/config_manager.h`:

- `Config_Quad::Name` is `char[16]` read with `strncpy_s(_TRUNCATE)`, so names are cut to
  15 characters here rather than silently by the layer.
- `Enabled` is written as `true`/`false`. SimpleIni accepts `1`/`0` too, but matching what
  the layer writes keeps the file from churning on every save.

## Results on real data

Replaying the 2026-08-31 capture — 391 observations over 60 frames, 12 markers:

| plate | residual | mean per-view spread |
|---|---|---|
| winctrl-C | 1.2 mm | 4.2 mm |
| winctrl-L | 3.3 mm | 3.2 mm |
| winctrl-R | 6.0 mm | 4.4 mm |

Against the ~20 mm out-of-plane budget from [PLAN.md](PLAN.md), all three are comfortable.
The residual is the honest number: it is how far the solved constellation sits from the
panel's *measured* geometry, so it cannot be flattered by averaging.

`winctrl-R` is the worst and is also the one whose markers 8 and 11 were only ever seen
square-on — the planar ambiguity, reported rather than hidden.

**The layout names are wrong.** Solved positions put "L" at +0.21 m (to the *right* of
"C" at −0.16 m) and "R" 20 cm *below*. The names were a guess made before anything was
measured; the solve is what is true.

## The marker-derived size is a STARTING POINT, not the answer

The cutouts exist so the pilot can reach the **physical buttons around the MFDs and the
centre console**. They are not for seeing the MFD screens - the sim renders that content
in-game, and passthrough of a real display looks worse than the in-game version.

So `place_from_plate` sizes the cutout to the panel's *usable display area*
(118.1 x 117.8 mm) because that is what the markers can measure, and that is deliberately
**smaller than what is wanted**. Growing it is expected:

- `--margin MM` grows every cutout symmetrically at placement time
- the Quads tab's Width and Height sliders do it live in the headset, in 5 mm steps,
  which is the faster way to find the right extent the first time

Accuracy expectations follow from this. Reaching a button tolerates far more error than
aligning to a screen edge would, so the 1.2-6.0 mm residuals measured here are ample.

The **centre console has no markers on it at all**. It needs either its own stickers, or a
pose expressed relative to the solved panel anchors so it rides them - the latter is the
remaining half of M07.

## Usage

```bash
python scripts/place_cutouts.py
```

Replays the last capture — no camera, no headset — and prints what it would write.
`--write` commits it, keeping the previous config as `config.ini.bak`.

| flag | effect |
|---|---|
| `--capture PATH` | replay a different capture |
| `--margin MM` | grow every cutout by MM on all sides |
| `--start N` | first quad index, so earlier cutouts are left alone |
| `--config PATH` | write somewhere else — useful for testing |

Sweep once with `scripts/solve_anchors.py`, then place as many times as you like.

## What is not done

- **Nothing re-solves at runtime.** This is a one-shot calibration written to a file. The
  live path — re-solving at 5–10 Hz on a worker thread, low-pass filtered, falling back to
  last-known pose when markers are hidden — is still ahead, and is C++ in the layer.
- **The camera offset is still a ruler measurement.** It is the one number in the whole
  chain that is not solved, so if a cutout sits *beside* its panel rather than on it,
  that is the first suspect.
- **Outlines are not placed.** A newly placed cutout is a rectangle; keeping an existing
  outline would apply the wrong shape to a different panel.
