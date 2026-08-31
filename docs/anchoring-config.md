# World-anchored passthrough: marker & anchor configuration

Design note for the `openxr-steamvr-passthrough` anchoring work.
Follows the existing `config.ini` conventions (`Camera0_*` style indexed keys) so it
parses with the same `CSimpleIni` machinery already in `config_manager.cpp`.

## 1. The id encodes the size

A marker's physical edge length is **derived from its ArUco id**, never configured
per-marker by hand. Ids are allocated in contiguous size classes:

| ids | sticker | marker edge | intended use |
|-----|---------|-------------|--------------|
| 0-19 | 30 x 30 mm | 22.4 mm | near panels, <= 60 cm |
| 20-31 | 50 x 50 mm | 37.3 mm | far panels, > 60 cm |
| 32-43 | - | mixed | diagnostic sheets, never mounted |
| 44-49 | - | - | reserved |

Rationale: a mixed cockpit needs small markers on the coaming and large ones on the
far side of a wide pit. Deriving size from id means the user peels a sticker, sticks
it down, and the solver already knows its scale. Getting the size wrong is the single
easiest way to corrupt a pose solve - `solvePnP` puts the error straight into range,
so a 30 mm marker declared as 50 mm sits 67% too far away - and this makes that
mistake unrepresentable rather than merely discouraged.

It also makes duplicate ids impossible across sheets, which would otherwise break
pose solving outright: two physically distinct markers claiming one identity.

Source of truth: `scripts/marker_ids.py` (Python, generates the sheets). The C++ side
must mirror `CLASSES` exactly; a mismatch is silent and produces a scale error, so
the table belongs in one header and is asserted against the generator output.

## 2. Schema

```ini
[Anchoring]
AnchorEnabled       = 1
AnchorMode          = 1     ; 0 = off, 1 = calibrate-once, 2 = continuous correction
MarkerDictionary    = 0     ; 0 = DICT_4X4_50
MarkerSizeOverride  = 0     ; mm; non-zero forces one size for ALL ids (escape hatch
                            ; for hand-cut markers; disables the id->size mapping)

; --- solved marker poses, in stage (play-space) coordinates ---
MarkerCount         = 4
Marker0_Id          = 3
Marker0_PosX        = 0.1842
Marker0_PosY        = 0.9531
Marker0_PosZ        = -0.4127
Marker0_RotX        = ...   ; rotation vector, radians
Marker0_Confidence  = 0.94  ; from observation count + reprojection error
Marker0_Locked      = 0     ; 1 = never re-solve this one

; --- passthrough quads ---
QuadCount           = 2
Quad0_Name          = LeftDDI
Quad0_PosX/Y/Z      = ...   ; centre, stage space
Quad0_RotX/Y/Z      = ...
Quad0_Width         = 0.152 ; metres
Quad0_Height        = 0.152
Quad0_Feather       = 0.004 ; edge softening, metres
Quad0_Enabled       = 1
```

`AnchorMode = 1` is the default and the one that matters for quality: markers are
observed during a short calibration, quad poses are baked into stage space, and
thereafter the quads ride the headset's own Lighthouse tracking. Markers are not
needed in view while flying.

`AnchorMode = 2` re-observes markers continuously to correct drift. Measured detection
rates (70-88% at 73 cm, hand-held) are ample for this because it only needs occasional
fixes, not per-frame lock - but it must low-pass the correction hard, or marker pose
noise is injected straight into the overlay.

## 3. Validation, at load

Reject rather than silently misbehave:

- duplicate `Marker*_Id` -> error, name both entries
- id with no size class and no `MarkerSizeOverride` -> error
- id in the diagnostic range 32-43 -> warn loudly; those are test sheets
- fewer than 3 non-collinear solved markers -> anchoring stays off, quads fall back
  to head-locked, and the reason is surfaced in the menu

Three markers is the floor for a stable plane. Coplanar markers give a weak solve
about the axis normal to their plane; the setup UI should say so when it detects it.

## 4. Open question: marker pose noise budget

Angular jitter dominates the error at distance - 1 degree of marker wobble is 17 mm
of overlay shift at 1 m. `scripts/jitter_test2.py` measures it per size. The threshold
for "acceptable" has not been set yet and should be driven by that measurement, not
guessed here.

## 5. Marker placement is freehand; the software measures it

Users cannot place markers to a specification, and must not be asked to. Nothing in the
setup requires measured positions, right angles, or equal spacing.

**Placement instruction to the user, in full:** stick 4-6 markers around the area you
want anchored, spread out, not in a line, on flat rigid surfaces. That is the whole
requirement.

**Geometry is recovered, not declared.** During calibration the user moves their head
around the cockpit while markers are observed from many viewpoints; bundle adjustment
then solves camera trajectory and marker poses together, in one frame, to sub-mm. This
is standard structure-from-motion with an unusually easy target - the features are
uniquely identified and their sizes are known.

Absolute scale comes from the id -> size mapping (section 1). Without a known physical
size, bundle adjustment leaves scale free; with it, scale is pinned by every marker
independently and can be cross-checked between them. A marker whose solved size
disagrees with its id's declared size is either mis-printed or printed at the wrong
scale - detect it and say so, since that is otherwise a silent range error.

### Conditioning check, at calibration time

Near-collinear constellations are ill-conditioned: they amplify systematic corner error
rather than averaging it down, and their accuracy then depends on where the markers
happen to fall in the lens. Measured: a 3.6:1 strip swung 9x in lateral accuracy with
position, while a 1:1 spread swung 2x (see marker-size-measurements.md).

The setup must therefore evaluate the solved constellation and report, not assume:

- PCA the solved marker centres; take the ratio of the two largest singular values
  - `< 2:1`   good
  - `2-3:1`   warn, suggest where to add a marker to fix it
  - `> 3:1`   refuse to bake; this is the row case and it will be unreliable
- third singular value >> 0 means the markers are non-coplanar - report as a bonus,
  since it is what breaks the planar tilt ambiguity
- report per-marker reprojection RMS after bundle adjustment; flag any outlier, which
  usually means that marker is on something that flexes or has been knocked

The UI should say "your markers are nearly in a line - add one above or below the others"
rather than silently producing an anchor that drifts. A user cannot diagnose poor
conditioning by eye; the solver can measure it exactly.

## 6. Flexibility: loose markers, plates, or both

Cockpits differ enormously. The system must not assume any particular hardware, so it
supports a spectrum, with the simplest option always sufficient.

**The user-facing rule never changes:** stick 4-6 markers around the area you want
anchored, spread out, not in a line, on something rigid.

### Tier 1 - loose markers (always available, no prerequisites)

Stickers placed freehand anywhere. Bundle adjustment recovers all marker poses; the id
gives scale. Works on any cockpit, including surfaces with no flat panel large enough
to hold a group. Requires a slightly longer calibration sweep because every marker's
pose is a free parameter.

### Tier 2 - plates (optional, better when the hardware allows)

A plate is a rigid surface carrying markers in a KNOWN arrangement, from a template we
ship (`scripts/make_plate.py`, parameterised by panel size). Because within-plate
geometry is fixed by construction:

- the solver finds 6 DoF per plate instead of 6 per marker - far fewer parameters, so
  calibration converges faster and from fewer viewpoints
- each plate is individually well-conditioned by design; the template refuses to emit a
  layout with aspect > 3:1
- it is self-validating: a marker knocked out of place shows as a reprojection outlier
  against its own plate, which loose markers cannot detect
- plates are portable. A plate can be re-hung, or moved between rigs, and only its pose
  needs re-solving

Template placement need not be exact; bundle adjustment refines from it. The template
supplies a good initial estimate, not a tolerance requirement.

Shipped: `PRINT-THESE/plates/plate-sticker-117x149-{1,2,3}.pdf` for 117 x 149 mm panels,
4 x 30 mm stickers per plate, 75 x 107 mm spread, 1.43:1. Generate other sizes with
`--w --h --ids`.

These are for panels that are NOT displays. Where a panel IS a display it should show its
markers instead (section 8) - bigger markers, better conditioning, and no adhesive on the
hardware. The two kinds are ALTERNATIVES for a given panel and both start at id 0, so
mixing them across panels needs distinct `--ids` ranges.

### Tier 3 - mixed

Plates plus loose markers, freely combined. Id ranges keep them distinct; a marker not
claimed by any plate is simply a free marker in the same bundle adjustment.

### Why plates suit multi-panel rigs particularly well

Panels mounted at different positions AND angles give wide global spread plus genuine
non-coplanarity - which is what attacks the planar tilt bistability that a single flat
constellation cannot fix. Three panels on different cockpit surfaces is a materially
better target than any single flat arrangement tested on a screen.

### Config

```ini
PlateCount        = 3
Plate0_Name       = LeftPanel
Plate0_Template   = winctrl-1     ; loads the shipped .json geometry
Plate0_PosX/Y/Z   = ...           ; SOLVED, not entered
Plate0_RotX/Y/Z   = ...
Plate0_RmsMM      = 0.31          ; per-plate fit quality, surfaced in the menu
```

Markers not listed in any plate fall back to individual `Marker*_` entries (section 2).

## 7. Size is in the id; role is in the config

**Size is encoded in the id and is immutable.** It is a physical property of the printed
marker; a wrong size feeds a scale error directly into solvePnP's range estimate, which
looks like a plausible pose rather than a fault. This must not be user-configurable.

**Role is NOT encoded in the id.** Whether a marker is loose or belongs to a plate is a
configuration choice, and the same sticker serves either. Baking role into id ranges
would cap the number of plates and the number of loose markers at print time, for no
benefit - and would prevent building a plate from 50 mm markers for a large, distant
panel.

One printed sheet therefore serves every cockpit. Conventional allocation:

| ids | typical use | note |
|-----|-------------|------|
| 0-11 | three plates of four | `Plate*_Ids` in config |
| 12-19 | eight loose 30 mm markers | |
| 20-31 | loose 50 mm markers, or a large/distant plate | |
| 32-43 | diagnostics | rejected if mounted |
| 44-49 | reserved | may extend either size class |

This is a convention, not a constraint. Four plates uses 0-15 and leaves 4 loose; no
plates at all leaves 20 loose. No reprint, no sheet variants.

`marker_ids.check_plate_ids()` enforces what actually matters: no duplicates, no
diagnostic ids, one size class per plate, at least 3 markers. The setup must call it
before accepting a plate definition.

## 8. Display plates (USB panels)

Cockpits with USB display panels (WinCtrl and similar) can DISPLAY markers rather than
carry stickers. This is strictly better where available:

- **geometry is exact by construction** - we place the pixels, so within-plate positions
  are known perfectly rather than estimated from a placement template
- **markers can be temporary** - shown during calibration, dismissed afterwards, so the
  panel loses no display area while flying. Only possible because `AnchorMode = 1` bakes
  the pose once; a continuous-correction mode would need them permanently visible
- **marker size is not limited to stocked sticker sizes** - a 117 x 149 mm panel fits
  four ~33 mm markers, better than the 22.4 mm of a 30 mm sticker
- **no measurement caveat** - all detection data in marker-size-measurements.md was
  taken on an emissive screen, so for display plates it is directly representative
  rather than optimistic

`scripts/show_plate.py` renders them. Scale comes from the panel's MEASURED visible
width, never from EDID - EDID has already been observed on this machine to disagree with
reality, and a scale error here is silent. Verify with `--ruler` before first use.

### Conditioning on tall or wide panels

Four corner markers on a strongly non-square panel give a poorly conditioned spread.
Measured example: an 800x1280 panel at 117 mm wide gives 69 x 139 mm, 2.02:1 MARGINAL.
`--square` centres a 1:1 spread instead (69 x 69 mm) - better conditioned, shorter
baseline. The tool reports the aspect and suggests the alternative rather than choosing
silently.

### Brightness

An emissive panel in a dark cockpit can bloom and drive the camera's exposure down.
`--bg` lowers the background level from white. Global-shutter cameras (the ELP is one)
have no tearing risk from panel refresh.

### Still worth sticking some markers

Display panels in one cockpit are usually near-coplanar with each other. A few loose
stickers on the coaming or side consoles add the depth variation that attacks the planar
tilt ambiguity, which no set of coplanar plates can fix.

## 9. Configuring USB display panels

Panels must be configurable in-app, not hard-coded. Users have different panels, at
different resolutions, and add or remove them.

```ini
DisplayPanelCount            = 3
DisplayPanel0_Name           = LeftMFD
DisplayPanel0_DeviceKey      = MONITOR\WCT1234\{guid}\0002   ; STABLE id, not an index
DisplayPanel0_WidthPx        = 1024
DisplayPanel0_HeightPx       = 768
DisplayPanel0_VisibleWidthMM = 101.6      ; MEASURED via ruler check, never from EDID
DisplayPanel0_ScaleVerified  = 1          ; set only after the ruler check passed
DisplayPanel0_MarkerIds      = 0,1,2,3
DisplayPanel0_MarkerMM       = 0          ; 0 = auto-fit
DisplayPanel0_SquareSpread   = 0          ; 1 forces 1:1 on a non-square panel
DisplayPanel0_Background     = 255        ; lower if the panel blooms in a dark pit
DisplayPanel0_ShowMarkers    = 0          ; runtime toggle, on during calibration only
```

### Identify panels by a stable key, never by index

Monitor indices renumber whenever a display is plugged, unplugged, or powered on in a
different order. Binding a panel's configuration to index 2 silently attaches it to
whichever monitor happens to be second next time - and since the config carries a
physical size, that means rendering markers at the wrong scale on the wrong screen.
Use the device instance path or EDID serial, and re-resolve indices at every startup.

### Scale must be verified, not entered

Product dimensions describe the MODULE, not the glass. A WinCtrl panel quoted at
117 x 149 mm is 1024x768 - a 4:3 ratio that cannot be 117 x 149, so the quoted figure
includes the bezel. Entering 117 when the glass is 101.6 mm puts +15.2% into every
solved distance: a panel at 60 cm reads as 69 cm, with no error raised.

The ruler check is the authority, and it self-corrects exactly:

    true_visible_width = entered_width * (measured_mm / 100)

One iteration converges, and measuring a drawn 100 mm line beats measuring glass edges
under a bezel. `DisplayPanel*_ScaleVerified` must gate use of the panel - an unverified
panel is refused for anchoring rather than trusted.

### Runtime marker toggle

Markers are shown during calibration and dismissed afterwards, so the panel loses no
area while flying. This is only sound because `AnchorMode = 1` bakes the pose once.
If continuous correction is ever enabled, markers must stay visible - reserve a corner
of the panel for a small permanent marker rather than covering the instrument.

### Rotation

Panels are often mounted rotated. Use the physical framebuffer orientation reported per
monitor, and render markers in that frame; do not assume landscape.

## 10. Usable area: panels are rarely fully visible

Cockpits are built from what is on hand. A coaming lip covers the top of a panel, a
bracket eats a corner, a panel sits recessed. The usable area is therefore a
sub-rectangle of the display and differs per panel, so it must be set BY EYE IN THE
COCKPIT - never inferred from a spec sheet.

```ini
DisplayPanel0_UsableX  = 0.0     ; mm, origin of the usable rect within the panel
DisplayPanel0_UsableY  = 25.0
DisplayPanel0_UsableW  = 110.0
DisplayPanel0_UsableH  = 109.7
```

`scripts/show_plate.py` adjusts this live on the panel: Tab selects an edge, arrows move
it (Shift for 10 mm), `[` `]` resize the markers, `b`/`B` dim or brighten the background,
`s` saves. The user drags each edge in until every marker is visible, and the display
reports marker size, spread, aspect and self-detection as they go.

Measured behaviour on a 768x1024 portrait panel at 110 mm wide:

| obstruction | marker | spread | verdict |
|-------------|--------|--------|---------|
| none | 30.9 mm | 63 x 100 | 1.58:1 GOOD |
| 25 mm top, 12 mm bottom | 30.9 mm | 63 x 63 | 1.01:1 GOOD |
| + 15 mm right | 26.6 mm | 52 x 67 | 1.28:1 GOOD |
| 55 mm top and bottom | 10.3 mm | 84 x 10 | 8.08:1 TOO COLLINEAR, 0/4 |

Two things follow. Obstruction is not automatically harmful - losing the top and bottom
SQUARED UP the usable area and improved conditioning from 1.58 to 1.01. And a badly
obscured panel degrades visibly rather than silently: markers auto-shrink to fit, fall
below the detection threshold, and the tool reports 0/4 and TOO COLLINEAR instead of
emitting a plate that cannot work.

The setup must surface the same three numbers - marker size, aspect verdict, and
self-detection - and refuse to save a plate that fails any of them. A user cannot judge
conditioning by eye; the tool can measure it exactly.
