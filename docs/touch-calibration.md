# Touch calibration

Build cutouts by touching your cockpit with a tracked controller.

Point at a real corner, pull the trigger, move to the next one. Four corners is a panel;
walk the edge for a console. That is the whole calibration.

## Why this replaces most of the chain

A controller tip is already in **stage coordinates** — the same space cutouts live in — so
a touched corner *is* the corner, at Lighthouse accuracy. Everything the camera path needs
in order to reach the same answer simply does not apply:

| camera + markers | controller |
|---|---|
| sweep the cockpit, holding still | touch a corner |
| lens intrinsics and distortion | — |
| marker size, and the detector's 1 px corner bias | — |
| camera-to-headset offset, measured with a ruler | — |
| range scale, planar ambiguity, square-on warnings | — |
| **measures the SCREEN only** | **measures anything you can reach** |

That last row is the one that matters. Markers can only ever measure the screen, because
the screen is what draws them. The bezel, the buttons, the UFC stack and the centre console
were all typed in from a datasheet — which is why the MFD cutouts came out too small, and
why the console has never had an anchor of its own.

Markers keep the job they are genuinely good at: **re-anchoring at runtime**, so a cutout
measured once stays put as the headset shifts.

## The one number that has to be right

The tip offset. Get it wrong and every point moves by the same amount — the panel comes out
the right size and the right shape, in the wrong place, and **no residual gives it away**.
It is therefore solved, not taken from the render model, which describes the controller the
driver thinks you have.

    python scripts/touch_cutouts.py --tip

Rest the tip in something that locates it — a screw head, a corner, a recess, not a flat
surface — and roll the controller around it in every direction while holding the trigger.
Each pose satisfies `R_i @ tip + p_i = centre` for one unknown tip and one unknown centre.
That is linear in both, so it is a least-squares solve rather than an optimisation.

Two guards, because the obvious one is not enough:

- **wobble** — the RMS residual, i.e. how far the tip actually moved. Rejected above 4 mm.
- **conditioning** — rotating about a single axis fits the data *perfectly* and still
  leaves the offset free along that axis. The wobble looks excellent and every later point
  is wrong by a constant, so rotation spread is checked separately and `TOO FLAT` refused.

## Measuring

    python scripts/touch_cutouts.py

| control | action |
|---|---|
| trigger | record a point |
| grip | undo the last point |
| menu | finish this cutout, then start the next |
| menu with nothing pending | finish everything and write |

**The order you touch is the shape.** Walking an edge is how a person describes an outline,
and re-sorting the points would quietly turn a deliberate concave console into its convex
hull — exactly the shape a cockpit console is not.

Four corners that form a rectangle are stored as Width/Height rather than as a traced
outline, so the menu's sliders keep working on them; a traced outline ignores those.

Touches that are not coplanar are reported. A finger that slipped off the bezel tilts the
plane, and it should say so rather than quietly producing a skewed cutout.

## Frames

A best-fit plane's in-plane axes come from SVD, which picks the **longest** spread as X.
For a 167 x 185 mm panel that is the tall axis, so Width and Height would come back swapped
and the stored rotation 90 degrees out — harmless to the maths, actively confusing to
anyone reading the config or reaching for the Height slider. `level_frame` rotates the frame
about its own normal until Y points as near world-up as it can. A panel facing straight up
has no meaningful "up" in its own plane, and there the original axes are kept.

## Still true from the camera path

- the layer and the settings menu both own `config.ini`, so the menu must be closed
- `--start N` leaves earlier cutouts alone
- the previous config is kept as `config.ini.bak`
