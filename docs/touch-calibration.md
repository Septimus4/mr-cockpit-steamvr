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
- **uncertainty** — the per-axis standard error of the tip offset, from the least-squares
  covariance. Rejected above 3 mm on any axis.

The second is the one that matters, and the wobble cannot substitute for it. Rotating
about a single axis fits the data *perfectly* — the wobble stays small — while leaving the
tip's component along that axis confounded with the pivot centre's. No amount of extra
data determines it, and every point measured afterwards is wrong by the same constant with
nothing to show for it. The standard error goes to infinity there; the wobble does not
move.

It must be a real matrix inverse, not a pseudo-inverse: `pinv` truncates small singular
values and so reports an unconstrained direction as having *zero* variance — turning the
worst axis into the apparent best one.

Measured on this rig: a wide pivot lands under 1 mm per axis, a 12-degree one around
0.9 mm, and a single-axis pivot correctly reports infinity.

The raw poses are saved alongside the result, so a calibration can be re-analysed — or
re-solved by a better method — without asking for the pivot to be done again.

## Measuring

    python scripts/touch_cutouts.py

**Touch with the same part of the controller you pivoted on.** The calibration measures
whatever point you planted, not some canonical "tip" — so if you rested it on the base
during calibration, measure with the base. Using a different part shifts every point by
the distance between them.

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
