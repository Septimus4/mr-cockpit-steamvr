PLATES
======

A plate is a rigid surface carrying markers in a known arrangement. There are two
kinds here, and they are ALTERNATIVES - use one or the other for a given panel,
never both, or you get duplicate ids.


DISPLAY PLATES - measured, never printed
----------------------------------------
For panels that are USB DISPLAYS. They SHOW their markers during calibration and
drop them afterwards, so the panel loses no area while flying.

  plate-winctrl-L.json   monitor 2   ids 0-3
  plate-winctrl-C.json   monitor 3   ids 4-7
  plate-winctrl-R.json   monitor 4   ids 8-11

Measured 2026-08-31, all three identical:

  panel           768 x 1024 px over 120.0 x 160.0 mm   (0.15625 mm/px)
  usable area     118.1 x 117.8 mm
  inset           top 41.25 mm, other three edges 0.94 mm (6 px)
  marker          32.8 mm, spread 69 x 69 mm, aspect 1.004 GOOD
  self-detect     4/4

The top inset is the mount; the ~1 mm on the other three edges is the bezel
overlapping the glass, which is why an edge-to-edge ruler was invisible.

  python scripts/show_plate.py --monitor 2 --panel-w 120 ^
      --inset 41.25,0.94,0.94,0.94 --ids 0,1,2,3 --name winctrl-L

Press h to hide the overlay once placed - the panel should show nothing but
markers while it is being used.


STICKER PLATES - printed, for panels that are NOT displays
-----------------------------------------------------------
Placement guides for a 117 x 149 mm panel. Lay the printed sheet on the panel,
mark the corners, peel and stick. Placement need not be exact; bundle adjustment
refines it. Print at 100%, and check the 100 mm reference before use.

  plate-sticker-117x149-1.pdf   ids 0-3
  plate-sticker-117x149-2.pdf   ids 4-7
  plate-sticker-117x149-3.pdf   ids 8-11

  22.4 mm marker (30 mm sticker), spread 75 x 107 mm, aspect 1.43:1 GOOD

Other panel sizes:

  python scripts/make_plate.py --w WIDTH --h HEIGHT --name NAME --ids a,b,c,d

The generator refuses a layout worse than 3:1, since a near-collinear plate
amplifies systematic error rather than averaging it down.


WHY THE IDS OVERLAP BETWEEN THE TWO KINDS
-----------------------------------------
Both start at id 0 because they are alternatives: a given panel is either a
display or a sticker surface, never both. Mounting a sticker plate on a panel
that also displays the same ids would put two physically distinct markers under
one identity, which corrupts pose solving rather than degrading it.

If you mix - some panels displays, others not - give each plate its own id range
with --ids. There are 32 ids available across the two size classes; see
docs/anchoring-config.md section 7.
