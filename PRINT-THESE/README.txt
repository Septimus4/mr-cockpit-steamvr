WHAT TO ORDER / PRINT
=====================

Short answer: ORDER THE 30 mm SHEET (and the 50 mm one, to test). Print nothing.


ORDER
-----
  2b-markers-STICKERS-30mm.pdf
      ids 0-19, 22.4 mm marker on a 30 mm square, 20 per sheet, 155 x 139 mm
      MATTE VINYL. 100% scale - no "fit to page", no "scale to fit".

  ONE sheet is enough. These are only for the coaming and the side consoles.
  The three WinCtrl panels are USB DISPLAYS, so they SHOW their markers during
  calibration and drop them afterwards - no stickers needed there at all.
  See scripts/show_plate.py and docs/anchoring-config.md section 8.

  MATTE, never gloss. Specular highlights kill detection.


  2-markers-STICKERS-50mm.pdf
      ids 20-31, 37.3 mm marker on a 50 mm square, 12 per sheet.
      Not strictly needed - it is for placements beyond ~73 cm, and the coaming
      and consoles are within reach. Ordered anyway because the marginal cost is
      small and it answers a real open question (below).

  Both sheets can be mounted at once: the id ranges do not overlap, so the
  software derives each marker's size from its id and the two mix freely.


WORTH TESTING WHEN THEY ARRIVE
------------------------------
  Stick one 30 mm and one 50 mm side by side on the same surface, same lighting,
  same distance, then run:

      .venv\Scripts\python.exe scripts\marker_test.py
      .venv\Scripts\python.exe scripts\jitter_test2.py

  This measures the one thing nothing so far has: HOW MUCH WORSE MATTE VINYL IS
  THAN AN EMISSIVE SCREEN. Every number in docs/marker-size-measurements.md was
  taken on a monitor, and that gap is still unquantified. The 50 mm acts as the
  control - it should still detect well even if vinyl costs more than expected,
  which tells you whether a shortfall is about the material or the size.


DO NOT PRINT
------------
  plates/plate-winctrl-*.pdf
      Sticker placement guides for 117 x 149 mm panels. SUPERSEDED for your
      hardware: your panels are displays. Kept for anyone whose panels are not.

  3-calibration-board-PAPER.pdf
      Camera intrinsics board. Already used - the calibration is done and stored
      in config-final-calibrated.ini.

  1-size-test-PAPER.pdf
      Diagnostic only. The size question is already answered by measurement:
      99%+ detection at 29.8 mm and above, 91% at 22.4 mm under harsher angles
      and motion than a cockpit imposes. See docs/marker-size-measurements.md.


ON SCREEN, NOT PRINTED
----------------------
  screen-size-test.png, screen-layout-test*.png
      Displayed via scripts/show_1to1.py. Diagnostics, already run.


ID ALLOCATION - do not renumber by hand
---------------------------------------
   0-19   30 mm stickers    22.4 mm marker
  20-31   50 mm stickers    37.3 mm marker
  32-43   diagnostics       never mount these
  44-49   reserved

  Size is derived from the id, so a marker's size can never be configured wrong.
  Defined in scripts/marker_ids.py; see docs/anchoring-config.md section 7.
