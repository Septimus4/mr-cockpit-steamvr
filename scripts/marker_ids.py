"""
Single source of truth for ArUco id allocation.

The id ENCODES the physical marker size, so a marker's size never has to be configured
by hand: stick it on, and the software knows how big it is from the id alone.  This is
what makes mixed 30 mm / 50 mm cockpits work - near panels get small markers, far
panels get large ones, and nothing needs telling which is which.

SIZE is encoded in the id and cannot be reconfigured - it is a physical property of
the printed marker, and getting it wrong feeds a scale error straight into solvePnP's
range estimate.  ROLE (loose marker vs member of a plate) is NOT encoded in the id: it
is declared in config, so one printed sheet serves any mix of plates and loose markers,
and a plate may be built from any size class.

DICT_4X4_50 gives 50 ids total.

  ids  0-19   30 mm sticker   22.4 mm marker   near panels  (<= 60 cm)
  ids 20-31   50 mm sticker   37.3 mm marker   far panels   (> 60 cm)
  ids 32-43   diagnostics     mixed            test sheets, never mounted
  ids 44-49   reserved                         may extend either size class

Typical allocation (role is config, so this is a convention, not a constraint):
  0-11  three plates of four      12-19  eight loose 30 mm      20-31  loose 50 mm
"""
DICT_NAME = "DICT_4X4_50"

CLASSES = [                       # (id_first, id_last, marker_mm, sticker_mm, label)
    (0,  19, 22.4, 30.0, "30 mm sticker"),
    (20, 31, 37.3, 50.0, "50 mm sticker"),
]
DIAG_BASE = 32                    # diagnostic sheets start here
DIAG_SIZES = [19.9, 22.4, 29.8, 37.3]   # 3 markers each -> ids 32..43

def size_of(mid):
    """Physical marker edge in mm for an id, or None if unallocated."""
    for lo, hi, mm_, _, _ in CLASSES:
        if lo <= mid <= hi: return mm_
    if DIAG_BASE <= mid < DIAG_BASE + 3*len(DIAG_SIZES):
        return DIAG_SIZES[(mid - DIAG_BASE)//3]
    return None

DIAG_MAP = {DIAG_BASE+k: DIAG_SIZES[k//3] for k in range(3*len(DIAG_SIZES))}

def check_plate_ids(ids):
    """Validate a proposed plate. Returns (ok, message)."""
    if len(ids) != len(set(ids)):
        return False, "duplicate ids within the plate"
    diag = [i for i in ids if DIAG_BASE <= i < DIAG_BASE + 3*len(DIAG_SIZES)]
    if diag:
        return False, f"id(s) {diag} are diagnostic-only and must not be mounted"
    sizes = {size_of(i) for i in ids}
    if None in sizes:
        return False, f"id(s) {[i for i in ids if size_of(i) is None]} have no size class"
    if len(sizes) > 1:
        return False, f"mixed marker sizes {sorted(sizes)} - a plate template needs one size"
    if len(ids) < 3:
        return False, "a plate needs at least 3 markers"
    return True, f"{len(ids)} markers at {sizes.pop()} mm"

if __name__ == "__main__":
    for lo, hi, mm_, st, lab in CLASSES:
        print(f"  ids {lo:2d}-{hi:2d}  {lab:14s} marker {mm_:5.1f} mm  ({hi-lo+1} markers)")
    print(f"  ids {DIAG_BASE}-{DIAG_BASE+11}  diagnostics    {DIAG_SIZES} mm, 3 each")
