"""
Tests for reading and writing cutout data in the layer's config.ini.

These matter because a formatting mismatch with the C++ parser is SILENT: the layer would
fall back to a rectangle, or load a subtly different outline, with nothing to say why.
The tests below encode what shared/config_manager.h actually accepts.
"""

import codecs
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracing.config_io import (
    MAX_NAME, MAX_POINTS, QuadConfig, format_points, parse_points, read_quads,
    write_keys,
    write_points, write_quad,
)

SAMPLE_INI = """\
[Main]
ProjectionMode = 3
FloorHeightOffset = 0.0

[Quads]
QuadSubdivisions = 1
QuadsExclusive = false
Quad0_Enabled = true
Quad0_Name = Left DDI
Quad0_PosX = -0.180000
Quad0_PosY = 1.050000
Quad0_PosZ = -0.640000
Quad0_RotX = -28.500000
Quad0_RotY = 12.000000
Quad0_RotZ = 0.000000
Quad0_Width = 0.150000
Quad0_Height = 0.150000
Quad0_Points = -0.07500,-0.07500;0.07500,-0.07500;0.07500,0.07500;-0.07500,0.07500
Quad1_Enabled = false
Quad1_Name =
Quad1_PosZ = -0.800000
Quad1_Points =

[Camera]
CameraFrameLayout = 2
"""


class TestFormatParse(unittest.TestCase):

    def test_round_trip(self):
        pts = [(0.0, 0.0), (0.5, 0.02), (0.52, 0.10), (0.48, 0.18), (0.30, 0.22)]
        back = parse_points(format_points(pts))

        self.assertEqual(len(back), len(pts))
        for (ax, ay), (bx, by) in zip(pts, back):
            self.assertAlmostEqual(ax, bx, places=5)
            self.assertAlmostEqual(ay, by, places=5)

    def test_format_uses_semicolons_and_commas(self):
        """The exact separators the C++ parser scans for."""
        v = format_points([(0, 0), (1, 0), (1, 1)])
        self.assertEqual(v, "0.00000,0.00000;1.00000,0.00000;1.00000,1.00000")

    def test_negative_and_small_values(self):
        pts = [(-0.075, -0.075), (0.075, -0.0001), (0.0, 0.075)]
        back = parse_points(format_points(pts))
        for (ax, ay), (bx, by) in zip(pts, back):
            self.assertAlmostEqual(ax, bx, places=5)
            self.assertAlmostEqual(ay, by, places=5)

    def test_fewer_than_three_points_is_no_outline(self):
        """Matches the C++ rule: under three points means use the rectangle."""
        self.assertEqual(format_points([(0, 0), (1, 1)]), "")
        self.assertEqual(parse_points("0,0;1,1"), [])

    def test_empty_value(self):
        self.assertEqual(parse_points(""), [])
        self.assertEqual(parse_points(None), [])

    def test_point_cap_enforced_on_write(self):
        pts = [(i * 0.01, i * 0.01) for i in range(MAX_POINTS + 20)]
        self.assertEqual(len(parse_points(format_points(pts))), MAX_POINTS)

    def test_point_cap_enforced_on_read(self):
        """The C++ parser stops at MAX_POINTS; a longer value must not overrun."""
        value = ";".join(f"{i*0.01:.5f},{i*0.01:.5f}" for i in range(MAX_POINTS + 20))
        self.assertEqual(len(parse_points(value)), MAX_POINTS)

    def test_tolerates_whitespace(self):
        pts = parse_points(" 0.0 , 0.0 ; 1.0 , 0.0 ; 1.0 , 1.0 ")
        self.assertEqual(len(pts), 3)

    def test_stops_at_malformed_input(self):
        """A truncated value yields what parsed cleanly, never an exception."""
        self.assertEqual(parse_points("0,0;1,0;1,1;garbage"), [(0, 0), (1, 0), (1, 1)])
        self.assertEqual(parse_points("nonsense"), [])

    def test_scientific_notation_accepted(self):
        pts = parse_points("1e-3,2e-3;1,0;0,1")
        self.assertEqual(len(pts), 3)
        self.assertAlmostEqual(pts[0][0], 0.001, places=9)


class TestReadQuads(unittest.TestCase):

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_INI)

    def tearDown(self):
        os.unlink(self.path)

    def test_reads_pose_and_outline(self):
        q = read_quads(self.path)[0]

        self.assertTrue(q.enabled)
        self.assertEqual(q.name, "Left DDI")
        self.assertAlmostEqual(q.position[0], -0.18, places=6)
        self.assertAlmostEqual(q.position[1], 1.05, places=6)
        self.assertAlmostEqual(q.position[2], -0.64, places=6)
        self.assertAlmostEqual(q.euler_deg[0], -28.5, places=6)
        self.assertAlmostEqual(q.euler_deg[1], 12.0, places=6)
        self.assertEqual(len(q.points), 4)

    def test_missing_keys_use_cpp_defaults(self):
        q = read_quads(self.path)[1]

        self.assertFalse(q.enabled)
        self.assertAlmostEqual(q.position[2], -0.8, places=6)
        self.assertAlmostEqual(q.width, 0.30, places=6)      # C++ default
        self.assertAlmostEqual(q.height, 0.20, places=6)     # C++ default
        self.assertEqual(q.points, [])

    def test_absent_quads_still_returned(self):
        quads = read_quads(self.path)
        self.assertEqual(len(quads), 8)
        self.assertFalse(quads[7].enabled)

    def test_label_falls_back_to_index(self):
        quads = read_quads(self.path)
        self.assertEqual(quads[0].label, "Left DDI")
        self.assertEqual(quads[1].label, "Quad 1")


class TestWritePoints(unittest.TestCase):

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_INI)

    def tearDown(self):
        os.unlink(self.path)

    def test_write_then_read_back(self):
        pts = [(-0.1, -0.05), (0.1, -0.05), (0.12, 0.06), (-0.08, 0.07)]
        write_points(0, pts, self.path)

        back = read_quads(self.path)[0].points
        self.assertEqual(len(back), 4)
        for (ax, ay), (bx, by) in zip(pts, back):
            self.assertAlmostEqual(ax, bx, places=5)
            self.assertAlmostEqual(ay, by, places=5)

    def _read(self):
        with open(self.path, encoding="utf-8") as f:
            return f.read()

    def test_other_settings_are_untouched(self):
        """
        The layer and the settings menu both own this file. Writing one outline must not
        disturb anything else, or the tool would quietly undo the user's settings.
        """
        before = self._read()
        write_points(0, [(0, 0), (0.1, 0), (0.1, 0.1)], self.path)
        after = self._read()

        for line in before.splitlines():
            if line.startswith("Quad0_Points"):
                continue
            self.assertIn(line, after, f"line lost or altered: {line!r}")

    def test_adds_key_when_absent(self):
        write_points(3, [(0, 0), (0.1, 0), (0.1, 0.1)], self.path)
        self.assertEqual(len(read_quads(self.path)[3].points), 3)

    def test_clearing_an_outline(self):
        write_points(0, [], self.path)

        self.assertEqual(read_quads(self.path)[0].points, [])
        self.assertIn("Quad0_Points =", self._read())

    def test_write_is_idempotent(self):
        pts = [(0, 0), (0.1, 0), (0.1, 0.1)]
        write_points(0, pts, self.path)
        first = self._read()
        write_points(0, pts, self.path)

        self.assertEqual(first, self._read())

    def test_does_not_write_into_another_section(self):
        """Quad keys belong in [Quads], never in [Camera] further down the file."""
        write_points(5, [(0, 0), (0.1, 0), (0.1, 0.1)], self.path)
        text = self._read()

        quads_at = text.index("[Quads]")
        camera_at = text.index("[Camera]")
        key_at = text.index("Quad5_Points")

        self.assertGreater(key_at, quads_at)
        self.assertLess(key_at, camera_at)


class TestBomHandling(unittest.TestCase):
    """
    The layer writes config.ini with a UTF-8 BOM. Read it as plain utf-8 and the FIRST
    section header parses as "﻿[Main]", which matches nothing - silently, and only
    for the first section, so everything below it keeps working and the bug hides.

    This was found the hard way: a key destined for [Main] was appended to the end of the
    file instead, landing inside [Quads] and creating a duplicate. The fixtures above use
    a plain file, so none of them caught it.
    """

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        with open(self.path, "wb") as f:
            f.write(codecs.BOM_UTF8 + SAMPLE_INI.encode("utf-8"))

    def tearDown(self):
        os.unlink(self.path)

    def test_first_section_is_still_found(self):
        """[Main] is the first line, right after the BOM - the case that failed."""
        from tracing.config_io import _read_ini
        sections = _read_ini(self.path)
        self.assertIn("Main", sections, "the BOM hid the first section header")
        self.assertEqual(sections["Main"].get("ProjectionMode"), "3")

    def test_quads_read_normally(self):
        q = read_quads(self.path)[0]
        self.assertTrue(q.enabled)
        self.assertEqual(len(q.points), 4)

    def test_write_preserves_the_bom(self):
        """
        The layer expects the file to start as it wrote it. Stripping the BOM on write
        would change a file another program owns.
        """
        write_points(0, [(0, 0), (0.1, 0), (0.1, 0.1)], self.path)
        with open(self.path, "rb") as f:
            self.assertTrue(f.read(3) == codecs.BOM_UTF8, "the BOM was dropped on write")

    def test_write_does_not_duplicate_or_relocate_keys(self):
        write_points(0, [(0, 0), (0.1, 0), (0.1, 0.1)], self.path)
        with open(self.path, encoding="utf-8-sig") as f:
            text = f.read()

        self.assertEqual(text.count("Quad0_Points"), 1, "key was duplicated")
        self.assertEqual(text.count("ProjectionMode"), 1, "an unrelated key was duplicated")
        self.assertLess(text.index("Quad0_Points"), text.index("[Camera]"),
                        "key escaped the [Quads] section")


class TestWriteQuad(unittest.TestCase):
    """
    Writing a whole cutout, which is how a solved anchor becomes one.

    The layer and the settings menu both own this file, so the danger is not a wrong
    value - it is a write that drops or relocates lines this tool does not model, which
    would look like the layer losing settings.
    """

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write(SAMPLE_INI)

    def tearDown(self):
        os.unlink(self.path)

    def test_round_trips_a_whole_cutout(self):
        q = QuadConfig(1, enabled=True, name="Centre panel",
                       position=(0.02, 1.031, -0.588), euler_deg=(-27.4, 3.1, -0.8),
                       width=0.118, height=0.1178,
                       points=[(-0.05, -0.05), (0.05, -0.05), (0.05, 0.05)])
        write_quad(q, self.path)

        back = read_quads(self.path)[1]

        self.assertTrue(back.enabled)
        self.assertEqual(back.name, "Centre panel")
        for a, b in zip(back.position, q.position):
            self.assertAlmostEqual(a, b, places=5)
        for a, b in zip(back.euler_deg, q.euler_deg):
            self.assertAlmostEqual(a, b, places=3)
        self.assertAlmostEqual(back.width, 0.118, places=5)
        self.assertAlmostEqual(back.height, 0.1178, places=5)
        self.assertEqual(len(back.points), 3)

    def test_keys_missing_from_the_file_are_added_inside_the_section(self):
        """
        The sample has no Quad1_PosX. Appended to the END OF FILE it would land in
        [Camera], where the layer's parser never looks - a silent no-op.
        """
        write_quad(QuadConfig(1, position=(0.5, 0.6, -0.7)), self.path)

        with open(self.path, encoding="utf-8-sig") as f:
            text = f.read()

        self.assertEqual(text.count("Quad1_PosX"), 1)
        self.assertLess(text.index("Quad1_PosX"), text.index("[Camera]"))
        self.assertAlmostEqual(read_quads(self.path)[1].position[0], 0.5, places=5)

    def test_leaves_other_cutouts_and_sections_untouched(self):
        before = read_quads(self.path)[0]
        write_quad(QuadConfig(1, name="new"), self.path)
        after = read_quads(self.path)[0]

        self.assertEqual(after.name, before.name)
        self.assertEqual(after.position, before.position)
        self.assertEqual(after.points, before.points)

        with open(self.path, encoding="utf-8-sig") as f:
            text = f.read()

        self.assertIn("FloorHeightOffset = 0.0", text)
        self.assertIn("CameraFrameLayout = 2", text)
        self.assertEqual(text.count("[Quads]"), 1)

    def test_rewriting_twice_does_not_grow_the_file(self):
        write_quad(QuadConfig(2, name="a", position=(1, 2, 3)), self.path)
        first = pathlib.Path(self.path).read_text(encoding="utf-8-sig")

        write_quad(QuadConfig(2, name="a", position=(1, 2, 3)), self.path)
        second = pathlib.Path(self.path).read_text(encoding="utf-8-sig")

        self.assertEqual(first, second, "a repeated write must be idempotent")

    def test_name_is_truncated_to_what_the_layer_can_hold(self):
        """
        Config_Quad::Name is char[16] read with _TRUNCATE. Writing a longer name would
        make the file disagree with what the layer actually loads.
        """
        write_quad(QuadConfig(3, name="a-very-long-panel-name"), self.path)

        back = read_quads(self.path)[3].name
        self.assertEqual(len(back), MAX_NAME)
        self.assertEqual(back, "a-very-long-pan")

    def test_enabled_is_written_the_way_the_layer_writes_it(self):
        write_quad(QuadConfig(3, enabled=True), self.path)

        with open(self.path, encoding="utf-8-sig") as f:
            self.assertIn("Quad3_Enabled = true", f.read())

        self.assertTrue(read_quads(self.path)[3].enabled)

    def test_write_keys_creates_a_missing_section(self):
        fd, path = tempfile.mkstemp(suffix=".ini")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("[Main]\nProjectionMode = 0\n")

            write_keys({"Quad0_PosX": "1.5"}, path)

            self.assertAlmostEqual(read_quads(path)[0].position[0], 1.5, places=5)
        finally:
            os.unlink(path)

    def test_write_keys_handles_a_file_with_no_final_newline(self):
        """Otherwise the appended key gets glued onto the last line and vanishes."""
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("[Quads]\nQuadSubdivisions = 1")

        write_keys({"Quad0_PosY": "2.25"}, self.path)

        self.assertAlmostEqual(read_quads(self.path)[0].position[1], 2.25, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
