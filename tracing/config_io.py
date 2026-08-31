"""
Reading and writing cutout data in the layer's config.ini.

The format here MUST match Config_QuadShape::ParseConfig and Config_Quad::ParseConfig in
shared/config_manager.h. A mismatch is silent - the layer would fall back to a rectangle,
or read a subtly different outline, with nothing to indicate why.

The parse/format pair is separated from all file access so it can be unit tested
directly, which is the whole point: this is the one place a formatting slip would be
invisible until something looked wrong in the headset.
"""

import codecs
import os
import pathlib
import re

# Must equal MAX_QUAD_POLYGON_POINTS in shared/config_manager.h.
MAX_POINTS = 32

# Must equal MAX_PASSTHROUGH_QUADS in shared/config_manager.h.
MAX_QUADS = 8

DEFAULT_CONFIG_PATH = os.path.join(
    os.environ.get("APPDATA", ""), "OpenXR SteamVR Passthrough", "config.ini")

_PAIR = re.compile(r"\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
                   r"\s*,\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def format_points(points):
    """
    Outline -> ini value. Five decimals is 10 micrometres, far finer than anything the
    alignment budget cares about and short enough to stay readable.

    Fewer than three points formats as empty, matching the C++ side's rule that such an
    outline means "use the rectangle".
    """
    if len(points) < 3:
        return ""

    return ";".join(f"{float(x):.5f},{float(y):.5f}" for x, y in points[:MAX_POINTS])


def parse_points(value):
    """
    ini value -> outline. Mirrors the C++ parser, including its two guards: stop at
    MAX_POINTS, and treat fewer than three points as no outline at all.
    """
    if not value:
        return []

    points = []
    pos = 0

    while pos < len(value) and len(points) < MAX_POINTS:
        m = _PAIR.match(value, pos)

        if m is None:
            break

        points.append((float(m.group(1)), float(m.group(2))))
        pos = m.end()

        while pos < len(value) and value[pos] in "; ":
            pos += 1

    return points if len(points) >= 3 else []


class QuadConfig:
    """One cutout, as far as the tracing tool is concerned."""

    def __init__(self, index, enabled=False, name="", position=(0.0, 0.0, -0.7),
                 euler_deg=(0.0, 0.0, 0.0), width=0.30, height=0.20, points=None):
        self.index = index
        self.enabled = enabled
        self.name = name
        self.position = tuple(float(v) for v in position)
        self.euler_deg = tuple(float(v) for v in euler_deg)
        self.width = float(width)
        self.height = float(height)
        self.points = list(points or [])

    @property
    def label(self):
        return self.name or f"Quad {self.index}"

    def __repr__(self):
        return (f"QuadConfig({self.index}, {self.label!r}, enabled={self.enabled}, "
                f"pos={self.position}, rot={self.euler_deg}, points={len(self.points)})")


def _read_ini(path):
    """
    Minimal ini reader.

    configparser is not used on purpose: it lowercases keys by default, rewrites the file
    wholesale, and would drop the layer's comments and ordering. This tool must edit a
    file another program owns, so it reads what it needs and writes back only the lines
    it changes.
    """
    sections = {}
    current = None

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            s = line.strip()

            if not s or s.startswith((";", "#")):
                continue

            if s.startswith("[") and s.endswith("]"):
                current = s[1:-1]
                sections.setdefault(current, {})
                continue

            if current is None or "=" not in s:
                continue

            k, _, v = s.partition("=")
            sections[current][k.strip()] = v.strip()

    return sections


def read_quads(path=None):
    """Load every cutout from the config. Missing keys fall back to the C++ defaults."""
    path = path or DEFAULT_CONFIG_PATH
    sections = _read_ini(path)
    quads_section = sections.get("Quads", {})

    def get(key, default, cast=float):
        raw = quads_section.get(key)
        if raw is None:
            return default
        try:
            return cast(raw)
        except ValueError:
            return default

    def get_bool(key, default):
        raw = quads_section.get(key)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    quads = []

    for i in range(MAX_QUADS):
        quads.append(QuadConfig(
            index=i,
            enabled=get_bool(f"Quad{i}_Enabled", False),
            name=quads_section.get(f"Quad{i}_Name", ""),
            position=(get(f"Quad{i}_PosX", 0.0), get(f"Quad{i}_PosY", 0.0),
                      get(f"Quad{i}_PosZ", -0.7)),
            euler_deg=(get(f"Quad{i}_RotX", 0.0), get(f"Quad{i}_RotY", 0.0),
                       get(f"Quad{i}_RotZ", 0.0)),
            width=get(f"Quad{i}_Width", 0.30),
            height=get(f"Quad{i}_Height", 0.20),
            points=parse_points(quads_section.get(f"Quad{i}_Points", "")),
        ))

    return quads


def write_points(index, points, path=None):
    """
    Write one cutout's outline, leaving the rest of the file exactly as it was.

    Returns the value written. Rewriting only the single line matters because the layer
    and the settings menu both own this file; a wholesale rewrite would drop anything
    this tool does not model.
    """
    path = path or DEFAULT_CONFIG_PATH
    key = f"Quad{index}_Points"
    value = format_points(points)

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    # The layer writes this file with a UTF-8 BOM. utf-8-sig strips it on read, so it
    # must be written back or the layer sees a file that no longer starts as it expects.
    had_bom = pathlib.Path(path).read_bytes().startswith(codecs.BOM_UTF8)

    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"

    in_quads = False
    quads_start = None
    quads_end = len(lines)
    replaced = False

    for i, line in enumerate(lines):
        s = line.strip()

        if s.startswith("[") and s.endswith("]"):
            if in_quads:
                quads_end = i
                break
            in_quads = (s == "[Quads]")
            if in_quads:
                quads_start = i
            continue

        if in_quads and s.startswith(key) and "=" in s and s.split("=")[0].strip() == key:
            lines[i] = f"{key} = {value}{newline}"
            replaced = True
            break

    if not replaced:
        if quads_start is None:
            lines.append(f"{newline}[Quads]{newline}")
            lines.append(f"{key} = {value}{newline}")
        else:
            lines.insert(quads_end, f"{key} = {value}{newline}")

    with open(path, "w", encoding="utf-8-sig" if had_bom else "utf-8", newline="") as f:
        f.writelines(lines)

    return value
