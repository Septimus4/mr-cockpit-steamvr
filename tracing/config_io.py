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

# Config_Quad::Name is char[16] and is read with strncpy_s(_TRUNCATE), so 15 characters
# survive. Truncating here rather than letting the layer do it silently keeps what is
# written the same as what is read back.
MAX_NAME = 15

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


def write_keys(values, path=None, section="Quads"):
    """
    Set several keys in one section, leaving every other line exactly as it was.

    Rewriting only the lines that change matters because the layer and the settings menu
    both own this file: a wholesale rewrite would drop everything this tool does not
    model. Keys that are absent are appended to the section rather than to the file, or
    the layer's parser would never reach them.
    """
    path = path or DEFAULT_CONFIG_PATH
    pending = dict(values)

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    # The layer writes this file with a UTF-8 BOM. utf-8-sig strips it on read, so it
    # must be written back or the layer sees a file that no longer starts as it expects.
    had_bom = pathlib.Path(path).read_bytes().startswith(codecs.BOM_UTF8)

    newline = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"

    in_section = False
    section_start = None
    section_end = len(lines)

    for i, line in enumerate(lines):
        s = line.strip()

        if s.startswith("[") and s.endswith("]"):
            if in_section:
                section_end = i
                in_section = False
                break

            in_section = (s == f"[{section}]")

            if in_section:
                section_start = i

            continue

        if not in_section or "=" not in s:
            continue

        key = s.split("=")[0].strip()

        if key in pending:
            lines[i] = f"{key} = {pending.pop(key)}{newline}"

    if pending:
        # A file whose last line has no newline would otherwise get a key glued onto it.
        if lines and not lines[-1].endswith(("\n", "\r")):
            lines[-1] += newline

        if section_start is None:
            lines.append(f"{newline}[{section}]{newline}")
            section_end = len(lines)

        for key in list(pending):
            lines.insert(section_end, f"{key} = {pending.pop(key)}{newline}")
            section_end += 1

    with open(path, "w", encoding="utf-8-sig" if had_bom else "utf-8", newline="") as f:
        f.writelines(lines)


def write_points(index, points, path=None):
    """
    Write one cutout's outline, leaving the rest of the file exactly as it was.

    Returns the value written.
    """
    value = format_points(points)
    write_keys({f"Quad{index}_Points": value}, path)

    return value


def write_quad(quad, path=None):
    """
    Write one whole cutout - pose, size, name, enabled state and outline.

    This is how a solved anchor becomes a cutout. Every field is written together because
    a half-written cutout is worse than none: a new pose with a stale size lands in the
    right place at the wrong scale, which reads as bad tracking rather than a partial
    write.
    """
    i = quad.index

    write_keys({
        # "true"/"false" rather than 1/0: the parser takes either, but the layer writes
        # words, and matching it keeps the file from churning every time it saves.
        f"Quad{i}_Enabled": "true" if quad.enabled else "false",
        f"Quad{i}_Name": quad.name[:MAX_NAME],
        f"Quad{i}_PosX": f"{quad.position[0]:.5f}",
        f"Quad{i}_PosY": f"{quad.position[1]:.5f}",
        f"Quad{i}_PosZ": f"{quad.position[2]:.5f}",
        f"Quad{i}_RotX": f"{quad.euler_deg[0]:.3f}",
        f"Quad{i}_RotY": f"{quad.euler_deg[1]:.3f}",
        f"Quad{i}_RotZ": f"{quad.euler_deg[2]:.3f}",
        f"Quad{i}_Width": f"{quad.width:.5f}",
        f"Quad{i}_Height": f"{quad.height:.5f}",
        f"Quad{i}_Points": format_points(quad.points),
    }, path)
