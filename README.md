# mr-cockpit-steamvr

Calibration, markers and cutout tooling for mixed-reality flight simulation with a
Bigscreen Beyond 2e, an ELP dual-lens camera, and
[openxr-steamvr-passthrough](https://github.com/Rectus/openxr-steamvr-passthrough).

**Start with [docs/PLAN.md](docs/PLAN.md)** - the goal, the design and its rationale, the
milestones, and the decisions log.

## Documents

| | |
|---|---|
| [docs/PLAN.md](docs/PLAN.md) | the plan, design decisions, performance budget, open questions |
| [docs/testing.md](docs/testing.md) | both test suites and what they protect |
| [docs/tracing.md](docs/tracing.md) | tracing a cutout outline from a camera frame |
| [docs/anchoring-config.md](docs/anchoring-config.md) | marker and anchor configuration design |
| [docs/marker-size-measurements.md](docs/marker-size-measurements.md) | measured detection rates, pose jitter, marker layout |
| [docs/m03-world-quads.md](docs/m03-world-quads.md) | world-anchored quads, DX11 and the menu |
| [docs/m04-vulkan.md](docs/m04-vulkan.md) | Vulkan parity |
| [docs/m05-polygon-cutouts.md](docs/m05-polygon-cutouts.md) | arbitrary polygon cutouts |

## Setup

`uv` keeps its Python and cache inside the project (`.uvpython/`, `.uvcache/`), because
this machine's shells run in an MSIX container where AppData writes are virtualised and
break the default locations:

    set UV_PYTHON_INSTALL_DIR=%CD%\.uvpython
    set UV_CACHE_DIR=%CD%\.uvcache
    uv venv --python 3.12 .venv
    uv pip install --python .venv\Scripts\python.exe numpy opencv-contrib-python pillow openvr

## Tests

    .venv\Scripts\python.exe -m unittest discover -s tests      # 73, no hardware
    cd ..\rectus\src && tests\run_tests.bat                     # 54 checks, C++ mesh

## Layout

    tracing/        geometry, capture and config I/O for the tracing tool (unit tested)
    tests/          the Python suite
    scripts/        the live tools (see below)
    docs/           see above
    PRINT-THESE/    sheets ready to order or print
    archive/        superseded scripts and completed-experiment data
    config-backups/ the calibrated camera config

### scripts/

| | |
|---|---|
| `marker_ids.py` | id -> marker size allocation, the source of truth |
| `make_30mm.py`, `make_sized.py` | the sticker sheets in PRINT-THESE |
| `make_screen_test.py` | the on-screen size test |
| `make_plate.py` | sticker placement guides for panels that are not displays |
| `show_1to1.py` | display an image at exactly 1:1, DPI-scaling proof |
| `show_plate.py` | show markers on a USB display panel at a known physical size |
| `marker_test.py` | detection rate per marker size |
| `jitter_test2.py` | pose jitter, single marker vs constellation |
| `trace_cutout.py` | trace a cutout outline from a camera frame |
| `cam.py`, `devlist.py`, `dpi.py` | shared helpers |

`archive/superseded-scripts/` holds earlier sheet generators, the redundant Python
calibration tooling (the release ships `camera-calibration.exe`), and the tools for
experiments that are finished and written up. Kept because they cannot be regenerated,
not because they should be run.
