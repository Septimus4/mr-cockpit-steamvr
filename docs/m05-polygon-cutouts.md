# M05: arbitrary polygon cutouts

Status: **plumbed end to end, builds clean, untested on hardware. No authoring UI yet.**

A cutout is now **a plane plus a 2D outline in that plane**. Rectangles remain the
fallback, so every existing quad keeps working untouched.

## Changes

| file | change |
|------|--------|
| `mesh.h` / `mesh.cpp` | `MeshCreatePolygon()` (ear clipping), `MeshCreateQuadSized()` |
| `shared/config_manager.h` | `Config_QuadShape`, `Config_QuadShapes`, `MAX_QUAD_POLYGON_POINTS 32` |
| `shared/config_manager.cpp` | parse/write outlines in the `[Quads]` section |
| `passthrough_renderer.h` | pose-only transform, `BuildQuadMesh()`, `QuadMeshNeedsRebuild()` |
| `passthrough_renderer_dx11.cpp` | per-cutout meshes and buffers, `UpdateQuadMeshes()` |
| `passthrough_renderer_vulkan.cpp` | per-cutout meshes, growable batch buffers |
| `passthrough-menu/quads_tab.inl` | outline status, Clear Outline, size sliders disabled |
| `passthrough-menu/settings_menu.cpp` | shape changes go by file + reload, not IPC |

## Ear clipping, not a triangle fan

Console outlines are **concave**. A fan from one vertex emits triangles outside the shape
wherever the outline turns inward, so ear clipping is required.

The algorithm was validated before wiring it in, by porting it and checking that the
triangle areas sum exactly to the polygon area (which catches overlaps, gaps, and
triangles outside the shape):

| case | triangles | expected | area match |
|------|-----------|----------|------------|
| square, CCW | 2 | 2 | exact |
| square, CW | 2 | 2 | exact |
| L-shape, 1 reflex | 4 | 4 | exact |
| U-shape, 2 reflex | 6 | 6 | exact |
| console-like, 7 points | 5 | 5 | exact |
| collinear points | rejected | - | - |
| fewer than 3 points | rejected | - | - |

Degenerate input is rejected rather than producing a broken mesh, and `BuildQuadMesh`
then falls back to the rectangle - so a bad outline degrades to a rectangle instead of
the cutout vanishing, which would look like the feature being broken.

## The transform is now pose-only

Previously the mesh was a unit quad and `Width`/`Height` were the transform's scale. That
cannot work for polygons: their points are already in metres, and scaling them by
Width/Height would silently distort a traced outline.

So the mesh carries its true size (`MeshCreateQuadSized` for rectangles) and the
mesh-to-world transform is pose only. Rectangles and polygons are then identical
downstream.

## Outlines travel by file, not IPC

`Config_Quads` is memcpy'd into a 496-byte IPC message and is already 424 bytes. Thirty-two
points would add 256 bytes **per cutout**.

So outlines live only in the config file, in a separate `Config_QuadShapes` that is never
sent over IPC. The menu writes the file and sends the existing
`MessageType_InformReloadConfigFile`; the layer re-reads. Poses keep flowing over IPC for
live dragging, which is the part that must be immediate. This removed the chunked-transfer
work entirely.

Format, verified round-trip:

```ini
[Quads]
Quad0_Points = 0.00000,0.00000;0.50000,0.02000;0.52000,0.10000;0.48000,0.18000
```

Metres, in the cutout's own plane, origin at the cutout's pose. Fewer than 3 points means
"use the rectangle".

## Mesh rebuilds exclude pose, deliberately

`QuadMeshNeedsRebuild` compares width, height, subdivisions and the outline - **not** the
pose. Pose changes every frame while a slider is dragged, and the mesh does not depend on
it; including it would recreate GPU buffers continuously during placement.

## What is NOT done

**There is no way to draw an outline yet.** The only route today is hand-editing
`Quad*_Points` in the ini. The tracing tool - capture a camera frame, click points,
back-project onto the cutout's plane - is the remaining M05 work.

Also outstanding, unchanged from M03/M04:

- untested on hardware
- Masked and AlphaTest prepasses are still not cutout-aware on either backend
