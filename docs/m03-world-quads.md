# M03: world-anchored passthrough quads

Status: **implemented for DX11 with menu UI, builds clean, untested on hardware.**

## What it does

Adds `ProjectionMode = 3` (`Projection_WorldQuad`). The quads are an ALIGNMENT surface,
not a mask.

Passthrough still covers the whole view, and what is VISIBLE is still decided by the
app's cutout - a chroma-key livery, or the layer alpha. What the quads change is the
projection GEOMETRY: where a quad sits, the passthrough image is projected onto that
quad's plane rather than onto the cylinder at `ProjectionDistanceFar`.

That is the whole point. A cylinder metres away cannot align with a panel 0.7 m from
your face, however its distance is tuned - the parallax is wrong at every setting. Put a
quad where the real panel is and the image lands at the panel's true depth and angle, so
what shows through the cutout matches the hardware.

`QuadsExclusive = true` switches to the other behaviour: passthrough only inside the
quads, cylinder not drawn. That suits a setup with no in-game cutout to mask against.

## The shader already existed

The original plan was to modify `passthrough_vs.hlsl`, replacing its cylinder scaling
with a quad-to-world transform. That turned out to be unnecessary: `mesh_rigid_vs.hlsl`
already does exactly this job for tracked-device render models.

```hlsl
float4 worldPos = mul(g_meshToWorldTransform, float4(inPosition, 1.0));
float4 clipSpacePos = mul(g_worldToCameraFrameProjectionLeft/Right, worldPos);
output.cameraReprojectedPos = clipSpacePos;
output.position = mul(g_worldToHMDProjection, worldPos);
```

Arbitrary geometry placed by a world matrix, with camera reprojection applied - a
world-anchored quad needs nothing else. So M03 required NO shader changes and no new
constant buffer layout, which removes the whole class of HLSL/C++ packing mismatch bugs
that modifying `vsViewConstantBuffer` would have risked.

## Changes

| file | change |
|------|--------|
| `mesh.h` / `mesh.cpp` | `MeshCreateQuad()` - subdivided unit quad, double-sided |
| `shared_structs.h` | `Projection_WorldQuad = 3` |
| `shared/config_manager.h` | `Config_Quad`, `Config_Quads`, `MAX_PASSTHROUGH_QUADS 8` |
| `shared/config_manager.cpp` | parse/write the `[Quads]` section |
| `passthrough_renderer.h` | quad mesh members, `GetQuadToWorldTransform()` |
| `passthrough_renderer_dx11.cpp` | `GenerateQuadMesh()`, `RenderPassthroughQuads()`, hooks |

Originals preserved as `*.m03bak`.

## Two decisions worth knowing

**The quad is double-sided.** Back-face culling is on, so a single-sided quad would
vanish completely when a user happens to orient it away from the viewer - a silent
failure with no feedback, and easy to hit when placing quads by hand in an ini. Two
extra triangles per cell removes the failure mode.

**The quad is subdivided (default 8x8), not two triangles.** The pixel shader applies a
UV distortion map, and undistortion varies non-linearly across a large quad; two
triangles would interpolate it between three corners each.

## The alpha prepass had to be taught about quads too

`RenderAlphaPrepassView` establishes where passthrough is permitted, so it has to match
the geometry the main pass actually covers.

In EXCLUSIVE mode that is the quads, so it draws them. In alignment mode (the default)
the cylinder still covers the view and the app's cutout does the masking, so the prepass
is left exactly as it was. Getting this backwards would either authorise the whole view
in exclusive mode, defeating the cutout, or mask the view down to the quads in alignment
mode, which is precisely what alignment mode exists to avoid.

## Config

```ini
[Main]
ProjectionMode = 3

[Quads]
QuadSubdivisions = 8

Quad0_Enabled = true
Quad0_PosX    = 0.0      ; centre, stage-space metres
Quad0_PosY    = 1.0      ; height above the floor - adjust to your seated eye level
Quad0_PosZ    = -0.7     ; negative Z is forward
Quad0_RotX    = 0.0      ; intrinsic Euler XYZ, DEGREES
Quad0_RotY    = 0.0
Quad0_RotZ    = 0.0
Quad0_Width   = 0.30     ; physical size, metres
Quad0_Height  = 0.20

; QuadsExclusive = false   ; default: quads align, cutout decides visibility
;                          ; true: passthrough ONLY inside the quads
```

Up to 8 quads, `Quad0_` through `Quad7_`.

Rotation is Euler degrees rather than a quaternion because until the marker solver lands
these are placed by hand-editing the ini, and degrees are the only form a person can
reason about. When the solver writes poses back it must use the same convention.

## Verified on hardware 2026-08-31

- the layer loads; Custom2D is unchanged, proving the build before new code is exercised
- `QuadsExclusive` draws a world-anchored quad
- pose edits stream over IPC and move the quad immediately
- **alignment mode composites cleanly** - the cylinder and the quad overlapping produces
  no brightness or contrast artifact, so the depth pass held in reserve is not needed.
  This was flagged as a risk and is now closed.

Moving a quad in alignment mode makes it visibly disagree with the cylinder in parallax,
because the two sit at different depths. That is the effect the mode exists to correct,
not a defect.

**A quad being eaten from one side as the head turns is `ClampCameraFrame`,** discarding
pixels whose reprojection falls outside the camera image - not a bug. See PLAN.md.

## Known limitations

- **DX11 only.** The Vulkan renderer is untouched, so quad mode will not work there.
  DCS is DX11; X-Plane 12 can use Vulkan, so that path will need the same treatment.
- **Masked and AlphaTest prepasses are not quad-aware.** Only the alpha prepass was
  updated. Apps using those blend modes may show passthrough outside the quads.
- **No anchoring.** Quads sit at fixed stage-space poses. Marker-driven placement is
  M04+; this milestone is the rendering substrate it will drive.
- **Untested on hardware.** It compiles and the logic is reviewed, nothing more.


# Menu UI

A **Quads** tab (between Stereo and Debug) edits quads live, so they can be nudged while
wearing the headset rather than by hand-editing an ini.

| file | change |
|------|--------|
| `settings_menu.h` | `TabQuads` |
| `settings_menu.cpp` | tab button, `Cockpit Quads` projection radio (both layouts), IPC dispatch |
| `quads_tab.inl` | the tab body, included from `DrawMenu` |
| `shared/menu_ipc.h` | `MessageType_SendConfig_Quads` |
| `menu_handler.cpp` | layer-side handler |

Per quad: enable, position XYZ, rotation XYZ in degrees, width and height, plus
`Place 1 m Ahead` and `Reset Size`. The tab warns when the projection mode is not set to
Cockpit Quads and offers a button to switch, and warns when no quads are enabled - both
states otherwise look identical to "it is broken".

`quads_tab.inl` is a separate file purely to avoid adding 150 lines to an already
3400-line `DrawMenu`; it is `#include`d and shares that function's locals.

## Two notes

**No "place at my head pose" button.** The obvious feature - drop the quad where the user
is looking - is wrong here: the menu is a separate process and its OpenVR pose is the
dashboard's, not the HMD's while the user is in the app. A fixed, predictable 1 m-ahead
starting point that the user then nudges is more honest than a pose that looks
authoritative and is subtly wrong.

**The IPC payload is checked at compile time.** `Config_Quads` is 292 bytes against a
496-byte payload, but eight quads is a number that invites raising. A `static_assert`
guards it, because overflowing would silently truncate quad poses rather than fail.
