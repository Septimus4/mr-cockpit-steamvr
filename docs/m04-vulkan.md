# M04: Vulkan parity for world quads

Status: **implemented, builds clean, untested on hardware.**

Matters because X-Plane 12 can run Vulkan, where quad mode previously did nothing at all.

## The problem

The DX11 path reuses `mesh_rigid_vs.hlsl` - a vertex shader that transforms geometry by
a per-quad mesh-to-world matrix. Vulkan has no equivalent:

- `mesh_rigid_vs` is not compiled to SPIR-V (`prebuild_commands.bat` builds only
  `fullscreen_quad_vs`, `passthrough_vs`, `passthrough_stereo_vs`)
- there is no render-model path in the Vulkan renderer to borrow from
- the descriptor set layout has no spare vertex-stage uniform binding
- all pipelines share one layout and differ mainly by blend state, so adding a shader
  would mean duplicating every blend-mode pipeline

## The approach: invert the mapping on the CPU

`passthrough_vs.hlsl` computes world position from vertex position using only PER-VIEW
uniforms, every one of which the CPU sets:

```hlsl
worldProjectionPos.xz = inPosition.xz * g_projectionDistance + g_projectionOriginWorld.xz;
worldProjectionPos.y  = inPosition.y  * scale                + offset;
```

So the CPU can invert it and hand the shader vertices that emerge at exactly the world
positions wanted. `ProjectionSpaceMapping` in `passthrough_renderer.h` does this.

Consequences, all good:

- **no new shader, no new pipeline, no descriptor change**
- quads inherit every existing blend mode automatically
- quads inherit the Vulkan-specific depth/Y fixups in `passthrough_vs`
- all enabled quads pre-transform into ONE vertex buffer, so they draw in ONE call with
  no per-quad uniform binding at all - which is what made per-draw data awkward in
  Vulkan in the first place

Cost: a CPU transform of a few hundred vertices per view per frame. Negligible.

The trade is that `ProjectionSpaceMapping` must stay in step with `passthrough_vs.hlsl`.
If that mapping changes, this breaks **silently** - quads would simply be drawn in the
wrong place, with no error. That is called out in the code comment.

## Changes

| file | change |
|------|--------|
| `passthrough_renderer.h` | `ProjectionSpaceMapping`, Vulkan quad members, two decls |
| `passthrough_renderer_vulkan.cpp` | `GenerateQuadMesh()`, `BuildQuadBatch()`, draw hooks |

Same two hooks as DX11: the main view draw, and the alpha prepass when `QuadsExclusive`.

## Two bugs caught during the work

**Scoping.** `vsViewBuffer` and `vertOffset` live inside the `if (BlendMode != Masked)`
block, but the draws happen after it closes. Hoisted.

**Masked mode would have placed quads completely wrong.** The first fix captured the
projection mapping from the view constant buffer as it was filled - but in Masked blend
mode that buffer is filled by `RenderMaskedPrepassView`, a different function entirely.
The capture would never run, the mapping would keep its defaults, and quads would be
drawn nowhere near the right place in exactly that mode. The mapping is now computed
directly from `renderParams` and the HMD pose, which works in both paths.

That one would have compiled, run, and produced a plausible-looking wrong result.

## Known limitations

- **Untested on hardware.** Compiles and the logic is reviewed, nothing more.
- The DX11 and Vulkan paths reach the same result by different means - DX11 by a
  per-quad matrix in a shader, Vulkan by CPU pre-transform. Deliberate, but it is two
  code paths to keep in agreement.
- Masked and AlphaTest prepasses are still not quad-aware on either backend.
