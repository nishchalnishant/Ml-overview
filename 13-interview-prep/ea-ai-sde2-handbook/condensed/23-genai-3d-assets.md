# Interview 23 — GenAI for 3D Asset Creation (Condensed)

**Problem:** Artists spend days modeling simple background props (crates, barrels, streetlamps). Design an internal tool where an artist types a prompt ("Rusty sci-fi crate") and gets an instant, game-ready textured 3D model usable in Frostbite.

## Core Architecture
```
Grey-box mesh + prompt → depth map render
   → Stable Diffusion + ControlNet(Depth)  [2D concept image, constrained to block-out]
   → Feed-forward Large Reconstruction Model (TripoSR/LRM)  [2D image → raw 3D mesh via triplane + marching cubes]
   → Post-process: decimate (500k→5k polys) → UV unwrap (xatlas) → bake texture
   → Export .glb/.fbx → Frostbite
```
- **ControlNet(Depth/Canny)** is the key technique: forces 2D generation to respect the artist's spatial/silhouette constraints — plain text-to-image would ignore gameplay bounds.
- **Feed-forward reconstruction (TripoSR/LRM)** chosen over NeRF/Gaussian Splatting/SDS optimization because game engines need explicit polygons, and speed (~seconds, not hours) matters for an artist-facing tool.
- Async job queue (SQS) + polling UI since generation takes 15-30s — never block on a synchronous HTTP call.

## Talking Points That Signal Seniority
- Explicitly rules out NeRFs/Gaussian Splatting as final output — great for rendering, unusable for rigging/animation/standard lighting pipelines.
- Names duplicate/hallucinated faces on the unseen backside unprompted as the classic Text-to-3D failure mode, and proposes multi-view diffusion as the fix.
- Flags that raw marching-cubes output is unusable without decimation + UV unwrap — treats mesh post-processing as mandatory, not optional polish.
- Proactively raises PBR material generation (normal/roughness/metallic maps), not just an albedo texture, since flat color textures look wrong under game engine lighting.
- Mentions floating/disconnected geometry as an artifact and proposes keeping only the largest connected component.
- Proposes an in-editor plugin (Maya/Blender/Frostbite) as the real end-state UX, not a standalone web tool.
- Suggests LoRA fine-tuning on the game's specific art style so generated assets stay visually consistent with the rest of the title.
- Catches that 512x512 SD output looks blurry stretched over a mesh — inserts an upscaler (Real-ESRGAN) before texture baking.

## Top 3 Tradeoffs
- **NeRF/Gaussian Splat vs polygonal mesh** — NeRFs look photoreal instantly but can't be rigged, animated, or lit by standard game engines; must commit to explicit polygons via marching cubes despite uglier raw output.
- **Optimization-based vs feed-forward (TripoSR)** — optimization-based generation gives high detail but takes 1-2 hours/asset; feed-forward takes seconds with less detail. For an interactive artist tool, feed-forward wins decisively.
- **Full generative pipeline vs procedural/texture-only** — Houdini+LLM-driven procedural generation guarantees perfect topology but limits creativity; a texture-only pipeline (paint onto pre-made meshes) eliminates topology risk entirely but caps geometric novelty.

## Biggest Pitfall
Proposing to hand a NeRF or Gaussian Splat directly to the Frostbite engine (or otherwise not distinguishing implicit 3D representations from game-ready polygon meshes) — this alone signals the candidate doesn't understand the production constraints and is an instant No-Hire/Lean No-Hire signal.
