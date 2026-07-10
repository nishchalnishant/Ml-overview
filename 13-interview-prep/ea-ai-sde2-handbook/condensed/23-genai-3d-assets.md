# Interview 23 — GenAI for 3D Asset Creation (Condensed)

**Problem:** Artists spend days modeling simple background props (crates, barrels, streetlamps). Design an internal tool where an artist types a prompt ("Rusty sci-fi crate") and gets an instant, game-ready textured 3D model usable in Frostbite.

## Clarifying Questions to Ask
- Geometry + textures, or textures onto an existing mesh? → Both; geometry must be low-poly with valid UVs for the game engine.
- Can artists supply a rough grey-box block-out to constrain size/silhouette? → Yes, and it should be used to control generation.
- Any model restrictions? → Any open-source foundation model (Stable Diffusion, ControlNet, Objaverse-trained models) is fair game.
- Topology requirement — clean quads/point clouds okay, or does the engine need decimated tri meshes with UVs? → Must be game-ready (bounded poly count, watertight, UV-mapped).
- Where does this run — artist's local GPU or a cloud service? → Cloud GPU microservice, async job model.

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
- Names the "Janus Problem" (duplicate/hallucinated faces on unseen backside) unprompted as the classic Text-to-3D failure mode, and proposes multi-view diffusion (MVDream/SyncDreamer) as the fix.
- Flags that raw marching-cubes output is unusable without decimation + UV unwrap — treats mesh post-processing as mandatory, not optional polish.
- Proactively raises PBR material generation (normal/roughness/metallic maps), not just an albedo texture, since flat color textures look wrong under game engine lighting.
- Mentions floating/disconnected geometry as an artifact and proposes keeping only the largest connected component.
- Proposes an in-editor plugin (Maya/Blender/Frostbite) as the real end-state UX, not a standalone web tool.
- Suggests LoRA fine-tuning on the game's specific art style so generated assets stay visually consistent with the rest of the title.
- Catches that 512x512 SD output looks blurry stretched over a mesh — inserts an upscaler (Real-ESRGAN) before texture baking.

## Top 3 Tradeoffs
- **NeRF/Gaussian Splat vs polygonal mesh** — NeRFs look photoreal instantly but can't be rigged, animated, or lit by standard game engines; must commit to explicit polygons via marching cubes despite uglier raw output.
- **SDS optimization (DreamFusion) vs feed-forward (TripoSR)** — SDS gives high detail but takes 1-2 hours/asset; feed-forward takes seconds with less detail. For an interactive artist tool, feed-forward wins decisively.
- **Full generative pipeline vs procedural/texture-only** — Houdini+LLM-driven procedural generation guarantees perfect topology but limits creativity; a texture-only pipeline (paint onto pre-made meshes) eliminates topology risk entirely but caps geometric novelty.

## Toughest Follow-ups
**Q: Marching cubes gives jagged triangles causing bad shadows — how do you get clean quads?**
Use an auto-retopology algorithm (QuadriFlow or Instant Meshes): align a vector field to the mesh's principal curvatures, then extract a clean low-poly quad mesh from that field. This is a distinct post-processing step from decimation.

**Q: A generated "Human Soldier" can't move — no skeletal rig. How do you automate rigging?**
Run pose estimation (MediaPipe-style) to find joint locations on the mesh, align a standard humanoid skeleton to those joints, then compute skinning weights via heat diffusion / geodesic voxel binding so the mesh deforms smoothly.

**Q: Need 10,000 unique trees; pipeline takes 15s/asset — how do you scale?**
Generative AI is the wrong tool here — too slow and prone to hallucinated topology. Use procedural L-systems or SpeedTree for geometry (milliseconds each), and reserve the ML pipeline only for generating a texture atlas (bark/leaf textures) applied across all procedural instances.

## Biggest Pitfall
Proposing to hand a NeRF or Gaussian Splat directly to the Frostbite engine (or otherwise not distinguishing implicit 3D representations from game-ready polygon meshes) — this alone signals the candidate doesn't understand the production constraints and is an instant No-Hire/Lean No-Hire signal.
