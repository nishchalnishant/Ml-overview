# Interview 13 — Semantic Search for Game Asset Discovery (Condensed)

Design multimodal search over 10M+ EA game assets (3D models, textures, audio) so artists can query by text ("rusty sci-fi container") or by uploaded image, instead of relying on exact filenames/manual tags.

## Clarifying Questions to Ask
- What modalities? → Text-to-Image, Image-to-Image, Text-to-3D.
- How are 3D models represented? → Raw `.fbx`/`.obj` + auto-generated 4 isometric 2D thumbnail renders per model.
- Pretrained or custom model? → Can use open-source (CLIP), but game-specific terms ("Zerg", "N7 Armor") may need fine-tuning.
- Scale + latency SLA? → 10M assets, search <500ms, indexing can be async batch.
- Is this web app or embedded in engine editor? (affects auth/UI integration)
- Indexing frequency for new daily assets? (affects streaming vs batch design)

## Core Architecture
- **Embedding model:** CLIP/OpenCLIP (ViT-B/32 or ViT-H-14) — maps text and images into one shared vector space, enabling zero-shot text↔image search with no manual tagging.
- **3D handling:** embed the 4 isometric thumbnail renders per asset (not the raw mesh) since no production-ready 3D-native embedding model exists.
- **Vector DB:** Qdrant/Milvus/FAISS with HNSW index for sub-ms ANN search at 10M+ scale (brute force is O(N), too slow).
- **Offline pipeline:** Spark/Ray batch job embeds assets at scale on GPU; Kafka + serverless GPU function embeds new daily assets near-real-time.
- **Serving:** text-embedding endpoint on CPU pods (cheap), image-embedding endpoint on GPU pods (T4) if traffic is high.
- **Metadata filtering:** project/team tags stored alongside vectors for filtered search (e.g., `project=battlefield_2042`).

```
Asset → CLIP Image Encoder → vector → Qdrant (HNSW)
Query text/image → CLIP Encoder → vector → cosine search → top-20 asset URIs
```

## Talking Points That Signal Seniority
- Proactively flags that 3D meshes can't go directly into CLIP — need 2D renders as a bridge, and specifically that camera angle (isometric, not top-down) matters for correct semantics.
- Names HNSW explicitly and states its recall/speed tradeoff vs brute-force exact search.
- Raises the "vocabulary gap" unprompted: CLIP knows "barrel" but not "Needler" or franchise-specific terms — proposes hybrid search (BM25 + dense) as the fix.
- Mentions vector quantization (INT8/PQ) or DiskANN for RAM cost control before being asked about scaling.
- Suggests fine-tuning CLIP via contrastive/triplet loss on internal EA concept-art data rather than training a CNN from scratch.
- Proposes personalization/re-ranking by studio/project affinity (e.g., boost "Dead Space" horror assets for that team) as a production improvement.
- Suggests using CLIP zero-shot classification to auto-tag legacy SQL metadata — reusing existing infra instead of building new models.
- Calls out that L2-normalized vectors are required for cosine similarity correctness in the vector DB config.

## Top 3 Tradeoffs
- **CLIP (zero-shot) vs custom CNN/ResNet:** CLIP works out-of-the-box with no labeled data; a custom CNN could be marginally better at niche art styles but requires manually tagging 10M assets first — not practical.
- **2D thumbnails vs native 3D embeddings (PointNet/3D-CNN):** thumbnails let you reuse mature 2D vision models; native 3D embedding is still experimental, slower, and harder to train — thumbnails win for time-to-ship.
- **FP32 vs INT8 quantized vectors:** FP32 is exact but 4x the RAM; for top-20 retrieval, exact ranking doesn't matter much, so INT8/PQ is the obvious cost-saving choice at 10M+ scale.

## Toughest Follow-ups
**Q: How do you cut HNSW index memory by 80%+ at 10M-100M scale?**
A: Stack techniques — Scalar Quantization (FP32→INT8, 4x reduction), then Product Quantization to replace sub-vectors with codebook centroid IDs, and move raw vectors to disk (DiskANN-style) keeping only the graph structure in RAM.

**Q: An artist uploads a hand-drawn sketch of a sword — base CLIP fails because sketches and photorealistic renders land in different regions of the embedding space. Fix it?**
A: Don't rely on base CLIP for this domain gap. Insert a small adapter/MLP on top of the CLIP image encoder and train it with Triplet Loss (anchor=sketch, positive=render of same object, negative=render of different object) to pull same-object sketch/render pairs together in vector space.

**Q: Use the existing CLIP architecture to auto-tag all 10M assets into a legacy SQL schema, without training new models?**
A: Zero-shot classification — precompute text embeddings for a fixed tag vocabulary (~1,000 tags), then dot-product each asset's image embedding against all tag embeddings and assign tags above a similarity threshold (e.g., >0.85).

## Biggest Pitfall
Treating this as a single-modality text-search problem (e.g., running BERT over filenames) instead of recognizing it needs a shared text-image embedding space — failing to connect text and image modalities is the fastest way to a No Hire.
