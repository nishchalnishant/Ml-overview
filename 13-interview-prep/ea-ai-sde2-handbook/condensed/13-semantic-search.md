# Interview 13 — Semantic Search for Game Asset Discovery (Condensed)

Design multimodal search over 10M+ EA game assets (3D models, textures, audio) so artists can query by text ("rusty sci-fi container") or by uploaded image, instead of relying on exact filenames/manual tags.

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

## Biggest Pitfall
Treating this as a single-modality text-search problem (e.g., running BERT over filenames) instead of recognizing it needs a shared text-image embedding space — failing to connect text and image modalities is the fastest way to a No Hire.
