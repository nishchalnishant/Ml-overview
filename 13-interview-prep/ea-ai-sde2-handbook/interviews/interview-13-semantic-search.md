# Interview 13 — Semantic Search for Game Asset Discovery
**EA SDE-2 AI Engineer · Estimated Duration: 75 minutes**

---

## Part 1 — Problem Statement

You are an AI Engineer on the Frostbite Engine tooling team. EA studios (DICE, BioWare, Motive) share a massive central repository of 10+ million game assets (3D models, textures, audio clips). Currently, developers can only find assets by exact filename or manually entered tags (e.g., `prop_barrel_metal_01.fbx`).

Your task is to **design a multimodal Semantic Search engine** so an artist can type "rusty sci-fi container" or upload an image of a barrel, and the system instantly returns relevant 3D models and textures.

---

## Part 2 — Intentionally Missing Information

The following critical details are **deliberately omitted**. A strong candidate will ask about all of them:

- Multimodal requirements (Text-to-Image? Image-to-Image? Text-to-3D?)
- How do we generate embeddings for 3D models? (We can't just pass a `.fbx` file into CLIP).
- Data volume and Indexing frequency.
- User Interface integration (Is this a web app or integrated into the game engine editor?)
- Performance SLA (How fast should a search take?)

---

## Part 3 — Ideal Clarifying Questions

> Interviewer will reveal answers only when directly asked.

1. **"What modalities are we supporting?"**
   → *Answer: Text-to-Texture (Image), Image-to-Texture, and Text-to-3D Model.*

2. **"How are 3D models represented in our data warehouse?"**
   → *Answer: We have the raw `.fbx`/`.obj` files, and our rendering pipeline automatically generates 4 isometric 2D thumbnail renders for every 3D model.*

3. **"Are we using pre-trained models or training our own?"**
   → *Answer: You can use open-source foundation models (like CLIP), but game assets look different from standard internet photos. You may need to adapt it.*

4. **"What is the scale and latency requirement?"**
   → *Answer: 10 million assets. Search must return in < 500ms. Indexing can be an asynchronous batch job.*

---

## Part 4 — Expected Assumptions

- **Architecture:** Multimodal embedding space using OpenAI CLIP (or OpenCLIP). Both text and images are mapped to the same vector space.
- **3D Handling:** Since 3D models have 2D thumbnail renders, we embed the 2D thumbnails to represent the 3D object.
- **Vector Search:** Approximate Nearest Neighbor (ANN) search using Milvus, Qdrant, or FAISS.

---

## Part 5 — High-Level Solution

```
  [Offline Indexing Pipeline]
  New Asset (Texture or 3D Render) 
       │
       ▼
  CLIP Image Encoder (e.g., ViT-B/32) ➔ Generates 512d Vector
       │
       ▼
  Vector Database (Milvus/FAISS)
  (Stores Vector + Metadata: {asset_id, project, file_path})

       =========================================================

  [Real-Time Search API]
  Artist types: "Rusty sci-fi barrel"
       │
       ▼
  CLIP Text Encoder ➔ Generates 512d Vector
       │
       ▼
  Vector Search (Cosine Similarity) in Database
       │
       ▼
  Return Top 20 Asset URIs to Frostbite Editor UI
```

**Core ML Component:** A shared embedding space (CLIP) where the cosine distance between the text vector for "rusty barrel" and the image vector of a rusty barrel is minimized.

---

## Part 6 — Step-by-Step Implementation

### Step 1: Handling Modalities
- **Textures (.png/.dds):** Convert to standard RGB image, pass through CLIP Image Encoder.
- **3D Models (.fbx):** Take the 4 isometric thumbnail renders. Embed all 4. Store them in the Vector DB linked to the same `asset_id`. If any of the 4 angles match the text query, the 3D model is returned.

### Step 2: The Embedding Model
- Use `OpenCLIP` (e.g., ViT-H-14). 
- *Crucial detail:* CLIP is trained on internet data. It knows what a "barrel" is, but might struggle with highly specific game terms like "Zerg" or "N7 Armor". We may need to fine-tune it (Contrastive tuning) using historical data of EA filenames paired with their images.

### Step 3: Vector Database Search
- Given 10 million assets, brute-force exact search ($O(N)$) takes too long.
- Use HNSW (Hierarchical Navigable Small World) index in Milvus/Qdrant to achieve sub-millisecond search times with ~98% recall accuracy.

---

## Part 7 — Complete Python Code

```python
"""
asset_search_api.py - Multimodal Semantic Search for Game Assets
"""
import logging
from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
import open_clip
from PIL import Image
from qdrant_client import QdrantClient
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Frostbite Semantic Search")

# ---------------------------------------------------------------------------
# ML Initialization
# ---------------------------------------------------------------------------
logger.info("Loading OpenCLIP model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

qdrant = QdrantClient(host="qdrant-server", port=6333)
COLLECTION_NAME = "game_assets_clip"

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
class TextSearchRequest(BaseModel):
    query: str
    project_filter: str = None  # e.g., "battlefield_2042"

@app.post("/v1/search/text")
async def search_by_text(req: TextSearchRequest, limit: int = 20):
    """Text-to-Image/3D Search"""
    
    # 1. Encode Text
    text_tokens = tokenizer([req.query]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True) # Normalize for Cosine Similarity
        
    query_vector = text_features[0].cpu().numpy().tolist()
    
    # 2. Vector Search (with optional metadata filter)
    query_filter = None
    if req.project_filter:
        from qdrant_client.http import models as rest
        query_filter = rest.Filter(
            must=[rest.FieldCondition(key="project", match=rest.MatchValue(value=req.project_filter))]
        )
        
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True
    )
    
    # 3. Format Response
    assets = []
    for hit in search_result:
        assets.append({
            "asset_id": hit.payload["asset_id"],
            "file_path": hit.payload["file_path"],
            "asset_type": hit.payload["asset_type"], # 'texture' or '3d_model'
            "score": hit.score
        })
        
    return {"results": assets}

@app.post("/v1/search/image")
async def search_by_image(file: UploadFile = File(...), limit: int = 20):
    """Image-to-Image/3D Search (Reverse Image Search)"""
    
    # 1. Read and Preprocess Image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # 2. Encode Image
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
    query_vector = image_features[0].cpu().numpy().tolist()
    
    # 3. Vector Search
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit,
        with_payload=True
    )
    
    return {"results": [{"asset_id": hit.payload["asset_id"], "score": hit.score} for hit in search_result]}
```

---

## Part 8 — Deployment

### Indexing Job (Batch)
- 10 million assets take significant time to embed.
- Deploy a scalable PySpark or Ray cluster. Workers pull images from S3, run them through CLIP (batch size 128 on GPUs), and push vectors to Qdrant.
- For new assets created daily, a Kafka topic triggers a serverless GPU function (AWS Lambda / modal.com) to embed and upsert into Qdrant instantly.

### Serving API (Real-Time)
- Text embedding is computationally cheap compared to image embedding. The `/search/text` endpoint can easily run on CPU-only Kubernetes pods to save money.
- The `/search/image` endpoint requires image preprocessing and is better suited for GPU pods (e.g., NVIDIA T4) if traffic is high.

---

## Part 9 — Unit Testing

```python
import torch
import numpy as np

def test_normalization():
    # Ensure vectors are L2 normalized. 
    # Qdrant requires this for accurate Cosine Similarity if configured for Dot Product.
    mock_feature = torch.tensor([[1.0, 2.0, 3.0]])
    norm_feature = mock_feature / mock_feature.norm(dim=-1, keepdim=True)
    
    # Magnitude should be 1.0
    magnitude = torch.norm(norm_feature).item()
    assert np.isclose(magnitude, 1.0)
```

---

## Part 10 — Integration Testing

- Create a small test collection in a local Qdrant container with 10 known assets (e.g., 5 cars, 5 trees).
- Query for "vehicle".
- Assert that the top 5 results returned by the API all have `asset_type` or metadata corresponding to cars, proving the CLIP model's semantic understanding aligns with the vector DB retrieval.

---

## Part 11 — Scaling Discussion

| Axis | Strategy |
|------|----------|
| **10M to 100M Assets** | HNSW index size grows linearly. 100M 512-d vectors in FP32 take ~200GB of RAM. We must shard the Qdrant cluster across multiple nodes. We can also use Product Quantization (PQ) or Scalar Quantization (INT8) to reduce the vector footprint by 4x with minimal recall loss. |
| **High Search Traffic** | Text encoding takes ~50ms on CPU. The HNSW search takes ~5ms. We can easily scale horizontally. |

---

## Part 12 — Tradeoffs

| Decision | Tradeoff |
|----------|----------|
| CLIP vs Custom CNN/ResNet | CLIP enables zero-shot text-to-image search instantly out of the box. A custom ResNet would require manual tagging of all 10M assets to train, but might be slightly better at classifying specific art styles. |
| 2D Thumbnails vs 3D Point Clouds | Generating thumbnails allows us to reuse standard 2D image models (CLIP). Native 3D embedding models (PointNet, 3D-CNNs) are heavily experimental, harder to train, and slower to index. |
| FP32 vs INT8 Vectors | FP32 yields perfect accuracy but takes 4x the RAM of INT8. For a search engine (where top 20 is sufficient, exact order doesn't matter perfectly), INT8 quantization is an obvious tradeoff to save massive infrastructure costs. |

---

## Part 13 — Alternative Approaches

1. **Multimodal LLMs (VLM):** Instead of CLIP, use a Vision-Language Model like LLaVA to generate a highly detailed text caption for every 3D asset (e.g., "A rusted metal barrel with a biohazard symbol, low poly"). Then use a standard text-to-text search engine (Elasticsearch). Slower to index, but allows very complex Boolean queries ("barrel AND NOT red").
2. **LoRA Fine-tuning on CLIP:** If CLIP struggles with game-specific concepts (e.g., "Madden NFL uniforms"), fine-tune the CLIP text/image encoders using a Contrastive loss on internal EA concept art data.

---

## Part 14 — Failure Scenarios

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| Vocabulary Gap | CLIP doesn't know what a "Needler" (Halo/sci-fi gun) is, returns random purple objects. | Supplement vector search with exact keyword search (Hybrid Search). If the asset filename is `needler_weapon.fbx`, BM25 will find it instantly even if the semantic vector fails. |
| Out of Memory (OOM) | Vector DB crashes during a bulk indexing job | Rate limit the ingestion pipeline. Ensure HNSW `ef_construction` parameters are tuned so index building doesn't spike RAM usage beyond node limits. |

---

## Part 15 — Debugging

**Symptom:** Artists search for "tree", but the top results are 3D models of grass and bushes. The actual tree models are ranked #50.

**Debugging steps:**
1. Is it an indexing bug? Verify the tree models actually exist in the Vector DB.
2. Check the 2D thumbnails for the 3D trees. If the thumbnail was rendered from a top-down orthographic camera, a tree looks exactly like a bush (a green circle).
3. **Fix:** Ensure the 3D rendering pipeline uses a 45-degree isometric camera angle for thumbnails so the vertical profile of the asset is visible to the ML model.

---

## Part 16 — Monitoring

| Metric | Alert Threshold |
|--------|----------------|
| `search_latency_p99_ms` | > 500ms → Degraded UX. Investigate DB shards. |
| `zero_result_searches` | > 5% → Users are searching for things we don't have, or our model lacks domain knowledge. Log these queries for fine-tuning. |
| `index_sync_lag_hours` | > 24h → Ingestion pipeline is broken. |

---

## Part 17 — Production Improvements

1. **Hybrid Search (Sparse + Dense):** Integrate Qdrant's BM25 sparse vectors. Combine the semantic score of CLIP with the exact text match of the asset's filename/metadata.
2. **Personalization:** Apply a lightweight re-ranker. If the artist searching works on the "Dead Space" team, boost assets tagged with `style: horror` or `project: dead_space` to the top of the semantic results.
3. **Color Filtering:** Add dominant color extraction during the indexing phase. Allow artists to filter by color palette (e.g., "Red sci-fi barrel").

---

## Part 18 — Follow-up Questions

> *Interviewer asks these after the initial solution is presented.*

1. **"CLIP embeddings are 512 dimensions. With 10 million assets, keeping the HNSW index in RAM is expensive. What specific techniques would you use to reduce the memory footprint by at least 80%?"**
2. **"An artist uploads a sketch of a sword. They want to find 3D models that match the sketch. Standard CLIP struggles here because it maps 'sketches' and 'photorealistic renders' to different parts of the vector space. How do you solve this Domain Gap?"**
3. **"We want to automatically tag the 10 million assets with text tags (e.g., 'metal', 'weapon', 'sci-fi') to populate our legacy SQL database. How can you use your existing CLIP architecture to do this without writing new models?"**

---

## Part 19 — Ideal Answers

**Q1 (Memory Reduction):**
> "To achieve an 80% reduction, we use a combination of Scalar Quantization and Product Quantization (PQ). First, we convert FP32 floats to INT8 (4x reduction). Second, we use PQ to divide the 512-d vector into sub-vectors and replace them with centroid IDs from a codebook. Additionally, we can move the raw vectors to disk (SSD) using an architecture like DiskANN, keeping only the HNSW graph navigation structure in RAM."

**Q2 (Sketch to 3D / Domain Gap):**
> "This requires fine-tuning or a specialized model. We shouldn't use base CLIP. We should use a model trained specifically for cross-domain retrieval, like 'CLIP for Sketch-to-Photo'. Alternatively, we can insert an adapter layer (a small MLP) on top of the CLIP image encoder, and train it using Triplet Loss: Anchor (Sketch), Positive (3D Render of same object), Negative (3D render of different object). This forces sketches and renders of the same object to share the same vector space."

**Q3 (Auto-Tagging / Zero-Shot Classification):**
> "We can use CLIP for Zero-Shot Image Classification. We define a fixed vocabulary of tags (e.g., a list of 1,000 tags). We pre-compute the text embedding for all 1,000 tags. When a new asset is indexed, we take its image embedding and calculate the dot product against the 1,000 tag embeddings. The tags with the highest cosine similarity (e.g., > 0.85) are automatically assigned to the asset and written to the SQL database."

---

## Part 20 — Evaluation Rubric

### Strong Hire
- Understands that 3D models require 2D renders to utilize foundation vision models.
- Explains HNSW and Vector DB scaling constraints accurately.
- Answers the Quantization (PQ/INT8) and Zero-Shot classification questions flawlessly.
- Anticipates the domain-specific vocabulary problem with out-of-the-box CLIP.

### Hire
- Successfully designs the CLIP + Vector DB architecture.
- Understands how Cosine Similarity connects text and images.
- Implements basic metadata filtering.
- Code uses PyTorch and Qdrant correctly.

### Lean Hire
- Suggests building a massive CNN from scratch instead of leveraging foundation models like CLIP.
- Does not understand how to scale Vector DBs beyond brute-force exact search.
- Solves the problem but the solution is computationally impractical for 10M assets.

### Lean No Hire
- Fails to bridge the gap between Text and Image (doesn't know about shared embedding spaces).
- Suggests using NLP models (BERT) on the filenames only, ignoring the multimodal requirement.

### No Hire
- Doesn't know what an embedding is.
- Cannot articulate how image similarity works mathematically.
