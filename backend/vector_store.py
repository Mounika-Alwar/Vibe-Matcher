import json
import numpy as np
import faiss

# Load product data (with precomputed embeddings included)
with open("products.json", "r") as f:
    products = json.load(f)

# Load FAISS index (already built offline)
index = faiss.read_index("vibe_index.faiss")

def cosine_similarity(a, b):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search(query_embedding, top_k=3, threshold=0.30):
    """
    query_embedding: list of floats (precomputed on frontend or external service)
    """

    q_emb = np.array(query_embedding, dtype="float32").reshape(1, -1)

    # FAISS nearest neighbor search
    distances, indices = index.search(q_emb, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        item = products[idx]
        sim = cosine_similarity(query_embedding, item["embedding"])
        results.append({
            "name": item["name"],
            "description": item["description"],
            "vibes": item.get("vibes", []),
            "category": item.get("category", "general"),
            "similarity": sim
        })

    # Sort by similarity (FAISS uses distance, we prefer cosine sim)
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Fallback: if highest similarity is too low
    if results[0]["similarity"] < threshold:
        categories = list({p.get("category", "general") for p in products})
        return {
            "result_type": "fallback",
            "suggest_categories": categories
        }

    return {
        "result_type": "match",
        "results": results
    }

