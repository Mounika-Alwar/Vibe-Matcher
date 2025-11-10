import json
import numpy as np
import faiss

# load products that already have embeddings
with open("products.json", "r") as f:
    products = json.load(f)

# convert embeddings to array
vectors = np.array([item["embedding"] for item in products]).astype("float32")

# load or rebuild FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search(query_embedding, top_k=3, threshold=0.30):
    scored = []
    for item in products:
        sim = cosine_similarity(query_embedding, item["embedding"])
        scored.append((sim, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    if top[0][0] < threshold:
        categories = list({p.get("category", "general") for p in products})
        return {
            "result_type": "fallback",
            "suggest_categories": categories
        }

    formatted = []
    for sim, item in top:
        formatted.append({
            "name": item["name"],
            "description": item["description"],
            "vibes": item["vibes"],
            "category": item.get("category", "general"),
            "similarity": float(sim)
        })

    return {
        "result_type": "match",
        "results": formatted
    }
