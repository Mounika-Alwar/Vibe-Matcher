import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# loading dataset
with open("products.json", "r") as f:
    products = json.load(f)

# loading embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# generating embeddings
for item in products:
    item["embedding"] = model.encode(item["description"]).tolist()

# Converting embeddings to a float32 numpy array
vectors = np.array([item["embedding"] for item in products]).astype("float32")

# Creating FAISS index
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

# Saving FAISS index
faiss.write_index(index, "vibe_index.faiss")

# Saving products metadata as JSON
with open("products.json", "w") as f:
    json.dump(products, f, indent=2)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search(query, top_k=3, threshold=0.30):
    q_emb = model.encode(query).astype("float32")

    scored = []
    for item in products:
        sim = cosine_similarity(q_emb, item["embedding"])
        scored.append((sim, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    top = scored[:top_k]

    if top[0][0] < threshold:
        categories = list({p.get("category", "general") for p in products})
        return {
            "result_type": "fallback",
            "message": "No strong match found. Here are some available styles:",
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