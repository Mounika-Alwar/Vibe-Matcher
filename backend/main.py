from fastapi import FastAPI
from pydantic import BaseModel
from vector_store import search
import time

app = FastAPI()

class QueryRequest(BaseModel):
    embedding: list
    top_k: int = 3

def classify_score(sim):
    if sim is None:
        return "no_match"
    if sim >= 0.7:
        return "good"
    elif sim >= 0.4:
        return "avg"
    return "bad"

@app.post("/match")
def match_vibe(data: QueryRequest):
    start = time.time()
    result = search(data.embedding, data.top_k)
    end = time.time()
    latency = end - start

    if result["result_type"] == "fallback":
        return {
            "result_type": "fallback",
            "quality": "no_match",
            "latency_seconds": latency,
            "suggest_categories": result["suggest_categories"]
        }

    top_item = result["results"][0]
    return {
        "result_type": "match",
        "top_item": top_item["name"],
        "similarity": top_item["similarity"],
        "quality": classify_score(top_item["similarity"]),
        "latency_seconds": latency,
        "top_results": result["results"]
    }

@app.get("/")
def home():
    return {"message": "API is running ✅"}

