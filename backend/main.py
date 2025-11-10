# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from vector_store import search
import time

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

def classify_score(sim):
    if sim is None:
        return "no_match"
    if sim >= 0.7:
        return "good"
    elif sim >= 0.4:
        return "avg"
    else:
        return "bad"

@app.post("/match")
def match_vibe(data: QueryRequest):
    start = time.time()
    result = search(data.query, data.top_k)
    end = time.time()
    latency = end - start

    if result["result_type"] == "fallback":
        return {
            "query": data.query,
            "result_type": "fallback",
            "top_item": None,
            "similarity": None,
            "latency_seconds": latency,
            "quality": "no_match",
            "suggest_categories": result["suggest_categories"]
        }

    top_item = result["results"][0]
    similarity = top_item["similarity"]

    return {
        "query": data.query,
        "result_type": "match",
        "top_item": top_item["name"],
        "similarity": similarity,
        "latency_seconds": latency,
        "quality": classify_score(similarity),
        "top_results": result["results"]
    }
