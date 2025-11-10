import streamlit as st
import requests
from sentence_transformers import SentenceTransformer

API_URL = "https://vibe-matcher-1.onrender.com/match"

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

st.title("Vibe Matcher ✨")

query = st.text_input("Describe your vibe:")
top_k = st.slider("Top K Results", 1, 5, 3)

if st.button("Find Match"):
    if not query.strip():
        st.warning("Enter a vibe first.")
    else:
        emb = model.encode(query).tolist()
        payload = {"embedding": emb, "top_k": top_k}
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.json(result)
        else:
            st.error(f"Error: {response.text}")

