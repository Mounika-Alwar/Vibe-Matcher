# Vibe Matcher: AI-Powered Fashion Recommendation System

## Overview
Vibe Matcher is an AI-powered recommendation system that understands a user's vibe or mood and suggests fashion products in real-time. The system uses text embeddings, vector search, and similarity scoring to provide top-k matches and handles cases where no strong match exists.

This repository includes:
- Jupyter Notebook for prototyping
- FastAPI backend for serving recommendations
- Streamlit frontend for live interaction
- FAISS vector store for embeddings search

## Features
- Converts product descriptions and user queries into vector embeddings
- Stores embeddings in FAISS for efficient similarity search
- Computes cosine similarity and ranks top-k items
- Provides fallback suggestions when no strong match exists
- Real-time API using FastAPI
- Interactive frontend using Streamlit
- Quality scoring of recommendations (good/avg/bad)

## Architecture Overview
1. **Data Preparation**: Small dataset of products with descriptions, vibes, and categories.
2. **Embedding Generation**: SentenceTransformer embeddings for products.
3. **Vector Search**: FAISS index for fast nearest neighbor search.
4. **Backend API**: FastAPI `/match` endpoint that returns top recommendations with similarity and quality scores.
5. **Frontend**: Streamlit app to input queries and display recommendations.
6. **Deployment**: Backend deployed on Render, frontend deployed on Streamlit Cloud.

## Live app
https://mounika-alwar-vibe-matcher-frontendapp-rrnpn9.streamlit.app/

## API 
https://vibe-matcher-1.onrender.com/
