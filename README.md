# Hybrid Graph/Vector Technical RAG & Evaluation Pipeline

Multi-hop technical retrieval combining `gemini-embedding-001` dense vector search, a `NetworkX` entity knowledge graph, Reciprocal Rank Fusion (RRF), Gemini 3 Flash structured citation synthesis, and automated Recall@K evaluation.

## Quickstart

    pip install -r requirements.txt
    pytest -v
    python -m eval.benchmark
