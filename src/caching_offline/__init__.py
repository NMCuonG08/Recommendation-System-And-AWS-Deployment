"""Offline caching: index Item2Vec embeddings into Qdrant and precompute
Redis-side retrieval artifacts (popular items + per-item similar items) used by
the API gateway at serve time."""