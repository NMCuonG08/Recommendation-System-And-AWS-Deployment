"""Index Item2Vec embeddings into Qdrant and precompute Redis retrieval caches.

Ported from the reference `src/caching_offline/load2redis_qdrant.ipynb`, adapted
to this port:

  - The Item2Vec champion is a Lightning checkpoint on disk
    (`models/output/item2vec/final_model/best-checkpoint.ckpt`, embedding key
    `skipgram_model.embeddings.weight`, dim 64).
  - The IDMapper is `id_mapper.json` with **string** MovieLens id keys and int
    `index_to_item`; normalized to int<->index so Qdrant payloads carry int
    movie ids (what the gateway / Feast expect).
  - Two Redis caches are written (same keys the reference gateway reads):
      * `popular_movie_score` — sorted set, score = rating count per movie
        (reference used `popular_parent_asin_score`; we count ratings per movie
        from `feature/output/train.parquet`).
      * `rec:{movieId}` — JSON `{target_item, rec_item_ids, rec_scores}`,
        top-K Qdrant-cosine neighbors per item. Reference re-scored neighbors
        with the item2vec model; we keep raw cosine order (the ranker re-scores
        candidates at serve time anyway).

Run (with `docker compose up -d redis qdrant`):
    uv run python -m src.caching_offline.load_qdrant
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (env-driven, mirrors the gateway + docker-compose defaults).
# ---------------------------------------------------------------------------
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION_NAME", "item2vec_collection")
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_DB = int(os.environ.get("REDIS_DB", 0))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "") or None

ITEM2VEC_FINAL_DIR = ROOT / "models" / "output" / "item2vec" / "final_model"

TOP_K_SIMILAR = int(os.environ.get("TOP_K_SIMILAR", 10))   # rec:{movieId} size
QDRANT_SEARCH_LIMIT = int(os.environ.get("QDRANT_SEARCH_LIMIT", 50))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
# Primary popularity source: raw interactions (one row per rating). Fall back to
# the per-(movieId, event_timestamp) aggregate snapshot when the raw train split
# is absent (only movie_rating_stats.parquet was materialized) — popular_movie_score
# must still be written or the gateway has no popularity candidates at all.
TRAIN_PARQUET = ROOT / "feature" / "output" / "train.parquet"
MOVIE_STATS_PARQUET = ROOT / "feature" / "output" / "movie_rating_stats.parquet"

# Embedding key in the item2vec Lightning checkpoint (LitSkipGram).
_EMBED_KEY = "skipgram_model.embeddings.weight"


def _redis_client():
    """Lazy redis import so the module loads without redis running."""
    import redis

    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=True,
    )


# ---------------------------------------------------------------------------
# Item2Vec champion embedding loading (disk).
# ---------------------------------------------------------------------------


def load_item2vec_embeddings() -> Tuple[torch.Tensor, Dict[int, int], List[int]]:
    """Return (embeddings, item_to_index, index_to_item) from the disk checkpoint.

    embeddings: [num_items, dim] (padding row dropped).
    item_to_index: int movie id -> index.
    index_to_item: list of int movie ids (index -> movie id).
    """
    ckpt_path = ITEM2VEC_FINAL_DIR / "best-checkpoint.ckpt"
    idm_path = ITEM2VEC_FINAL_DIR / "id_mapper.json"
    if not ckpt_path.is_file() or not idm_path.is_file():
        raise FileNotFoundError(
            f"Item2Vec artifacts missing: {ckpt_path} / {idm_path}"
        )
    logger.info("Loading Item2Vec checkpoint: %s", ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    weight = state[_EMBED_KEY].detach().cpu()  # [num_items+1, dim]

    idm = json.load(open(idm_path, "r", encoding="utf-8"))
    item_to_index = {int(k): int(v) for k, v in idm["item_to_index"].items()}  # str -> int
    index_to_item = [int(x) for x in idm["index_to_item"]]

    n_items = len(index_to_item)
    if weight.shape[0] != n_items + 1:
        logger.warning(
            "embedding rows %d != items+1 %d; using first %d rows",
            weight.shape[0], n_items + 1, n_items,
        )
    embeddings = weight[:n_items]  # drop padding row
    logger.info("Loaded embeddings: %s for %d items", tuple(embeddings.shape), n_items)
    return embeddings, item_to_index, index_to_item


# ---------------------------------------------------------------------------
# Qdrant indexing.
# ---------------------------------------------------------------------------


def index_embeddings_to_qdrant(
    embeddings: torch.Tensor,
    index_to_item: List[int],
    qdrant_url: str = QDRANT_URL,
    collection_name: str = QDRANT_COLLECTION,
) -> None:
    """Create the Qdrant collection (cosine) and upsert all item embeddings.

    Point id = index, payload = {"item_id": movie_id}. Mirrors the reference's
    index_embeddings_to_qdrant (delete-then-create, cosine distance).
    """
    client = QdrantClient(url=qdrant_url)
    logger.info("Connecting to Qdrant at %s", qdrant_url)

    vector_dim = int(embeddings.shape[1])
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        logger.info("Deleted existing collection '%s'.", collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s' dim=%d", collection_name, vector_dim)

    vectors = embeddings.numpy().astype(np.float32)
    points = [
        PointStruct(id=idx, vector=vec.tolist(), payload={"item_id": int(index_to_item[idx])})
        for idx, vec in enumerate(vectors)
    ]
    for start in range(0, len(points), BATCH_SIZE):
        client.upsert(collection_name=collection_name, points=points[start:start + BATCH_SIZE])
    logger.info("Upserted %d embeddings into '%s'", len(points), collection_name)


# ---------------------------------------------------------------------------
# Popular items cache (Redis sorted set).
# ---------------------------------------------------------------------------


def compute_popular_items(
    redis_client,
    train_parquet: Path = TRAIN_PARQUET,
    movie_stats_parquet: Path = MOVIE_STATS_PARQUET,
) -> int:
    """Score movies by rating count and load Redis zset `popular_movie_score`.

    The gateway reads this zset as the popularity-based candidate source
    (reference: `popular_parent_asin_score`). Prefers the raw interaction split
    (`train.parquet`, one row per rating -> groupby size) and falls back to the
    per-(movieId, event_timestamp) aggregate snapshot
    (`movie_rating_stats.parquet`), taking the latest snapshot's 90-day rating
    count per movie so each movie is counted once.
    """
    if train_parquet.is_file():
        logger.info("Computing popular items from interactions: %s", train_parquet)
        df = pd.read_parquet(train_parquet)
        counts = df.groupby("movieId").size().sort_values(ascending=False)
    elif movie_stats_parquet.is_file():
        logger.info("train.parquet missing; falling back to %s", movie_stats_parquet)
        df = pd.read_parquet(movie_stats_parquet)
        # movie_rating_stats has one row per (movieId, event_timestamp) snapshot.
        # Keep the latest snapshot per movie and use its 90-day rating count.
        latest = (
            df.sort_values("event_timestamp")
            .groupby("movieId", as_index=False)
            .tail(1)
        )
        counts = (
            latest.set_index("movieId")["movie_rating_cnt_90d"]
            .sort_values(ascending=False)
        )
    else:
        raise FileNotFoundError(
            f"Neither {train_parquet} nor {movie_stats_parquet} exist; cannot "
            "compute popular items. Run the feature ETL first."
        )
    logger.info("Popular items: %d movies scored", len(counts))

    key = "popular_movie_score"
    redis_client.delete(key)
    items = list(counts.items())
    for start in range(0, len(items), BATCH_SIZE):
        chunk = {str(int(mid)): float(cnt) for mid, cnt in items[start:start + BATCH_SIZE]}
        redis_client.zadd(key, chunk)
    logger.info("Wrote %d popular items to Redis key '%s'", len(items), key)
    return len(items)


# ---------------------------------------------------------------------------
# Per-item similar-items cache (rec:{movieId} -> Redis JSON).
# ---------------------------------------------------------------------------


def compute_similar_items(
    qdrant_client: QdrantClient,
    embeddings: np.ndarray,
    index_to_item: List[int],
    collection_name: str = QDRANT_COLLECTION,
    top_k: int = TOP_K_SIMILAR,
    search_limit: int = QDRANT_SEARCH_LIMIT,
    redis_client=None,
) -> int:
    """For each item, query Qdrant by its vector for cosine neighbors and write
    `rec:{movieId}` JSON ({target_item, rec_item_ids, rec_scores}) to Redis.

    Mirrors the reference's compute_recommendations, minus the item2vec
    re-scoring (we keep raw cosine order; the ranker re-scores at serve time).
    """
    n_items = len(index_to_item)
    logger.info("Querying Qdrant by vector for %d items (top_k=%d)", n_items, top_k)
    written = 0
    for idx in range(n_items):
        movie_id = int(index_to_item[idx])
        # qdrant-client >= 1.7 removed `.search(query_vector=...)`; use
        # `.query_points(query=...)` which returns `.points` (list of ScoredPoint).
        response = qdrant_client.query_points(
            collection_name=collection_name,
            query=embeddings[idx].tolist(),
            limit=search_limit + 1,
        )
        neighbors = response.points
        neighbor_payloads = [
            (n.payload.get("item_id") if n.payload else None, n.score)
            for n in neighbors
            if n.id != idx
        ][:top_k]
        if not neighbor_payloads:
            continue
        rec_item_ids = [int(p[0]) for p in neighbor_payloads if p[0] is not None]
        rec_scores = [float(p[1]) for p in neighbor_payloads if p[0] is not None]
        rec = {"target_item": movie_id, "rec_item_ids": rec_item_ids, "rec_scores": rec_scores}
        if redis_client is not None:
            redis_client.set(f"rec:{movie_id}", json.dumps(rec))
        written += 1
        if written % 500 == 0:
            logger.info("  ...%d/%d similar-items cached", written, n_items)
    logger.info("Wrote rec:<movieId> for %d items to Redis", written)
    return written


def main() -> None:
    embeddings, _item_to_index, index_to_item = load_item2vec_embeddings()

    try:
        index_embeddings_to_qdrant(embeddings, index_to_item)
    except UnexpectedResponse as exc:
        logger.error("Qdrant indexing failed (%s). Is `docker compose up -d qdrant` running?", exc)
        raise

    qdrant_client = QdrantClient(url=QDRANT_URL)
    redis_client = _redis_client()

    try:
        compute_popular_items(redis_client)
    except Exception as exc:
        logger.warning("Popular items failed (%s). Is Redis running?", exc)

    try:
        compute_similar_items(
            qdrant_client, embeddings.numpy().astype(np.float32), index_to_item,
            redis_client=redis_client,
        )
    except Exception as exc:
        logger.warning("Similar-items computation failed (%s).", exc)

    logger.info("Done. Qdrant collection='%s', Redis caches written.", QDRANT_COLLECTION)


if __name__ == "__main__":
    main()