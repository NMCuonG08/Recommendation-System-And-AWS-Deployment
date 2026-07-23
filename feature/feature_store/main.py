"""FastAPI service exposing Feast online features for the recommender.

Ported from the reference `feature_store/main.py`, adapted to MovieLens:

  - entity key `movieId` (int) instead of `parent_asin` (str)
  - entity key `userId` (int) instead of `user_id` (str)
  - feature views `movie_feature_view` / `user_feature_view` (see feature_views.py)

Endpoints (mirrors the reference gateway's Feast proxy contract):
  POST /features_movie    { "movie_id": int }  -> movie aggregate features
  POST /user_features      { "user_id": int }   -> user sequence features

Run (from feature/feature_store):
    uv run uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from feast import FeatureStore

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

app = FastAPI(title="Feast Feature Store API (MovieLens)")
# Default to feature_store.yaml (in the docker container this is the docker
# config mounted by compose; for local dev set FEAST_CONFIG_FILE to
# feature_store.local.yaml or copy it over feature_store.yaml).
FEAST_CONFIG_FILE = os.environ.get("FEAST_CONFIG_FILE", "feature_store.yaml")
store = FeatureStore(repo_path=str(REPO_DIR), fs_yaml_file=FEAST_CONFIG_FILE)


class MovieInput(BaseModel):
    """Pydantic model for validating movie id input."""
    movie_id: int


class UserInput(BaseModel):
    """Pydantic model for validating user id input."""
    user_id: int


@app.post("/features_movie")
def get_features_movie(input: MovieInput) -> dict:
    """Retrieve online features for a given movieId from the Feast FeatureStore.

    Mirrors reference `get_features_parent_asin` but for movie aggregate views.
    """
    store.refresh_registry()
    entity_rows = [{"movieId": input.movie_id}]
    features = [
        "movie_feature_view:movie_rating_cnt_90d",
        "movie_feature_view:movie_rating_avg_prev_rating_90d",
        "movie_feature_view:movie_rating_cnt_30d",
        "movie_feature_view:movie_rating_avg_prev_rating_30d",
        "movie_feature_view:movie_rating_cnt_7d",
        "movie_feature_view:movie_rating_avg_prev_rating_7d",
        "movie_feature_view:title",
        "movie_feature_view:genres",
    ]
    return store.get_online_features(
        features=features,
        entity_rows=entity_rows,
    ).to_dict()


@app.post("/user_features")
def get_features_user(input: UserInput) -> dict:
    """Retrieve online features for a given userId from the Feast FeatureStore.

    Mirrors reference `get_features_user`; returns the user's recent-10 movie
    sequence + ts buckets consumed by the ranker's GRU input.
    """
    store.refresh_registry()
    entity_rows = [{"userId": input.user_id}]
    features = [
        "user_feature_view:user_rating_cnt_90d",
        "user_feature_view:user_rating_avg_prev_rating_90d",
        "user_feature_view:user_rating_list_10_recent_movie",
        "user_feature_view:user_rating_list_10_recent_movie_timestamp",
        "user_feature_view:item_sequence_ts",
        "user_feature_view:item_sequence_ts_bucket",
    ]
    return store.get_online_features(
        features=features,
        entity_rows=entity_rows,
    ).to_dict()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}