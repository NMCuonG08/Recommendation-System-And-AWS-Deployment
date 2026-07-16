"""Feast FeatureViews for the MovieLens feature store.

Sources are local parquet files produced by `feature/etl/003-feature-etl.ipynb`:

- `movie_rating_stats.parquet`  -> movie_feature_view  (rating aggregates + title/genres)
- `user_rating_stats.parquet`   -> user_feature_view   (user rating aggregates + recent-10 sequence)

The raw train/val interaction parquets are NOT registered as feature views —
`feature/engineer/004-features.ipynb` reads them directly by path (the reference
project used Feast views only to locate the path, which we skip to avoid
pointless point-in-time machinery on the training data itself).

Online serving enabled for the aggregate views (the ones a live recommender
queries per user/movie).
"""

from datetime import timedelta

from feast import FeatureView, Field
from feast.infra.offline_stores.file_source import FileSource
from feast.types import Array, Float32, Int64, String

from entities import movie, user

OUTPUT_DIR = "../output"


def _movie_source():
    return FileSource(
        path=f"{OUTPUT_DIR}/movie_rating_stats.parquet",
        timestamp_field="event_timestamp",
        file_format=None,  # parquet inferred from .parquet extension
    )


def _user_source():
    return FileSource(
        path=f"{OUTPUT_DIR}/user_rating_stats.parquet",
        timestamp_field="event_timestamp",
        file_format=None,
    )


movie_feature_view = FeatureView(
    name="movie_feature_view",
    entities=[movie],
    ttl=timedelta(days=10000),
    schema=[
        Field(name="movie_rating_cnt_90d", dtype=Int64, description="Ratings for this movie in the last 90 days."),
        Field(name="movie_rating_avg_prev_rating_90d", dtype=Float32, description="Avg rating for this movie in the last 90 days."),
        Field(name="movie_rating_cnt_30d", dtype=Int64, description="Ratings for this movie in the last 30 days."),
        Field(name="movie_rating_avg_prev_rating_30d", dtype=Float32, description="Avg rating for this movie in the last 30 days."),
        Field(name="movie_rating_cnt_7d", dtype=Int64, description="Ratings for this movie in the last 7 days."),
        Field(name="movie_rating_avg_prev_rating_7d", dtype=Float32, description="Avg rating for this movie in the last 7 days."),
        Field(name="title", dtype=String, description="Movie title (with year)."),
        Field(name="genres", dtype=String, description="Pipe-delimited genres."),
    ],
    source=_movie_source(),
    online=True,
)

user_feature_view = FeatureView(
    name="user_feature_view",
    entities=[user],
    ttl=timedelta(days=10000),
    schema=[
        Field(name="user_rating_cnt_90d", dtype=Int64, description="Ratings by this user in the last 90 days."),
        Field(name="user_rating_avg_prev_rating_90d", dtype=Float32, description="Avg rating by this user in the last 90 days."),
        Field(name="user_rating_list_10_recent_movie", dtype=String, description="Comma-separated 10 most recent movie ids rated by the user."),
        Field(name="user_rating_list_10_recent_movie_timestamp", dtype=String, description="Comma-separated timestamps for the 10 most recent ratings."),
        Field(name="item_sequence_ts", dtype=Array(Int64), description="Unix timestamps (ms) for the 10 most recent user ratings (padded -1)."),
        Field(name="item_sequence_ts_bucket", dtype=Array(Int64), description="Time-difference buckets for the 10 most recent user ratings."),
    ],
    source=_user_source(),
    online=True,
)