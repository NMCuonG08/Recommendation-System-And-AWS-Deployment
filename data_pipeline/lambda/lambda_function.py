"""AWS Lambda handler — real-time CDC update of Feast `user_feature_view`.

Ported from the reference project's `data_pipeline_aws/lambda/lambda_function.py`,
adapted for the MovieLens schema:

- DMS Kinesis target record shape: ``{"data": {"userId", "movieId", "rating",
  "timestamp"}, "metadata": {...}}`` (camelCase MovieLens columns, NOT the
  reference's ``user_id`` / ``parent_asin``).
- Feast entity ``user`` join key is ``userId`` (Int64) — see
  ``feature/feature_store/entities.py``.
- Feature view fields are ``user_rating_list_10_recent_movie`` /
  ``user_rating_list_10_recent_movie_timestamp`` (NOT ``..._asin``).
- Time bucketization reuses ``timestamp_bucket.from_ts_to_bucket`` (copied into
  the image) instead of an inlined duplicate.

Env vars (set by Terraform, resolved at runtime — never hardcoded):
  REGISTRY_PATH_SECRET_ARN  Secrets Manager ARN holding the Feast sql registry
    URI (RDS ``registry_feature_store``). Preferred over REGISTRY_PATH so the
    URI never sits in the Lambda env config.
  REGISTRY_PATH      Fallback: the Feast sql registry URI directly (if no secret).
  FEAST_REPO         Lambda image repo path (defaults to ``/var/task``)
  FEAST_YAML         feature_store.yaml name (defaults to ``feature_store.yaml``)
  AWS_DEFAULT_REGION DynamoDB online store region
"""
from __future__ import annotations

import base64
import json
import logging
import os
import traceback
from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore
from tenacity import retry, stop_after_attempt, wait_fixed

from timestamp_bucket import from_ts_to_bucket

# --------------------------------------------------------------------------- #
# Init (runs once per warm container)
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_VIEW_NAME = "user_feature_view"
SEQUENCE_LEN = 10


def _resolve_registry_path() -> str:
    """Resolve the Feast sql registry URI from Secrets Manager or env.

    If ``REGISTRY_PATH_SECRET_ARN`` is set, fetch the URI from Secrets Manager
    (so it never lives in the Lambda env config). Otherwise fall back to the
    ``REGISTRY_PATH`` env var. The resolved value is exported back to the env
    because ``feature_store.yaml`` references ``${REGISTRY_PATH}``.
    """
    secret_arn = os.getenv("REGISTRY_PATH_SECRET_ARN")
    if secret_arn:
        import boto3  # local import: only needed in the secret path
        client = boto3.client("secretsmanager", region_name=os.getenv("AWS_DEFAULT_REGION"))
        uri = client.get_secret_value(SecretId=secret_arn)["SecretString"].strip()
        os.environ["REGISTRY_PATH"] = uri
        return uri
    return os.environ.get("REGISTRY_PATH", "")


def _init_store() -> FeatureStore:
    """Initialize the Feast FeatureStore from the config bundled in the image."""
    _resolve_registry_path()
    repo_path = os.getenv("FEAST_REPO", "/var/task")
    fs_yaml = os.getenv("FEAST_YAML", "feature_store.yaml")
    return FeatureStore(repo_path=repo_path, fs_yaml_file=fs_yaml)


store = _init_store()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def fetch_user_features(user_id: int) -> dict:
    """Fetch the current online feature vector for a user (with retry)."""
    features = [
        f"{FEATURE_VIEW_NAME}:user_rating_cnt_90d",
        f"{FEATURE_VIEW_NAME}:user_rating_avg_prev_rating_90d",
        f"{FEATURE_VIEW_NAME}:user_rating_list_10_recent_movie",
        f"{FEATURE_VIEW_NAME}:user_rating_list_10_recent_movie_timestamp",
        f"{FEATURE_VIEW_NAME}:item_sequence_ts",
        f"{FEATURE_VIEW_NAME}:item_sequence_ts_bucket",
    ]
    logger.info("fetch_user_features userId=%s", user_id)
    return store.get_online_features(
        features=features, entity_rows=[{"userId": user_id}]
    ).to_dict()


def _iso_to_unix(ts: str) -> int:
    """Convert an ISO/timestamp string to unix seconds; -1 on failure."""
    try:
        return int(pd.to_datetime(ts).timestamp())
    except Exception:
        return -1


def _pad_left(seq: list[int], length: int, fill: int = -1) -> list[int]:
    """Left-pad ``seq`` to ``length`` with ``fill`` (matches offline pipeline)."""
    if len(seq) >= length:
        return seq[-length:]
    return [fill] * (length - len(seq)) + seq


def update_and_write_to_online(user_id: int, movie_id: int, rating: float, timestamp):
    """Append the new interaction to the user's sequence and write back online."""
    fv = store.get_feature_view(FEATURE_VIEW_NAME)
    schema_fields = [f.name for f in fv.schema]
    current = fetch_user_features(user_id)

    prev_movies_raw = current.get("user_rating_list_10_recent_movie", [""])[0]
    prev_ts_raw = current.get("user_rating_list_10_recent_movie_timestamp", [""])[0]
    old_movies = prev_movies_raw.split(",") if prev_movies_raw else []
    old_ts = prev_ts_raw.split(",") if prev_ts_raw else []

    new_movies = (old_movies + [str(movie_id)])[-SEQUENCE_LEN:]
    new_ts_str = (old_ts + [str(timestamp)])[-SEQUENCE_LEN:]

    item_sequence_ts = [t for t in (_iso_to_unix(s) for s in new_ts_str)]
    item_sequence_ts = _pad_left(item_sequence_ts, SEQUENCE_LEN)

    current_ts = (
        item_sequence_ts[-1]
        if item_sequence_ts and item_sequence_ts[-1] != -1
        else int(datetime.now(timezone.utc).timestamp())
    )
    item_sequence_ts_bucket = [
        -1 if ts == -1 else from_ts_to_bucket(ts, current_ts)
        for ts in item_sequence_ts
    ]

    feature_data: dict = {}
    for k in schema_fields:
        if k == "user_rating_list_10_recent_movie":
            feature_data[k] = ",".join(new_movies)
        elif k == "user_rating_list_10_recent_movie_timestamp":
            feature_data[k] = ",".join(new_ts_str)
        elif k == "item_sequence_ts":
            feature_data[k] = item_sequence_ts
        elif k == "item_sequence_ts_bucket":
            feature_data[k] = item_sequence_ts_bucket
        else:
            val = current.get(k, [None])
            feature_data[k] = val[0] if isinstance(val, list) and len(val) == 1 else val

    feature_data["userId"] = user_id
    feature_data["event_timestamp"] = datetime.now(timezone.utc)
    store.write_to_online_store(FEATURE_VIEW_NAME, pd.DataFrame([feature_data]))
    logger.info("Updated feature userId=%s movieId=%s", user_id, movie_id)


def lambda_handler(event, context):
    """Process a batch of CDC records from SQS FIFO or Kinesis stream."""
    logger.info("Received event: %s", json.dumps(event)[:1000])
    for record in event.get("Records", []):
        try:
            if "body" in record:
                body = record["body"]
                decoded = json.loads(body) if isinstance(body, str) else body
            elif "kinesis" in record:
                payload = record["kinesis"]["data"]
                decoded = json.loads(base64.b64decode(payload))
            else:
                decoded = record
                
            data = decoded.get("data", decoded)  # DMS wraps rows in {"data": ...}
            user_id = int(data["userId"])
            movie_id = int(data["movieId"])
            rating = float(data["rating"])
            timestamp = data["timestamp"]
            update_and_write_to_online(user_id, movie_id, rating, timestamp)
        except Exception as ex:  # noqa: BLE001 - log + continue per REF pattern
            logger.error("Error processing record: %s", ex)
            logger.error(traceback.format_exc())
    return {"status": "done"}