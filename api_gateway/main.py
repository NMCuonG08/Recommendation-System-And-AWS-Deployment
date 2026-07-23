"""FastAPI API Gateway for the two-stage MovieLens recommender.

Ported from the reference `api_gateway/main.py`, adapted to this port:

  - ids are int MovieLens ids (sent to Triton as STRING tensors; the id_mapper
    Python backend parses them to dense indices).
  - user features come from Feast `user_feature_view` (recent-10 movie sequence
    + ts buckets) via the Feast API (`/user_features`).
  - item features come from Feast `movie_feature_view` (title, genres, rating
    cnt/avg 90d/30d/7d) via the Feast API (`/features_movie`).
  - candidates = popular items (Redis `popular_movie_score`) ∪ similar items
    (Redis `rec:{movieId}`) minus the user's seen sequence.
  - the Triton `ensemble` (id_mapper + item_pipeline + ranker ONNX) scores each
    candidate; top-10 returned.

Env (mirrors docker-compose + .env):
  USER_FEATURE_URL   Feast API /user_features   (default http://localhost:8000/user_features)
  ITEM_FEATURE_URL   Feast API /features_movie  (default http://localhost:8000/features_movie)
  TRITON_URL         Triton gRPC                (default localhost:8001)
  MODEL_NAME         Triton model               (default ensemble)
  REDIS_HOST/PORT/DB/PASSWORD
  OTEL_EXPORTER_OTLP_ENDPOINT  if set, OTLP tracing is enabled (else skipped).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import aiohttp
import numpy as np
import redis
from aiohttp import ClientTimeout
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from tritonclient.grpc.aio import InferenceServerClient, InferInput, InferRequestedOutput

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Optional OpenTelemetry — enabled only if an OTLP endpoint is configured.
OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
if OTLP_ENDPOINT:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource(attributes={
        "service.name": "api-gateway",
        "service.namespace": "monitoring",
        "service.instance.id": os.getenv("HOSTNAME", "api-gateway"),
    })
    trace.set_tracer_provider(TracerProvider(resource=resource))
    tracer = trace.get_tracer(__name__)
    otlp_exporter = OTLPSpanExporter(
        endpoint=OTLP_ENDPOINT, headers={"Content-Type": "application/json"},
    )
    trace.get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))
else:
    tracer = None
    logger.info("OTEL_EXPORTER_OTLP_ENDPOINT unset; tracing disabled.")


def _span(name: str):
    """Context manager that starts a span only when tracing is enabled."""
    if tracer is not None:
        return tracer.start_as_current_span(name)
    return contextlib.nullcontext()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_session = aiohttp.ClientSession()
    app.state.triton_client = InferenceServerClient(url=TRITON_URL, ssl=False)
    logger.info("Initialized aiohttp ClientSession and Triton client (url=%s)", TRITON_URL)
    try:
        redis_client.ping()
        logger.info("Redis connection established")
    except redis.RedisError as exc:
        logger.error("Failed to connect to Redis: %s", exc)
        raise RuntimeError("Redis connection failed")
    try:
        yield
    finally:
        await app.state.http_session.close()
        await app.state.triton_client.close()
        logger.info("Closed aiohttp ClientSession and Triton client")


app = FastAPI(title="Recommender API Gateway (MovieLens)", lifespan=lifespan)

if OTLP_ENDPOINT:
    FastAPIInstrumentor.instrument_app(app)  # type: ignore[name-defined]


# ---------------------------------------------------------------------------
# Config (env-driven; no hardcoded secrets — defaults are local dev only).
# ---------------------------------------------------------------------------
redis_pool = redis.ConnectionPool(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    db=int(os.environ.get("REDIS_DB", 0)),
    password=os.environ.get("REDIS_PASSWORD", "") or None,
    decode_responses=True,
)
redis_client = redis.Redis(connection_pool=redis_pool)

USER_FEATURE_URL = os.environ.get("USER_FEATURE_URL", "http://localhost:8000/user_features")
ITEM_FEATURE_URL = os.environ.get("ITEM_FEATURE_URL", "http://localhost:8000/features_movie")
TRITON_URL = os.environ.get("TRITON_URL", "localhost:8001")
MODEL_NAME = os.environ.get("MODEL_NAME", "ensemble")
SEQ_LEN = int(os.environ.get("SEQ_LEN", 10))
MAX_ITEM_ID_LEN = int(os.environ.get("MAX_ITEM_ID_LEN", 1))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
MAX_SEMAPHORE = int(os.environ.get("MAX_FETCH_CONCURRENCY", 50))
MAX_CANDIDATES = int(os.environ.get("MAX_CANDIDATES", 100))
POPULAR_REDIS_KEY = os.environ.get("POPULAR_REDIS_KEY", "popular_movie_score")


class UserRequest(BaseModel):
    user_id: int


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def safe_float(value, default=0.0):
    if value is None or value == "None":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default=0):
    if value is None or value == "None":
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Endpoints: popular / user_features / features_movie / similar_items / infer.
# ---------------------------------------------------------------------------


@app.get("/popular_items")
def get_popular_items(top_k: int = 10):
    """Top-K popular movies from the Redis sorted set."""
    try:
        top_items = redis_client.zrevrange(POPULAR_REDIS_KEY, 0, top_k - 1, withscores=True)
        result = [
            {"rank": i + 1, "movie_id": int(mid), "score": float(score)}
            for i, (mid, score) in enumerate(top_items)
        ]
        return {"popular_items": result}
    except redis.RedisError as exc:
        logger.error("Redis error in get_popular_items: %s", exc)
        raise HTTPException(status_code=500, detail="Redis error")


@app.get("/user_features")
async def get_user_features(user_id: int = Query(1)):
    """Proxy to the Feast API for user online features."""
    try:
        async with app.state.http_session.post(
            USER_FEATURE_URL, json={"user_id": user_id}, timeout=3
        ) as response:
            response.raise_for_status()
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.error("Error calling user service: %s", exc)
        raise HTTPException(status_code=502, detail=f"Error calling user service: {exc}")


@app.get("/features_movie")
async def get_item_features(movie_id: int = Query(1)):
    """Proxy to the Feast API for movie online features."""
    try:
        async with app.state.http_session.post(
            ITEM_FEATURE_URL, json={"movie_id": movie_id}, timeout=3
        ) as response:
            response.raise_for_status()
            return await response.json()
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.error("Error calling item service: %s", exc)
        raise HTTPException(status_code=502, detail=f"Error calling item service: {exc}")


@app.get("/similar_items")
def get_similar_items(movie_id: int = Query(1)):
    """Retrieve precomputed similar items for a movie from Redis."""
    key = f"rec:{movie_id}"
    try:
        rec_json = redis_client.get(key)
        if rec_json:
            rec_data = json.loads(rec_json)
            return {"movie_id": movie_id, "recommendations": rec_data}
        raise HTTPException(status_code=404, detail=f"No recommendation found for movie {movie_id}")
    except redis.RedisError as exc:
        logger.error("Redis error in get_similar_items: %s", exc)
        raise HTTPException(status_code=500, detail="Redis error")
    except json.JSONDecodeError as exc:
        logger.error("JSON decode error for similar items %s: %s", key, exc)
        raise HTTPException(status_code=500, detail="Invalid recommendation data")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/infer")
async def infer_score(user_id: int = Query(1)):
    """Two-stage recommend: popular+similar candidates -> Triton ranker -> top-10."""
    total_start = time.time()
    with _span("infer_endpoint"):
        try:
            # ---- Fetch user features (recent-10 sequence + ts buckets) ----
            t0 = time.time()
            with _span("fetch_user_features"):
                async with app.state.http_session.post(
                    USER_FEATURE_URL, json={"user_id": user_id}, timeout=3
                ) as response:
                    response.raise_for_status()
                    user_features = await response.json()
            logger.info("[TIMING] Fetch user_features: %.3fs", time.time() - t0)

            seq_csv = user_features.get("user_rating_list_10_recent_movie", [""])[0]
            input_seq = [s for s in seq_csv.split(",") if s] if seq_csv else []
            input_seq_ts = user_features.get("item_sequence_ts_bucket", [[]])[0]
            input_seq_ts = [int(t) for t in input_seq_ts] if input_seq_ts else []

            # ---- Popular candidates (Redis) ----
            t2 = time.time()
            with _span("fetch_popular_items"):
                popular_items = redis_client.zrevrange(POPULAR_REDIS_KEY, 0, 19)
            logger.info("[TIMING] Fetch popular_items (Redis): %.3fs", time.time() - t2)

            # ---- Similar items for last interacted movie (Redis) ----
            last_item = input_seq[-1] if input_seq else None
            similar_items: List[str] = []
            if last_item:
                key = f"rec:{last_item}"
                with _span("fetch_similar_items"):
                    rec_json = redis_client.get(key)
                    if rec_json:
                        try:
                            similar_items = [
                                str(s) for s in json.loads(rec_json).get("rec_item_ids", [])
                            ]
                        except json.JSONDecodeError as exc:
                            logger.error("JSON decode error for %s: %s", key, exc)

            # ---- Candidate set (popular ∪ similar, minus seen) ----
            seen_set = set(input_seq)
            candidates = list({*popular_items, *similar_items} - seen_set)[:MAX_CANDIDATES]
            if not candidates:
                logger.warning("No valid candidates found")
                raise HTTPException(status_code=404, detail="No valid candidates")

            # ---- Static per-user inputs (repeated across the candidate batch) ----
            padded_seq = input_seq[-SEQ_LEN:] + ["-1"] * max(0, SEQ_LEN - len(input_seq))
            padded_seq_ts = input_seq_ts[-SEQ_LEN:] + [-1] * max(0, SEQ_LEN - len(input_seq_ts))
            user_ids_infer = np.array([str(user_id)], dtype=object).reshape(1, MAX_ITEM_ID_LEN)
            input_seq_array = np.array([padded_seq], dtype=object)
            input_seq_ts_array = np.array([padded_seq_ts], dtype=np.int64)

            # ---- Fetch item features concurrently (with Redis cache) ----
            t6 = time.time()

            async def fetch_item_features(item, semaphore):
                cache_key = f"item_features:{item}"
                with _span(f"fetch_item_features_{item}"):
                    cached = redis_client.get(cache_key)
                    if cached:
                        try:
                            return item, json.loads(cached)
                        except json.JSONDecodeError:
                            logger.error("Invalid cached data for %s", item)
                    timeout = ClientTimeout(total=2, connect=0.5, sock_read=1.5)
                    async with semaphore:
                        for attempt in range(3):
                            try:
                                async with app.state.http_session.post(
                                    ITEM_FEATURE_URL, json={"movie_id": int(item)}, timeout=timeout
                                ) as response:
                                    response.raise_for_status()
                                    features = await response.json()
                                    try:
                                        redis_client.setex(cache_key, 300, json.dumps(features))
                                    except redis.RedisError as exc:
                                        logger.error("Failed to cache item features for %s: %s", item, exc)
                                    return item, features
                            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                                logger.warning("Attempt %d failed for %s: %s", attempt + 1, item, exc)
                                if attempt == 2:
                                    logger.error("Failed to fetch item features for %s: %s", item, exc)
                                    return item, None
                                await asyncio.sleep(0.1 * (attempt + 1))

            semaphore = asyncio.Semaphore(MAX_SEMAPHORE)
            item_features_results = await asyncio.gather(
                *[fetch_item_features(c, semaphore) for c in candidates],
                return_exceptions=True,
            )
            logger.info("[TIMING] Fetch %d item_features (async): %.3fs",
                        len(candidates), time.time() - t6)

            valid_item_features = [
                r for r in item_features_results
                if isinstance(r, tuple) and r[1] is not None
            ]
            if not valid_item_features:
                logger.warning("No valid item features found")
                raise HTTPException(status_code=404, detail="No valid item features found")

            # ---- Triton inference, batched ----
            t8 = time.time()

            async def infer_batch(batch_items, batch_features):
                with _span("triton_inference_batch"):
                    batch_size = len(batch_items)
                    # target_items = candidate movieId strings (id_mapper -> index)
                    target_items = np.array(batch_items, dtype=object).reshape(batch_size, MAX_ITEM_ID_LEN)
                    # titles / genres = real movie metadata strings (item_pipeline)
                    titles = np.array(
                        [f.get("title", [""])[0] for f in batch_features],
                        dtype=object,
                    ).reshape(batch_size, MAX_ITEM_ID_LEN)
                    genres = np.array(
                        [f.get("genres", [""])[0] for f in batch_features],
                        dtype=object,
                    ).reshape(batch_size, MAX_ITEM_ID_LEN)
                    cnt_90d = np.array(
                        [safe_int(f.get("movie_rating_cnt_90d", [0])[0]) for f in batch_features],
                        dtype=np.int64,
                    ).reshape(batch_size, MAX_ITEM_ID_LEN)
                    avg_90d = np.array(
                        [safe_float(f.get("movie_rating_avg_prev_rating_90d", [0.0])[0]) for f in batch_features],
                        dtype=np.float32,
                    ).reshape(batch_size, MAX_ITEM_ID_LEN)
                    cnt_30d = np.array(
                        [safe_int(f.get("movie_rating_cnt_30d", [0])[0]) for f in batch_features],
                        dtype=np.int64,
                    ).reshape(batch_size, MAX_ITEM_ID_LEN)
                    avg_30d = np.array(
                        [safe_float(f.get("movie_rating_avg_prev_rating_30d", [0.0])[0]) for f in batch_features],
                        dtype=np.float32,
                    ).reshape(batch_size, MAX_ITEM_ID_LEN)
                    cnt_7d = np.array(
                        [safe_int(f.get("movie_rating_cnt_7d", [0])[0]) for f in batch_features],
                        dtype=np.int64,
                    ).reshape(batch_size, MAX_ITEM_ID_LEN)
                    avg_7d = np.array(
                        [safe_float(f.get("movie_rating_avg_prev_rating_7d", [0.0])[0]) for f in batch_features],
                        dtype=np.float32,
                    ).reshape(batch_size, MAX_ITEM_ID_LEN)

                    user_ids = np.repeat(user_ids_infer, batch_size, axis=0)
                    input_seq_t = np.repeat(input_seq_array, batch_size, axis=0)
                    input_seq_ts_bucket = np.repeat(input_seq_ts_array, batch_size, axis=0)

                    input_data = {
                        "user_ids": user_ids,
                        "input_seq": input_seq_t,
                        "input_seq_ts_bucket": input_seq_ts_bucket,
                        "target_items": target_items,
                        "titles": titles,
                        "genres": genres,
                        "movie_rating_cnt_90d": cnt_90d,
                        "movie_rating_avg_prev_rating_90d": avg_90d,
                        "movie_rating_cnt_30d": cnt_30d,
                        "movie_rating_avg_prev_rating_30d": avg_30d,
                        "movie_rating_cnt_7d": cnt_7d,
                        "movie_rating_avg_prev_rating_7d": avg_7d,
                    }

                    inputs = []
                    for name, data in input_data.items():
                        dtype = data.dtype
                        if dtype == object:
                            dtype_str = "BYTES"
                        elif dtype == np.int64:
                            dtype_str = "INT64"
                        elif dtype == np.float32:
                            dtype_str = "FP32"
                        else:
                            raise ValueError(f"Unsupported dtype {dtype} for {name}")
                        inp = InferInput(name, data.shape, dtype_str)
                        inp.set_data_from_numpy(data)
                        inputs.append(inp)

                    outputs = [InferRequestedOutput("output")]
                    try:
                        response = await app.state.triton_client.infer(
                            model_name=MODEL_NAME, inputs=inputs, outputs=outputs,
                        )
                        scores = response.as_numpy("output").flatten()
                        return [
                            {"movie_id": int(item), "score": float(score)}
                            for item, score in zip(batch_items, scores)
                        ]
                    except Exception as exc:
                        logger.error("Triton inference failed for batch: %s", exc)
                        return []

            inference_results = []
            for i in range(0, len(valid_item_features), BATCH_SIZE):
                batch = valid_item_features[i:i + BATCH_SIZE]
                batch_items = [item for item, _ in batch]
                batch_features = [features for _, features in batch]
                inference_results.extend(await infer_batch(batch_items, batch_features))
            logger.info("[TIMING] Triton inference %d items (batched): %.3fs",
                        len(valid_item_features), time.time() - t8)

            if not inference_results:
                logger.warning("No valid scores computed")
                raise HTTPException(status_code=404, detail="No valid scores computed")
            inference_results.sort(key=lambda x: x["score"], reverse=True)
            logger.info("[TIMING] Total /infer request: %.3fs", time.time() - total_start)
            return {"user_id": user_id, "recommendations": inference_results[:10]}

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Error in /infer: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")