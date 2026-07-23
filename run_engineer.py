"""Run 004 feature engineer headlessly.

Step 1: replicate 003's `_copy_with_event_ts` to produce feature/output/
{train,val}.parquet (with event_timestamp + source) from the raw data in
notebooks/data/001-prepare-dataset/. The event_timestamp conversion is identical
to 003 so the (movieId, event_timestamp) / (userId, event_timestamp) merges
against the existing movie/user_rating_stats.parquet match exactly.

Step 2: run the 004-features logic (IDMapper, movie+user merges, item_sequence,
train/val split, item_metadata_pipeline fit) and write
feature/output/engineer/{train_features,val_features,idm,item_metadata_pipeline}.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from feature.id_mapper import IDMapper, map_indice  # noqa: E402
from feature.features.tfm import (  # noqa: E402
    title_pipeline_steps, genres_pipeline_steps, rating_agg_pipeline_steps,
)

OUTPUT_DIR = ROOT / "feature" / "output"
ENGINEER_DIR = OUTPUT_DIR / "engineer"
DATA_DIR = ROOT / "notebooks" / "data" / "001-prepare-dataset"
USER_COL, ITEM_COL, RATING_COL, TS_COL = "userId", "movieId", "rating", "timestamp"
EVENT_TS = "event_timestamp"
SEQ_LEN = 10
WINDOWS = [90, 30, 7]


def copy_with_event_ts(name: str, source: str) -> None:
    src = DATA_DIR / name
    dst = OUTPUT_DIR / name
    df = pd.read_parquet(src)
    df[EVENT_TS] = pd.to_datetime(df[TS_COL], unit="s", utc=True).dt.tz_localize(None)
    df["source"] = source
    keep = [USER_COL, ITEM_COL, RATING_COL, EVENT_TS, "source"]
    df[keep].to_parquet(dst, index=False)
    logger.info("copied {} -> {} shape={}", name, dst, df[keep].shape)


def convert_movie_to_idx(inp, idm, sequence_length=SEQ_LEN, padding_value=-1):
    if inp is None or (isinstance(inp, float) and np.isnan(inp)) or str(inp).strip() == "":
        return [padding_value] * sequence_length
    movie_ids = [int(x) for x in str(inp).split(",") if x.strip()]
    indices = [idm.get_item_index(mid) for mid in movie_ids]
    pad_needed = sequence_length - len(indices)
    if pad_needed > 0:
        indices = [padding_value] * pad_needed + indices
    return indices[:sequence_length]


def main() -> None:
    ENGINEER_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: produce feature/output/{train,val}.parquet (matches 003's copy step).
    copy_with_event_ts("train.parquet", "train")
    copy_with_event_ts("val.parquet", "val")

    # Step 2: 004 engineer.
    train_df = pd.read_parquet(OUTPUT_DIR / "train.parquet")
    val_df = pd.read_parquet(OUTPUT_DIR / "val.parquet")
    logger.info("train={} val={}", train_df.shape, val_df.shape)

    full_df = pd.concat([train_df, val_df], ignore_index=True)
    unique_user_ids = sorted(train_df[USER_COL].unique())
    unique_item_ids = sorted(train_df[ITEM_COL].unique())
    idm = IDMapper().fit(unique_user_ids, unique_item_ids)
    logger.info("IDMapper: users={} items={}", len(unique_user_ids), len(unique_item_ids))

    full_df = map_indice(full_df, idm, USER_COL, ITEM_COL)
    logger.info("full_df={} cols={}", full_df.shape, list(full_df.columns))

    movie_stats = pd.read_parquet(OUTPUT_DIR / "movie_rating_stats.parquet")
    full_df = full_df.merge(movie_stats, how="left", on=[ITEM_COL, EVENT_TS], validate="m:1")
    logger.info("after movie merge: {}", full_df.shape)
    assert full_df["title"].notna().all(), "movie merge left nulls"

    user_stats = pd.read_parquet(OUTPUT_DIR / "user_rating_stats.parquet")
    user_stats = user_stats.drop_duplicates([USER_COL, EVENT_TS], keep="last").reset_index(drop=True)
    full_df = full_df.merge(user_stats, how="left", on=[USER_COL, EVENT_TS], validate="m:1")
    logger.info("after user merge: {} cols={}", full_df.shape, list(full_df.columns))
    assert full_df["user_rating_cnt_90d"].notna().all(), "user merge left nulls"

    full_df = full_df.assign(
        item_sequence=lambda d: d["user_rating_list_10_recent_movie"].apply(
            lambda x: convert_movie_to_idx(x, idm)
        )
    )
    logger.info("built item_sequence")

    val_timestamp = val_df[EVENT_TS].min()
    train_out = full_df.loc[lambda d: d[EVENT_TS] < val_timestamp].copy()
    val_out = full_df.loc[lambda d: d[EVENT_TS] >= val_timestamp].copy()
    logger.info("train_out={} val_out={}", train_out.shape, val_out.shape)
    assert train_out.shape[0] == len(train_df), "train row count drifted"
    assert val_out.shape[0] == len(val_df), "val row count drifted"

    for df in (train_out, val_out):
        n = df[[USER_COL, ITEM_COL, EVENT_TS]].duplicated().sum()
        assert n == 0, f"{n} duplicates"

    train_out.to_parquet(ENGINEER_DIR / "train_features.parquet", index=False)
    val_out.to_parquet(ENGINEER_DIR / "val_features.parquet", index=False)
    idm.save(ENGINEER_DIR / "idm.json")
    logger.info("wrote train_features/val_features/idm to {}", ENGINEER_DIR)

    rating_agg_cols = [c for d in WINDOWS for c in (f"movie_rating_cnt_{d}d", f"movie_rating_avg_prev_rating_{d}d")]
    tfm = [
        ("title", Pipeline(title_pipeline_steps()), ["title"]),
        ("genres", Pipeline(genres_pipeline_steps()), "genres"),
        ("rating_agg", Pipeline(rating_agg_pipeline_steps()), rating_agg_cols),
    ]
    preprocessing = ColumnTransformer(transformers=tfm, remainder="drop")
    item_metadata_pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("normalizer", StandardScaler())])
    fit_df = train_out.drop_duplicates(subset=[ITEM_COL])
    item_metadata_pipeline.fit(fit_df)
    logger.info("fit pipeline on {} unique movies", len(fit_df))
    with open(ENGINEER_DIR / "item_metadata_pipeline.dill", "wb") as f:
        dill.dump(item_metadata_pipeline, f)
    logger.info("wrote item_metadata_pipeline.dill")

    key_cols = [USER_COL, ITEM_COL, "user_indice", "item_indice", "item_sequence",
                "item_sequence_ts_bucket", RATING_COL, EVENT_TS]
    nulls = train_out[key_cols].isna().sum().sum() + val_out[key_cols].isna().sum().sum()
    logger.info("key-col nulls={}", nulls)
    assert nulls == 0, "nulls in key columns"
    missing = set(val_out[ITEM_COL].unique()) - set(train_out[ITEM_COL].unique())
    assert not missing, f"{len(missing)} cold-start val items"
    logger.info("004 OK: train={} val={} cols={}", train_out.shape, val_out.shape, list(train_out.columns))


if __name__ == "__main__":
    main()