"""Run 005 negative sampling headlessly (replicates feature/engineer/005)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from feature.features.negative_sampling import generate_negative_samples  # noqa: E402

OUTPUT_DIR = ROOT / "feature" / "output"
ENGINEER_DIR = OUTPUT_DIR / "engineer"
USER_COL, ITEM_COL, RATING_COL, TS_COL = "userId", "movieId", "rating", "event_timestamp"
WINDOWS = [90, 30, 7]
SEED = 42


def main() -> None:
    train_df = pd.read_parquet(ENGINEER_DIR / "train_features.parquet")
    val_df = pd.read_parquet(ENGINEER_DIR / "val_features.parquet")
    logger.info("train={} val={}", train_df.shape, val_df.shape)
    val_timestamp = val_df[TS_COL].min()
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info("full_df={}", full_df.shape)

    meta_features = ["title", "genres"]
    item_features_df = full_df.drop_duplicates(subset=[ITEM_COL])[
        [ITEM_COL, "item_indice", *meta_features]
    ].copy()

    transfer_features = [
        "item_sequence", "user_rating_list_10_recent_movie_timestamp",
        "item_sequence_ts", "item_sequence_ts_bucket", USER_COL,
        "user_rating_cnt_90d", "user_rating_avg_prev_rating_90d",
        "user_rating_list_10_recent_movie",
    ]
    neg_df = generate_negative_samples(
        full_df, user_col="user_indice", item_col="item_indice",
        label_col=RATING_COL, timestamp_col=TS_COL, neg_label=0,
        neg_to_pos_ratio=1, seed=SEED, features=transfer_features,
    )
    neg_df = neg_df.merge(item_features_df, how="left", on="item_indice", validate="m:1")
    logger.info("neg_df={} cols={}", neg_df.shape, list(neg_df.columns))

    item_ts_cols = [f"movie_rating_cnt_{d}d" for d in WINDOWS] + \
                   [f"movie_rating_avg_prev_rating_{d}d" for d in WINDOWS]
    movie_stats = pd.read_parquet(OUTPUT_DIR / "movie_rating_stats.parquet")
    ms_sorted = movie_stats.sort_values(TS_COL)
    neg_sorted = neg_df[[ITEM_COL, TS_COL]].drop_duplicates().sort_values(TS_COL)
    neg_ts_df = pd.merge_asof(
        neg_sorted, ms_sorted[[ITEM_COL, TS_COL, *item_ts_cols]],
        on=TS_COL, by=ITEM_COL, direction="backward",
    )
    neg_df = neg_df.merge(neg_ts_df, how="left", on=[ITEM_COL, TS_COL], validate="m:1")
    neg_df[item_ts_cols] = neg_df[item_ts_cols].fillna(0)
    logger.info("neg_df after ts merge={}", neg_df.shape)

    full_features_df = (
        pd.concat([full_df, neg_df], axis=0).reset_index(drop=True)
        .sample(frac=1, replace=False, random_state=SEED)
    )
    key_cols = [USER_COL, ITEM_COL, "user_indice", "item_indice", "item_sequence",
                "item_sequence_ts_bucket", RATING_COL, TS_COL]
    assert full_features_df[key_cols].isna().sum().sum() == 0, "nulls in key cols"

    train_neg = full_features_df.loc[lambda d: d[TS_COL] < val_timestamp].copy()
    val_neg = full_features_df.loc[lambda d: d[TS_COL] >= val_timestamp].copy()
    logger.info("train_neg={} val_neg={}", train_neg.shape, val_neg.shape)

    full_features_df.to_parquet(ENGINEER_DIR / "full_features_neg_sampling_df.parquet", index=False)
    train_neg.to_parquet(ENGINEER_DIR / "train_features_neg_df.parquet", index=False)
    val_neg.to_parquet(ENGINEER_DIR / "val_features_neg_df.parquet", index=False)
    logger.info("005 persisted OK")


if __name__ == "__main__":
    main()