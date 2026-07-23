"""Replicate notebooks/001-prepare-dataset.ipynb without Kaggle.

Downloads MovieLens ml-latest-small directly from GroupLens and produces
notebooks/data/001-prepare-dataset/{train,val,raw_meta}.parquet with the same
k-core + temporal-split logic as 001, so downstream 003/004/train stay
consistent with the existing Feast stats parquets.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_prep_utils import filter_min_interactions, calculate_sparsity  # noqa: E402

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
PERSIST = ROOT / "notebooks" / "data" / "001-prepare-dataset"
USER_COL, ITEM_COL, RATING_COL, TS_COL = "userId", "movieId", "rating", "timestamp"
MIN_USER, MIN_ITEM, VAL_DAYS = 20, 10, 90


def main() -> None:
    PERSIST.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", URL)
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        names = z.namelist()
        ratings_name = next(n for n in names if n.endswith("ratings.csv"))
        movies_name = next(n for n in names if n.endswith("movies.csv"))
        ratings_df = pd.read_csv(z.open(ratings_name))
        movies_df = pd.read_csv(z.open(movies_name))
    logger.info("ratings=%s movies=%s", ratings_df.shape, movies_df.shape)

    ratings_df = ratings_df.assign(
        rating=lambda d: d[RATING_COL].astype("float64"),
        ts_dt=lambda d: pd.to_datetime(d[TS_COL], unit="s", utc=True),
    )
    logger.info(
        "time range: %s -> %s | users=%d items=%d ratings=%d",
        ratings_df["ts_dt"].min(), ratings_df["ts_dt"].max(),
        ratings_df[USER_COL].nunique(), ratings_df[ITEM_COL].nunique(), len(ratings_df),
    )

    before = len(ratings_df)
    filtered = filter_min_interactions(
        ratings_df, user_col=USER_COL, item_col=ITEM_COL,
        min_user=MIN_USER, min_item=MIN_ITEM,
    )
    sparsity = calculate_sparsity(filtered, user_col=USER_COL, item_col=ITEM_COL)
    logger.info(
        "k-core: %d -> %d (users=%d items=%d) sparsity=%.4f%%",
        before, len(filtered), filtered[USER_COL].nunique(),
        filtered[ITEM_COL].nunique(), sparsity * 100,
    )

    filtered = filtered.sort_values("ts_dt").reset_index(drop=True)
    val_start = filtered["ts_dt"].max() - pd.to_timedelta(VAL_DAYS, unit="D")
    train_df = filtered.loc[filtered["ts_dt"] < val_start].copy()
    val_cand = filtered.loc[filtered["ts_dt"] >= val_start].copy()
    train_users = set(train_df[USER_COL].unique())
    train_items = set(train_df[ITEM_COL].unique())
    val_df = val_cand.loc[
        val_cand[USER_COL].isin(train_users) & val_cand[ITEM_COL].isin(train_items)
    ].copy()
    # Cold-start assertions.
    assert not (set(val_df[USER_COL].unique()) - train_users), "cold-start users"
    assert not (set(val_df[ITEM_COL].unique()) - train_items), "cold-start items"
    logger.info("train=%d val=%d", len(train_df), len(val_df))

    cols = [USER_COL, ITEM_COL, RATING_COL, TS_COL]
    train_df[cols].to_parquet(PERSIST / "train.parquet", index=False)
    val_df[cols].to_parquet(PERSIST / "val.parquet", index=False)
    movies_df[[ITEM_COL, "title", "genres"]].to_parquet(
        PERSIST / "raw_meta.parquet", index=False
    )
    logger.info("wrote train/val/raw_meta to %s", PERSIST)


if __name__ == "__main__":
    main()