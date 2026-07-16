"""Builds feature/etl/003-feature-etl.ipynb from inline cell sources.

Run:  uv run python feature/etl/_build_003.py
Throwaway builder — not part of the pipeline.
"""

import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md("""# 003 — Feature ETL (MovieLens, local pandas)

Compute the aggregate feature parquets that the Feast Feature Store serves,
porting the reference project's AWS Glue Spark job (`data_pipeline_aws/glue.py`)
to plain pandas — MovieLens is small (≈80k ratings), so Spark is unnecessary.

Outputs (to `feature/output/`):
- `movie_rating_stats.parquet` — per (movie, event): rating cnt/avg over the
  previous 90/30/7 days (excluding the current event) + title + genres.
- `user_rating_stats.parquet`  — per (user, event): 90-day cnt/avg, the 10 most
  recent rated movies (ids + timestamps), and the padded timestamp sequence +
  its time-difference buckets used by the sequence ranking model.
- `train.parquet` / `val.parquet` — copied from `notebooks/001` output with an
  added `event_timestamp` datetime column for downstream joins.

These two aggregate parquets back `movie_feature_view` and `user_feature_view`
in `feature/feature_store/`.""")

code("""import os
import json
import shutil

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

# Notebook lives in feature/etl/; project root is two levels up.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
import sys
sys.path.insert(0, _PROJECT_ROOT)

from feature.features.timestamp_bucket import bucketize_seconds_diff


class Args(BaseModel):
    run_name: str = "003-feature-etl"
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    user_col: str = "userId"
    item_col: str = "movieId"
    rating_col: str = "rating"
    timestamp_col: str = "timestamp"          # epoch seconds in the source parquet

    windows_days: list = [90, 30, 7]           # rating-aggregate lookback windows
    sequence_length: int = 10                  # recent-items sequence length

    # populated by init() — must be declared as fields for pydantic v2 assignment
    project_root: str = ""
    data_dir: str = ""
    output_dir: str = ""

    def init(self):
        self.project_root = _PROJECT_ROOT
        self.data_dir = os.path.join(self.project_root, "notebooks", "data", "001-prepare-dataset")
        self.output_dir = os.path.join(self.project_root, "feature", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        return self


args = Args().init()
logger.info(f"data_dir={args.data_dir}")
logger.info(f"output_dir={args.output_dir}")
print(json.dumps({"windows_days": args.windows_days, "sequence_length": args.sequence_length}, indent=2))""")

md("""## Load interactions + metadata (from 001 output)

Merge movie `title`/`genres`, derive a naive-UTC `event_timestamp` and a
millisecond `timestamp_unix` (mirrors Glue's `unix_timestamp * 1000`).""")

code("""train_df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
val_df = pd.read_parquet(os.path.join(args.data_dir, "val.parquet"))
meta_df = pd.read_parquet(os.path.join(args.data_dir, "raw_meta.parquet"))
logger.info(f"train={train_df.shape}  val={val_df.shape}  meta={meta_df.shape}")

full_df = pd.concat(
    [train_df.assign(source="train"), val_df.assign(source="val")],
    ignore_index=True,
).merge(meta_df, how="left", on=args.item_col)

full_df["event_timestamp"] = pd.to_datetime(full_df[args.timestamp_col], unit="s", utc=True).dt.tz_localize(None)
full_df["timestamp_unix"] = full_df["event_timestamp"].astype("int64") // 10**6   # ns -> ms
full_df[args.rating_col] = full_df[args.rating_col].astype("float64")
full_df = full_df.sort_values([args.user_col, args.item_col, "timestamp_unix"]).reset_index(drop=True)

logger.info(f"full_df={full_df.shape}  cols={list(full_df.columns)}")
logger.info(f"event_timestamp range: {full_df['event_timestamp'].min()}  ->  {full_df['event_timestamp'].max()}")
full_df.head()""")

md("""## Movie rating-aggregate features

Per `(movieId, event_timestamp)` dedup (keep last `timestamp_unix`), then for
each event count/average the ratings in the prior 90/30/7 days (current event
excluded via `closed='left'`). Carry `title` and `genres` through.""")

code("""def _window_cnt_avg(tu, rating, days):
    \"\"\"Per-event count/avg of prior events within `days` days (current excluded).

    Numpy scan over ms-sorted `tu` — robust to duplicate timestamps (pandas
    time-rolling on a non-unique DatetimeIndex silently produces NaN).
    \"\"\"
    w = days * 86_400_000
    n = len(tu)
    cnt = np.zeros(n, dtype=np.int64)
    avg = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mask = (tu < tu[i]) & (tu >= tu[i] - w)
        c = int(mask.sum())
        cnt[i] = c
        if c > 0:
            avg[i] = float(rating[mask].mean())
    return cnt, avg


def compute_movie_stats(full_df, item_col, windows):
    base = full_df[[item_col, "event_timestamp", "timestamp_unix", "rating", "title", "genres"]].copy()
    base = (
        base.sort_values([item_col, "event_timestamp", "timestamp_unix"])
        .drop_duplicates([item_col, "event_timestamp"], keep="last")
        .sort_values([item_col, "timestamp_unix"])
        .reset_index(drop=True)
    )

    parts = []
    for _, g in base.groupby(item_col, sort=False):
        g = g.sort_values("timestamp_unix").reset_index(drop=True)
        tu = g["timestamp_unix"].values
        r = g["rating"].values
        for d in windows:
            cnt, avg = _window_cnt_avg(tu, r, d)
            g[f"movie_rating_cnt_{d}d"] = cnt
            g[f"movie_rating_avg_prev_rating_{d}d"] = avg
        parts.append(g)

    out = pd.concat(parts, ignore_index=True)
    keep = [item_col, "event_timestamp", "title", "genres"] + [
        c for d in windows for c in (f"movie_rating_cnt_{d}d", f"movie_rating_avg_prev_rating_{d}d")
    ]
    return out[keep]


movie_stats = compute_movie_stats(full_df, args.item_col, args.windows_days)
movie_fp = os.path.join(args.output_dir, "movie_rating_stats.parquet")
movie_stats.to_parquet(movie_fp, index=False)
logger.info(f"wrote {movie_fp}  shape={movie_stats.shape}")
logger.info(f"movie_stats cols={list(movie_stats.columns)}")
movie_stats.head()""")

md("""## User rating-aggregate + recent-sequence features

Per user (sorted by time): 90-day prev count/avg rating, the 10 most recent
rated movies (comma-joined ids + ISO timestamps), and a padded left-to-length-10
list of those events' unix-**second** timestamps plus its time-difference
buckets (the sequence model's positional encoding).

Padding (`-1`) goes on the left so the most recent event sits at the end of the
sequence — same convention as the reference `pad_timestamp_sequence`.""")

code("""def compute_user_stats(full_df, user_col, item_col, seq_len):
    base = full_df[[user_col, item_col, "event_timestamp", "timestamp_unix", "rating"]].copy()
    base = base.sort_values([user_col, "timestamp_unix"]).reset_index(drop=True)

    records = []
    for _, g in base.groupby(user_col, sort=False):
        g = g.sort_values("timestamp_unix").reset_index(drop=True)
        tu = g["timestamp_unix"].values
        et = g["event_timestamp"].values
        items = g[item_col].values
        r = g["rating"].values

        cnt90, avg90 = _window_cnt_avg(tu, r, 90)

        n = len(g)
        rec_movie, rec_ts, seq_ts, seq_bucket = [], [], [], []
        for i in range(n):
            sl = slice(max(0, i - seq_len), i)
            prev_items = items[sl]
            prev_ets = et[sl]
            prev_tu = tu[sl]

            rec_movie.append(",".join(str(int(x)) for x in prev_items))
            rec_ts.append(",".join(pd.Timestamp(x).strftime("%Y-%m-%dT%H:%M:%S.%fZ") for x in prev_ets))

            sec_list = (prev_tu // 1000).astype(int).tolist()       # ms -> seconds
            pad_needed = seq_len - len(sec_list)
            if pad_needed > 0:
                sec_list = [-1] * pad_needed + sec_list             # left-pad
            sec_list = sec_list[:seq_len]
            seq_ts.append(sec_list)

            cur_sec = int(tu[i]) // 1000
            seq_bucket.append([
                bucketize_seconds_diff(cur_sec - s) if s != -1 else -1
                for s in sec_list
            ])

        records.append(pd.DataFrame({
            user_col: g[user_col].values,
            "event_timestamp": et,
            "user_rating_cnt_90d": cnt90,
            "user_rating_avg_prev_rating_90d": avg90,
            "user_rating_list_10_recent_movie": rec_movie,
            "user_rating_list_10_recent_movie_timestamp": rec_ts,
            "item_sequence_ts": seq_ts,
            "item_sequence_ts_bucket": seq_bucket,
        }))

    return pd.concat(records, ignore_index=True)


user_stats = compute_user_stats(full_df, args.user_col, args.item_col, args.sequence_length)
user_fp = os.path.join(args.output_dir, "user_rating_stats.parquet")
user_stats.to_parquet(user_fp, index=False)
logger.info(f"wrote {user_fp}  shape={user_stats.shape}")
logger.info(f"user_stats cols={list(user_stats.columns)}")
user_stats.head()""")

md("""## Copy train/val interactions to feature/output

Add an `event_timestamp` datetime column (from epoch seconds) so downstream
notebooks join on a real timestamp. Keep `userId`, `movieId`, `rating`,
`event_timestamp`, `source`.""")

code("""def _copy_with_event_ts(name, source):
    src = os.path.join(args.data_dir, name)
    dst = os.path.join(args.output_dir, name)
    df = pd.read_parquet(src)
    df["event_timestamp"] = pd.to_datetime(df[args.timestamp_col], unit="s", utc=True).dt.tz_localize(None)
    df["source"] = source
    keep = [args.user_col, args.item_col, args.rating_col, "event_timestamp", "source"]
    df[keep].to_parquet(dst, index=False)
    logger.info(f"copied {name} -> {dst}  shape={df[keep].shape}")
    return dst


train_dst = _copy_with_event_ts("train.parquet", "train")
val_dst = _copy_with_event_ts("val.parquet", "val")""")

md("""## Verify outputs

Sanity-check shapes, columns, and a couple of feature values before Feast
registration (`feature/feature_store` -> `feast apply`).""")

code("""for f in ["movie_rating_stats.parquet", "user_rating_stats.parquet", "train.parquet", "val.parquet"]:
    df = pd.read_parquet(os.path.join(args.output_dir, f))
    logger.info(f"{f}: shape={df.shape}  cols={list(df.columns)}")

# Spot check: a user with many ratings should have a non-trivial recent-10 list.
busy = user_stats.loc[user_stats["user_rating_cnt_90d"].idxmax()]
logger.info(
    f"busiest user event: user={int(busy[args.user_col])} "
    f"cnt_90d={int(busy['user_rating_cnt_90d'])} "
    f"recent_movies={busy['user_rating_list_10_recent_movie'][:80]}..."
)
assert movie_stats["movie_rating_cnt_90d"].between(0, 100000).all(), "movie cnt out of range"
assert user_stats["user_rating_cnt_90d"].between(0, 100000).all(), "user cnt out of range"
assert user_stats["item_sequence_ts"].apply(len).eq(args.sequence_length).all(), "sequence length mismatch"
logger.info("verification OK")""")

md("""## Summary

- `movie_rating_stats.parquet` and `user_rating_stats.parquet` written to
  `feature/output/` — the two Feast offline-store sources.
- `train.parquet` / `val.parquet` copied to `feature/output/` with an
  `event_timestamp` column for downstream joins.
- All window aggregates and the 10-event sequences computed in pandas (no
  Spark / no AWS Glue).

Next: from `feature/feature_store/` run `feast apply` then
`feast materialize-incremental` (or `feast materialize <start> <end>`) to push
the online-serving views into Redis, then run `004-features.ipynb`.""")

nb.cells = cells
out_path = os.path.join(os.path.dirname(__file__), "003-feature-etl.ipynb")
nbf.write(nb, out_path)
print(f"wrote {out_path}  ({len(cells)} cells)")