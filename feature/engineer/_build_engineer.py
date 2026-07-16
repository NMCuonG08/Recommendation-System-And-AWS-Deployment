"""Builds 004-features / 005-negative-sample / 006-prep-item2vec notebooks.

Run:  uv run python feature/engineer/_build_engineer.py
Throwaway builder — not part of the pipeline.
"""

import os

import nbformat as nbf


def build(path, cells_src):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    cells = []
    for kind, src in cells_src:
        cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nb.cells = cells
    nbf.write(nb, path)
    print(f"wrote {path}  ({len(cells)} cells)")


# ============================================================ 004-features
c004 = []
c004.append(("md", """# 004 — Features (MovieLens, Feast offline join)

Port of the reference `src/feature_engineer/001-features.ipynb`, adapted to
MovieLens and the local Feast repo (`feature/feature_store`).

Reads the interaction parquets from `feature/output/` (produced by `003`),
pulls **movie** and **user** aggregate features from Feast via point-in-time
`get_historical_features`, maps raw ids to dense indices (`IDMapper`), builds
the recent-10 `item_sequence`, fits an sklearn `ColumnTransformer`
(title TF-IDF + genres count-vectorizer + rating-aggregate StandardScaler),
and persists `train_features.parquet` / `val_features.parquet` +
`idm.json` + `item_metadata_pipeline.dill` to `feature/output/engineer/`."""))

c004.append(("code", """import os
import json

import numpy as np
import pandas as pd
import dill
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
import sys
sys.path.insert(0, _PROJECT_ROOT)

from feature.id_mapper import IDMapper, map_indice
from feature.features.tfm import title_pipeline_steps, genres_pipeline_steps, rating_agg_pipeline_steps


class Args(BaseModel):
    run_name: str = "004-features"
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    user_col: str = "userId"
    item_col: str = "movieId"
    rating_col: str = "rating"
    timestamp_col: str = "event_timestamp"
    sequence_length: int = 10
    windows_days: list = [90, 30, 7]
    # populated by init()
    project_root: str = ""
    output_dir: str = ""
    engineer_dir: str = ""

    def init(self):
        self.project_root = _PROJECT_ROOT
        self.output_dir = os.path.join(self.project_root, "feature", "output")
        self.engineer_dir = os.path.join(self.output_dir, "engineer")
        os.makedirs(self.engineer_dir, exist_ok=True)
        return self


args = Args().init()
logger.info(f"output_dir={args.output_dir}  engineer_dir={args.engineer_dir}")
print(json.dumps({"sequence_length": args.sequence_length, "windows_days": args.windows_days}, indent=2))"""))

c004.append(("md", "## Load interactions + IDMapper\n\nConcatenate train/val, fit the id mapper on sorted train ids, attach dense\n`user_indice` / `item_indice`."))

c004.append(("code", """train_df = pd.read_parquet(os.path.join(args.output_dir, "train.parquet"))
val_df = pd.read_parquet(os.path.join(args.output_dir, "val.parquet"))
logger.info(f"train={train_df.shape}  val={val_df.shape}")

full_df = pd.concat([train_df, val_df], ignore_index=True)
unique_user_ids = sorted(train_df[args.user_col].unique())
unique_item_ids = sorted(train_df[args.item_col].unique())
idm = IDMapper().fit(unique_user_ids, unique_item_ids)
logger.info(f"IDMapper: users={len(unique_user_ids)}  items={len(unique_item_ids)}")

full_df = map_indice(full_df, idm, args.user_col, args.item_col)
logger.info(f"full_df={full_df.shape}  cols={list(full_df.columns)}")
full_df.head()"""))

c004.append(("md", "## Merge movie features\n\n`003` already computed per-`(movieId, event_timestamp)` aggregates, so a plain\nleft-merge on `(movieId, event_timestamp)` is an exact point-in-time match — no\nasof needed. (Feast's file offline store wraps this in a dask asof-join that\nOOMs at 79k rows; the Feast repo stays for online/Redis serving only.)"))

c004.append(("code", """movie_stats = pd.read_parquet(os.path.join(args.output_dir, "movie_rating_stats.parquet"))
logger.info(f"movie_stats={movie_stats.shape}  cols={list(movie_stats.columns)}")

full_features_df = full_df.merge(movie_stats, how="left", on=[args.item_col, args.timestamp_col], validate="m:1")
logger.info(f"after movie merge: {full_features_df.shape}")
assert full_features_df["title"].notna().all(), "movie merge left nulls"
full_features_df.head()"""))

c004.append(("md", "## Merge user features + build item_sequence\n\nUser 90-day aggregates, the recent-10 movie list, and the padded timestamp\nsequence + buckets — same exact `(userId, event_timestamp)` match. `item_sequence`\nconverts the recent-10 movie ids to dense indices (left-padded -1)."))

c004.append(("code", """user_stats = pd.read_parquet(os.path.join(args.output_dir, "user_rating_stats.parquet"))
# MovieLens records batch ratings with the same (userId, event_timestamp) second.
# 003 emits one row per event; collapse to one per (userId, event_timestamp)
# keeping the LAST event (the most complete recent-10 list) so the join is m:1.
user_stats = user_stats.drop_duplicates([args.user_col, args.timestamp_col], keep="last").reset_index(drop=True)
logger.info(f"user_stats (deduped)={user_stats.shape}  cols={list(user_stats.columns)}")

full_features_df = full_features_df.merge(user_stats, how="left", on=[args.user_col, args.timestamp_col], validate="m:1")
logger.info(f"after user merge: {full_features_df.shape}  cols={list(full_features_df.columns)}")
assert full_features_df["user_rating_cnt_90d"].notna().all(), "user merge left nulls"


def convert_movie_to_idx(inp, sequence_length=10, padding_value=-1):
    if inp is None or (isinstance(inp, float) and np.isnan(inp)) or str(inp).strip() == "":
        return [padding_value] * sequence_length
    movie_ids = [int(x) for x in str(inp).split(",") if x.strip()]
    indices = [idm.get_item_index(mid) for mid in movie_ids]
    pad_needed = sequence_length - len(indices)
    if pad_needed > 0:
        indices = [padding_value] * pad_needed + indices
    return indices[:sequence_length]


full_features_df = full_features_df.assign(
    item_sequence=lambda d: d["user_rating_list_10_recent_movie"].apply(convert_movie_to_idx)
)
logger.info("built item_sequence")
full_features_df[["user_rating_list_10_recent_movie", "item_sequence"]].head()"""))

c004.append(("md", "## Split back to train / val + persist\n\nSame temporal split as `003`: `event_timestamp < val_timestamp` -> train."))

c004.append(("code", """val_timestamp = val_df[args.timestamp_col].min()
logger.info(f"val_timestamp={val_timestamp}")

train_out = full_features_df.loc[lambda d: d[args.timestamp_col] < val_timestamp].copy()
val_out = full_features_df.loc[lambda d: d[args.timestamp_col] >= val_timestamp].copy()
logger.info(f"train_out={train_out.shape}  val_out={val_out.shape}")
assert train_out.shape[0] == len(train_df), "train row count drifted"
assert val_out.shape[0] == len(val_df), "val row count drifted"


def check_dup(df):
    n = df[[args.user_col, args.item_col, args.timestamp_col]].duplicated().sum()
    assert n == 0, f"{n} duplicates"


check_dup(train_out)
check_dup(val_out)

train_fp = os.path.join(args.engineer_dir, "train_features.parquet")
val_fp = os.path.join(args.engineer_dir, "val_features.parquet")
idm_fp = os.path.join(args.engineer_dir, "idm.json")
train_out.to_parquet(train_fp, index=False)
val_out.to_parquet(val_fp, index=False)
idm.save(idm_fp)
logger.info(f"wrote {train_fp}  {val_fp}  {idm_fp}")"""))

c004.append(("md", "## Fit sklearn metadata pipeline\n\n`ColumnTransformer` over title (TF-IDF), genres (multi-label count vectorizer),\nand the 6 movie rating aggregates (StandardScaler). Fit on unique movies in\ntrain, persist with `dill` for reuse at training/serving time."))

c004.append(("code", """rating_agg_cols = [c for d in args.windows_days for c in (f"movie_rating_cnt_{d}d", f"movie_rating_avg_prev_rating_{d}d")]
logger.info(f"rating_agg_cols={rating_agg_cols}")

tfm = [
    ("title", Pipeline(title_pipeline_steps()), ["title"]),
    ("genres", Pipeline(genres_pipeline_steps()), "genres"),
    ("rating_agg", Pipeline(rating_agg_pipeline_steps()), rating_agg_cols),
]

preprocessing = ColumnTransformer(transformers=tfm, remainder="drop")
item_metadata_pipeline = Pipeline(steps=[("preprocessing", preprocessing), ("normalizer", StandardScaler())])

fit_df = train_out.drop_duplicates(subset=[args.item_col])
item_metadata_pipeline.fit(fit_df)
logger.info(f"fit pipeline on {len(fit_df)} unique movies")

dill_fp = os.path.join(args.engineer_dir, "item_metadata_pipeline.dill")
with open(dill_fp, "wb") as f:
    dill.dump(item_metadata_pipeline, f)
logger.info(f"wrote {dill_fp}")"""))

c004.append(("md", "## Verify\n\nSanity-check shapes, nulls on key columns, and that val items/users are all in train (no cold-start)."))

c004.append(("code", """key_cols = [args.user_col, args.item_col, "user_indice", "item_indice", "item_sequence", "item_sequence_ts_bucket", args.rating_col, args.timestamp_col]
nulls = train_out[key_cols].isna().sum().sum() + val_out[key_cols].isna().sum().sum()
logger.info(f"key-col nulls={nulls}")
assert nulls == 0, "nulls in key columns"

train_items = set(train_out[args.item_col].unique())
val_items = set(val_out[args.item_col].unique())
missing = val_items - train_items
logger.info(f"val items not in train: {len(missing)}")
assert not missing, f"{len(missing)} cold-start val items"

logger.info(f"train_features={train_out.shape}  val_features={val_out.shape}  cols={list(train_out.columns)}")
logger.info("004 verification OK")"""))

build(os.path.join(os.path.dirname(__file__), "004-features.ipynb"), c004)


# ============================================================ 005-negative-sample
c005 = []
c005.append(("md", """# 005 — Negative Sampling (MovieLens)

Port of the reference `011-negative-sample.ipynb`. For each positive
user–movie interaction, sample one popularity-weighted negative movie the
user did NOT interact with. Negative rows inherit the user-side features of
their positive row and pick up the item-side features (title, genres) of the
sampled movie. The time-dependent rating aggregates for the sampled movie at
the negative event timestamp come from a pandas `merge_asof` (latest movie
stats at-or-before that timestamp) — the point-in-time join Feast's file
offline store would do, but without the dask OOM.

Outputs `train_features_neg_df.parquet` / `val_features_neg_df.parquet` /
`full_features_neg_sampling_df.parquet` to `feature/output/engineer/`."""))

c005.append(("code", """import os

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

_PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
import sys
sys.path.insert(0, _PROJECT_ROOT)

from feature.features.negative_sampling import generate_negative_samples


class Args(BaseModel):
    run_name: str = "005-negative-sample"
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    user_col: str = "userId"
    item_col: str = "movieId"
    rating_col: str = "rating"
    timestamp_col: str = "event_timestamp"
    neg_to_pos_ratio: int = 1
    windows_days: list = [90, 30, 7]
    project_root: str = ""
    output_dir: str = ""
    engineer_dir: str = ""

    def init(self):
        self.project_root = _PROJECT_ROOT
        self.output_dir = os.path.join(self.project_root, "feature", "output")
        self.engineer_dir = os.path.join(self.output_dir, "engineer")
        return self


args = Args().init()
logger.info(f"engineer_dir={args.engineer_dir}")"""))

c005.append(("code", """train_df = pd.read_parquet(os.path.join(args.engineer_dir, "train_features.parquet"))
val_df = pd.read_parquet(os.path.join(args.engineer_dir, "val_features.parquet"))
logger.info(f"train={train_df.shape}  val={val_df.shape}")

val_timestamp = val_df[args.timestamp_col].min()
logger.info(f"val_timestamp={val_timestamp}")
full_df = pd.concat([train_df, val_df], ignore_index=True)
logger.info(f"full_df={full_df.shape}")"""))

c005.append(("md", "## Generate negatives\n\nStatic item metadata (title, genres) carried via `item_features_df`; the\ntime-dependent rating aggregates for the sampled negative items are fetched\nfrom Feast at the negative event timestamp."))

c005.append(("code", """meta_features = ["title", "genres"]
item_features_df = full_df.drop_duplicates(subset=[args.item_col])[
    [args.item_col, "item_indice", *meta_features]
].copy()

transfer_features = [
    "item_sequence",
    "user_rating_list_10_recent_movie_timestamp",
    "item_sequence_ts",
    "item_sequence_ts_bucket",
    args.user_col,
    "user_rating_cnt_90d",
    "user_rating_avg_prev_rating_90d",
    "user_rating_list_10_recent_movie",
]

neg_df = generate_negative_samples(
    full_df,
    user_col="user_indice",
    item_col="item_indice",
    label_col=args.rating_col,
    timestamp_col=args.timestamp_col,
    neg_label=0,
    neg_to_pos_ratio=args.neg_to_pos_ratio,
    seed=args.random_seed,
    features=transfer_features,
)
neg_df = neg_df.merge(item_features_df, how="left", on="item_indice", validate="m:1")
logger.info(f"neg_df={neg_df.shape}  cols={list(neg_df.columns)}")"""))

c005.append(("code", """item_ts_cols = [f"movie_rating_cnt_{d}d" for d in args.windows_days] + \\
               [f"movie_rating_avg_prev_rating_{d}d" for d in args.windows_days]

movie_stats = pd.read_parquet(os.path.join(args.output_dir, "movie_rating_stats.parquet"))
ms_sorted = movie_stats.sort_values(args.timestamp_col)

neg_sorted = neg_df[[args.item_col, args.timestamp_col]].drop_duplicates().sort_values(args.timestamp_col)
neg_ts_df = pd.merge_asof(
    neg_sorted, ms_sorted[[args.item_col, args.timestamp_col, *item_ts_cols]],
    on=args.timestamp_col, by=args.item_col, direction="backward",
)
logger.info(f"neg_ts_df={neg_ts_df.shape}")

neg_df = neg_df.merge(neg_ts_df, how="left", on=[args.item_col, args.timestamp_col], validate="m:1")
n_nulls = int(neg_df[item_ts_cols[0]].isna().sum())
logger.info(f"neg_df after ts merge={neg_df.shape}  cold-movie nulls filled={n_nulls}")
# A negative movie may have no ratings at-or-before the event timestamp (cold at
# that point in time) -> asof returns NaN. Fill with 0 (cnt=0, avg=0): the movie
# had no prior history in any window.
neg_df[item_ts_cols] = neg_df[item_ts_cols].fillna(0) """))

c005.append(("md", "## Concat pos + neg, shuffle, split back to train/val, persist"))

c005.append(("code", """full_features_df = (
    pd.concat([full_df, neg_df], axis=0)
    .reset_index(drop=True)
    .sample(frac=1, replace=False, random_state=args.random_seed)
)
logger.info(f"pos+neg={full_features_df.shape}")

key_cols = [args.user_col, args.item_col, "user_indice", "item_indice", "item_sequence", "item_sequence_ts_bucket", args.rating_col, args.timestamp_col]
assert full_features_df[key_cols].isna().sum().sum() == 0, "nulls in key columns"

train_neg_df = full_features_df.loc[lambda d: d[args.timestamp_col] < val_timestamp].copy()
val_neg_df = full_features_df.loc[lambda d: d[args.timestamp_col] >= val_timestamp].copy()
logger.info(f"train_neg={train_neg_df.shape}  val_neg={val_neg_df.shape}")

full_features_df.to_parquet(os.path.join(args.engineer_dir, "full_features_neg_sampling_df.parquet"), index=False)
train_neg_df.to_parquet(os.path.join(args.engineer_dir, "train_features_neg_df.parquet"), index=False)
val_neg_df.to_parquet(os.path.join(args.engineer_dir, "val_features_neg_df.parquet"), index=False)
logger.info("005 persisted OK")"""))

build(os.path.join(os.path.dirname(__file__), "005-negative-sample.ipynb"), c005)


# ============================================================ 006-prep-item2vec
c006 = []
c006.append(("md", """# 006 — Prep Item2Vec (MovieLens)

Port of the reference `021-prep-item2vec.ipynb`. Build per-user movie-id
sequences (chronological, length > 1) from the feature parquets and persist
them as JSONL for item2vec / sequence-model training, plus a 2-sequence
overfit batch."""))

c006.append(("code", """import json
import os

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

_PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))


class Args(BaseModel):
    run_name: str = "006-prep-item2vec"
    user_col: str = "userId"
    item_col: str = "movieId"
    timestamp_col: str = "event_timestamp"
    project_root: str = ""
    engineer_dir: str = ""

    def init(self):
        self.project_root = _PROJECT_ROOT
        self.engineer_dir = os.path.join(self.project_root, "feature", "output", "engineer")
        return self


args = Args().init()


def get_sequence(df, user_col, item_col, ts_col):
    return (
        df.sort_values(ts_col)
        .groupby(user_col)[item_col]
        .agg(list)
        .loc[lambda s: s.apply(len) > 1]
        .values.tolist()
    )"""))

c006.append(("code", """train_df = pd.read_parquet(os.path.join(args.engineer_dir, "train_features.parquet"))
val_df = pd.read_parquet(os.path.join(args.engineer_dir, "val_features.parquet"))
logger.info(f"train={train_df.shape}  val={val_df.shape}")

item_sequence = get_sequence(train_df, args.user_col, args.item_col, args.timestamp_col)
val_item_sequence = get_sequence(val_df, args.user_col, args.item_col, args.timestamp_col)
logger.info(f"train sequences={len(item_sequence):,}  val sequences={len(val_item_sequence):,}")"""))

c006.append(("code", """def write_jsonl(path, sequences):
    with open(path, "w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(json.dumps([int(x) for x in seq]) + "\\n")

write_jsonl(os.path.join(args.engineer_dir, "train_item_sequence.jsonl"), item_sequence)
write_jsonl(os.path.join(args.engineer_dir, "val_item_sequence.jsonl"), val_item_sequence)

# 2-sequence overfit batch
write_jsonl(os.path.join(args.engineer_dir, "batch_sequences_overfit.jsonl"), item_sequence[:2])
logger.info("006 persisted OK")"""))

build(os.path.join(os.path.dirname(__file__), "006-prep-item2vec.ipynb"), c006)