"""PyTorch Datasets wrapping the ranking feature DataFrame.

Ported from the reference `src/model_ranking_sequence/dataset.py` (only the
ranking datasets — the SkipGram IterableDataset belongs to item2vec).

`UserItemBinaryDFDataset` binarizes the rating column (`rating > 0` → 1.0,
else 0.0) so the ranker is trained as a binary click/no-click classifier over
the positive + popularity-negative-sampled interactions.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class UserItemRatingDFDataset(Dataset):
    """Map-style dataset over a user-item interaction DataFrame.

    Expects columns: `user_col`, `item_col`, `rating_col`, `timestamp_col`,
    `item_sequence` (list[int]), `item_sequence_ts_bucket` (list[int]).
    Optional `item_feature` is a precomputed dense feature matrix indexed by row.
    """

    def __init__(
        self,
        df,
        user_col: str,
        item_col: str,
        rating_col: str,
        timestamp_col: str,
        item_feature=None,
    ):
        self.df = df.assign(**{rating_col: df[rating_col].astype(np.float32)})
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.item_feature = item_feature

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.df[self.user_col].iloc[idx]
        item = self.df[self.item_col].iloc[idx]
        rating = self.df[self.rating_col].iloc[idx]
        item_sequence = []
        if "item_sequence" in self.df:
            item_sequence = self.df["item_sequence"].iloc[idx]
        item_sequence_ts_bucket = []
        if "item_sequence_ts_bucket" in self.df:
            item_sequence_ts_bucket = self.df["item_sequence_ts_bucket"].iloc[idx]
        item_feature = []
        if self.item_feature is not None:
            item_feature = self.item_feature[idx]
        return dict(
            user=torch.as_tensor(user),
            item=torch.as_tensor(item),
            rating=torch.as_tensor(rating),
            item_sequence=torch.tensor(item_sequence, dtype=torch.long),
            item_sequence_ts_bucket=torch.tensor(item_sequence_ts_bucket, dtype=torch.long),
            item_feature=(
                torch.as_tensor(item_feature) if self.item_feature is not None else []
            ),
        )


class UserItemBinaryDFDataset(UserItemRatingDFDataset):
    """Binarized variant: `rating > 0` → 1.0 (positive), else 0.0 (negative).

    Negative samples (from popularity-weighted negative sampling) carry
    `rating == 0`; real interactions carry `rating > 0`. The ranker predicts
    P(interact) = 1 for positives, 0 for negatives.
    """

    def __init__(
        self,
        df,
        user_col: str,
        item_col: str,
        rating_col: str,
        timestamp_col: str,
        item_feature=None,
    ):
        self.df = df.assign(**{rating_col: df[rating_col].gt(0).astype(np.float32)})
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.timestamp_col = timestamp_col
        self.item_feature = item_feature