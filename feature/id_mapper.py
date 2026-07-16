"""Bidirectional id <-> indice mapper for users and items.

Ported from the reference project's `src/id_mapper.py`, adapted so the
MovieLens integer ids map to contiguous 0-based indices for embedding
tables (item2vec / GRU ranking model). Unknown ids map to a trailing
"unknown" index.
"""

from __future__ import annotations

import json
from typing import Iterable


class IDMapper:
    """Map raw user/item ids to dense contiguous indices and back."""

    def __init__(self) -> None:
        self.user_to_index: dict = {}
        self.index_to_user: list = []
        self.item_to_index: dict = {}
        self.index_to_item: list = []
        self.unknown_user_index: int = -1
        self.unknown_item_index: int = -1

    def fit(self, user_ids: Iterable, item_ids: Iterable) -> "IDMapper":
        """Build mappings from sorted-then-enumerated id sequences.

        Sorting first keeps the mapping stable across reruns.
        """
        user_ids = list(user_ids)
        item_ids = list(item_ids)
        self.user_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
        self.index_to_user = user_ids
        self.item_to_index = {iid: idx for idx, iid in enumerate(item_ids)}
        self.index_to_item = item_ids
        self.unknown_user_index = len(self.user_to_index)
        self.unknown_item_index = len(self.item_to_index)
        return self

    def get_user_index(self, user_id) -> int:
        return self.user_to_index.get(user_id, self.unknown_user_index)

    def get_item_index(self, item_id) -> int:
        return self.item_to_index.get(item_id, self.unknown_item_index)

    def get_user_id(self, index: int):
        if index < len(self.index_to_user):
            return self.index_to_user[index]
        return "unknown_user"

    def get_item_id(self, index: int):
        if index < len(self.index_to_item):
            return self.index_to_item[index]
        return "unknown_item"

    def save(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "user_to_index": {int(k): int(v) for k, v in self.user_to_index.items()},
                    "index_to_user": [int(x) for x in self.index_to_user],
                    "item_to_index": {int(k): int(v) for k, v in self.item_to_index.items()},
                    "index_to_item": [int(x) for x in self.index_to_item],
                },
                f,
            )

    def load(self, filepath: str) -> "IDMapper":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON dict keys come back as str; rebuild with int keys so numpy int64
        # lookups from a pandas frame still match (int and np.int64 hash equal).
        self.user_to_index = {int(k): int(v) for k, v in data["user_to_index"].items()}
        self.index_to_user = [int(x) for x in data["index_to_user"]]
        self.item_to_index = {int(k): int(v) for k, v in data["item_to_index"].items()}
        self.index_to_item = [int(x) for x in data["index_to_item"]]
        self.unknown_user_index = len(self.user_to_index)
        self.unknown_item_index = len(self.item_to_index)
        return self


def map_indice(df, idm: IDMapper, user_col: str = "userId", item_col: str = "movieId"):
    """Attach `user_indice` and `item_indice` columns to a frame."""
    return df.assign(
        user_indice=lambda d: d[user_col].apply(lambda u: idm.get_user_index(u)),
        item_indice=lambda d: d[item_col].apply(lambda i: idm.get_item_index(i)),
    )