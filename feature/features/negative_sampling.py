"""Popularity-weighted negative sampling for implicit-feedback training.

Ported verbatim from the reference project's
`src/feature_engineer/negative_sampling.py`. For each positive user-item
interaction, sample `neg_to_pos_ratio` items the user did NOT interact with,
weighted by item popularity. Used to build a balanced positive/negative
training set for the ranking model.
"""

from __future__ import annotations

from typing import List

import numpy as np
from tqdm.auto import tqdm


def generate_negative_samples(
    df,
    user_col="user_indice",
    item_col="item_indice",
    label_col="rating",
    timestamp_col="event_timestamp",
    neg_label=0,
    neg_to_pos_ratio=1,
    seed=None,
    features: List[str] = None,
):
    """Generate negative samples for a user-item interaction DataFrame.

    Args:
        df: DataFrame containing user-item interactions.
        user_col: Column name representing users.
        item_col: Column name representing items.
        label_col: Column name for the interaction label (e.g. rating).
        timestamp_col: Column name for the event timestamp.
        neg_label: Label assigned to negative samples (default 0).
        neg_to_pos_ratio: Negatives per positive.
        seed: Optional RNG seed for reproducibility.
        features: Extra columns to transfer from positive to negative rows.

    Returns:
        DataFrame of generated negative samples.
    """
    if features is None:
        features = []

    if seed is not None:
        np.random.seed(seed)

    # Item popularity = how often each item appears.
    item_popularity = df[item_col].value_counts()
    items = item_popularity.index.values
    all_items_set = set(items)

    # Map each user to the set of items they interacted with.
    user_item_dict = df.groupby(user_col)[item_col].apply(set).to_dict()

    popularity = item_popularity.values.astype(np.float64)
    total_popularity = popularity.sum()
    if total_popularity == 0:
        sampling_probs = np.ones(len(items)) / len(items)
    else:
        sampling_probs = popularity / total_popularity

    item_to_index = {item: idx for idx, item in enumerate(items)}

    def generate_negative_samples_for_user(row):
        user = row[user_col]
        pos_items = user_item_dict[user]

        negative_candidates = all_items_set - pos_items
        num_neg_candidates = len(negative_candidates)
        if num_neg_candidates == 0:
            return []

        num_neg = min(neg_to_pos_ratio, num_neg_candidates)
        negative_candidates_list = list(negative_candidates)

        candidate_indices = [item_to_index[item] for item in negative_candidates_list]
        candidate_probs = sampling_probs[candidate_indices]
        candidate_probs /= candidate_probs.sum()

        return np.random.choice(
            negative_candidates_list, size=num_neg, replace=False, p=candidate_probs
        )

    tqdm.pandas()
    df_negative = (
        df.copy()
        .assign(
            negative_samples=lambda d: d.progress_apply(
                generate_negative_samples_for_user, axis=1
            ),
            **{label_col: neg_label},
        )
        .explode("negative_samples")
        .drop(columns=[item_col])
        .rename(columns={"negative_samples": item_col})[
            [user_col, item_col, label_col, timestamp_col, *features]
        ]
    )

    return df_negative