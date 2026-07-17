"""GRU sequence ranker model.

Ported verbatim from the reference `src/model_ranking_sequence/model.py`.
A `Ranker` predicts a user-item interaction score from four feature sources
concatenated then fed through an MLP:

  1. GRU hidden state over the user's item-interaction sequence (item IDs +
     timestamp buckets), using **frozen Item2Vec item embeddings** as the
     per-step item embedding.
  2. Target item embedding (Item2Vec, frozen).
  3. Learned user embedding.
  4. Item-feature tower (numeric item features from Feast/the feature pipeline).

Output is a sigmoid score in [0, 1] = P(user interacts with target item).
`recommend()` scores all candidate items for a batch of users and returns
top-k (used for offline ranking evaluation).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Ranker(nn.Module):
    """GRU ranker predicting user-item interaction probability from sequences."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        item_sequence_ts_bucket_size: int,
        bucket_embedding_dim: int,
        item_feature_size: int,
        item_embedding: nn.Embedding | None = None,
        dropout: float = 0.2,
    ):
        """Initialize the Ranker.

        Args:
            num_users: Number of unique users (embedding rows).
            num_items: Number of unique items (incl. padding row).
            embedding_dim: Item/user embedding dimensionality (must match the
                frozen Item2Vec embedding passed as `item_embedding`).
            item_sequence_ts_bucket_size: Number of timestamp buckets.
            bucket_embedding_dim: Timestamp-bucket embedding dimensionality.
            item_feature_size: Input width of the item-feature tower.
            item_embedding: Pretrained (frozen) Item2Vec item embedding. If
                None, a fresh trainable embedding is created.
            dropout: Dropout probability in the MLP / item-feature tower.
        """
        super().__init__()

        self.num_items = num_items
        self.num_users = num_users

        self.item_embedding = item_embedding
        if item_embedding is None:
            self.item_embedding = nn.Embedding(
                num_items + 1, embedding_dim, padding_idx=num_items
            )

        self.user_embedding = nn.Embedding(num_users, embedding_dim)

        self.item_sequence_ts_bucket_embedding = nn.Embedding(
            item_sequence_ts_bucket_size + 1,
            bucket_embedding_dim,
            padding_idx=item_sequence_ts_bucket_size,
        )

        self.gru = nn.GRU(
            input_size=embedding_dim + bucket_embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.item_feature_tower = nn.Sequential(
            nn.Linear(item_feature_size, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            self.relu,
            self.dropout,
        )

        # 4 sources of features concatenated: GRU seq, target item, user, item feats.
        input_dim = embedding_dim * 4
        self.fc_rating = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            self.relu,
            self.dropout,
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, user_ids, input_seq, input_seq_ts_bucket, item_features, target_item):
        """Predict interaction scores for a batch of (user, target_item) pairs.

        Args:
            user_ids: [batch] user indices.
            input_seq: [batch, seq_len] item indices the user interacted with.
            input_seq_ts_bucket: [batch, seq_len] timestamp buckets per step.
            item_features: [batch, item_feature_size] target-item features.
            target_item: [batch] target item indices to score.

        Returns:
            [batch, 1] sigmoid scores.
        """
        padding_idx_tensor = torch.tensor(self.item_embedding.padding_idx)
        input_seq = torch.where(input_seq == -1, padding_idx_tensor, input_seq)
        target_item = torch.where(target_item == -1, padding_idx_tensor, target_item)

        embedded_id_seq = self.item_embedding(input_seq)

        bucket_padding_idx_tensor = torch.tensor(
            self.item_sequence_ts_bucket_embedding.padding_idx
        )
        input_seq_ts_bucket = torch.where(
            input_seq_ts_bucket == -1, bucket_padding_idx_tensor, input_seq_ts_bucket
        )
        embedded_ts_bucket_seq = self.item_sequence_ts_bucket_embedding(input_seq_ts_bucket)

        embedded_seq = torch.cat((embedded_id_seq, embedded_ts_bucket_seq), dim=-1)

        item_features_tower_output = self.item_feature_tower(item_features)

        _, hidden_state = self.gru(embedded_seq)
        gru_output = hidden_state.squeeze(0)  # [batch, embedding_dim]

        embedded_target = self.item_embedding(target_item)
        user_embeddings = self.user_embedding(user_ids)

        combined_embedding = torch.cat(
            (gru_output, embedded_target, user_embeddings, item_features_tower_output),
            dim=1,
        )
        return self.fc_rating(combined_embedding)

    def predict(self, user, item_sequence, input_seq_ts_bucket, item_features, target_item):
        """Predict scores (alias of forward, for eval clarity)."""
        return self.forward(
            user, item_sequence, input_seq_ts_bucket, item_features, target_item
        )

    def recommend(
        self,
        users: torch.Tensor,
        item_sequences: torch.Tensor,
        item_ts_bucket_sequences: torch.Tensor,
        item_features: torch.Tensor,
        item_indices: torch.Tensor,
        k: int,
        batch_size: int = 1024,
    ):
        """Score all candidate items for each user and return top-k item indices.

        Args:
            users: [N] user indices.
            item_sequences: [N, seq_len] per-user interaction sequences.
            item_ts_bucket_sequences: [N, seq_len] timestamp buckets.
            item_features: [M, feat_dim] features for all candidate items.
            item_indices: [M] candidate item indices.
            k: Top-k to return per user.
            batch_size: Item chunk size for the scoring loop.

        Returns:
            [N, K] top-k item indices per user.
        """
        self.eval()
        num_users = users.size(0)
        num_items = item_indices.size(0)

        item_padding_idx = self.item_embedding.padding_idx
        seq_bucket_padding_idx = self.item_sequence_ts_bucket_embedding.padding_idx

        item_sequences = item_sequences.clone()
        item_sequences[item_sequences < 0] = item_padding_idx
        item_ts_bucket_sequences = item_ts_bucket_sequences.clone()
        item_ts_bucket_sequences[item_ts_bucket_sequences < 0] = seq_bucket_padding_idx
        item_indices = item_indices.clone()
        item_indices[item_indices < 0] = item_padding_idx

        # Clamp out-of-vocab indices to padding.
        item_sequences[item_sequences >= self.num_items] = item_padding_idx
        item_indices[item_indices >= self.num_items] = item_padding_idx

        with torch.no_grad():
            user_emb = self.user_embedding(users)  # [N, d]
            item_seq_emb = self.item_embedding(item_sequences)  # [N, seq_len, d]
            ts_bucket_emb = self.item_sequence_ts_bucket_embedding(
                item_ts_bucket_sequences
            )  # [N, seq_len, d]
            seq_input = torch.cat([item_seq_emb, ts_bucket_emb], dim=-1)
            _, seq_hidden = self.gru(seq_input)
            seq_hidden = seq_hidden.squeeze(0)  # [N, d]

            user_seq_emb = torch.cat([seq_hidden, user_emb], dim=1)  # [N, 2d]

            all_scores = None
            for i in range(0, num_items, batch_size):
                idx = slice(i, min(i + batch_size, num_items))
                items = item_indices[idx]  # [b]
                items_emb = self.item_embedding(items)  # [b, d]
                items_feat = item_features[idx]  # [b, feat_dim]
                item_feat_proj = self.item_feature_tower(items_feat)  # [b, d]

                items_emb_exp = items_emb.unsqueeze(0).expand(num_users, -1, -1)
                item_feat_proj_exp = item_feat_proj.unsqueeze(0).expand(num_users, -1, -1)
                user_seq_emb_exp = user_seq_emb.unsqueeze(1).expand(
                    -1, items_emb.shape[0], -1
                )

                full_input = torch.cat(
                    [user_seq_emb_exp, items_emb_exp, item_feat_proj_exp], dim=2
                )  # [N, b, 4d]
                flat_input = full_input.view(-1, full_input.shape[2])  # [N*b, 4d]
                out = self.fc_rating(flat_input)  # [N*b, 1]
                score = out.view(num_users, items_emb.shape[0])  # [N, b]

                all_scores = score if all_scores is None else torch.cat(
                    [all_scores, score], dim=1
                )

            _, topk_indices = torch.topk(all_scores, k, dim=1)
            return item_indices[topk_indices]  # [N, K]