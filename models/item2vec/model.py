"""SkipGram model for learning item embeddings from interaction sequences.

Ported from the reference `src/model_item2vec/model.py`. Unchanged behavior —
only the intra-package import was switched to a relative import.
"""

from typing import List

import torch
import torch.nn as nn

from models.item2vec.dataset import SkipGramDataset


class SkipGram(nn.Module):
    """SkipGram model for learning item embeddings via positive + negative sampling."""

    def __init__(self, num_items: int, embedding_dim: int):
        """Initialize the SkipGram model.

        Args:
            num_items: Total number of unique items in the vocabulary.
            embedding_dim: Dimensionality of the embedding vectors.
        """
        super().__init__()
        # +1 slot is a padding/unknown index so raw OOB ids do not crash Embedding.
        self.embeddings = nn.Embedding(
            num_items + 1, embedding_dim, padding_idx=num_items
        )
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(
        self, target_items: torch.Tensor, context_items: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity scores between target and context items.

        Args:
            target_items: Tensor of target item indices, shape (batch_size,).
            context_items: Tensor of context item indices, shape (batch_size,).

        Returns:
            Predicted probabilities (sigmoid of dot product), shape (batch_size,).
        """
        target_embeds = self.embeddings(target_items)   # (B, D)
        context_embeds = self.embeddings(context_items)  # (B, D)
        similarity_scores = torch.sum(target_embeds * context_embeds, dim=-1)
        return torch.sigmoid(similarity_scores)

    def get_item_embedding(self, item_idx: int) -> torch.Tensor:
        """Retrieve the embedding vector for a specific item index."""
        return self.embeddings(torch.tensor(item_idx, dtype=torch.long))

    def predict_train_batch(
        self, batch_input: dict, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Predict scores for a batch of training data.

        Args:
            batch_input: Dict with 'target_items' and 'context_items' tensors.
            device: Device to run on. Defaults to CPU.
        """
        target_items = batch_input["target_items"].to(device)
        context_items = batch_input["context_items"].to(device)
        return self.forward(target_items, context_items)

    @classmethod
    def get_expected_dataset_type(cls) -> List[type]:
        """Expected dataset type(s) for training this model."""
        return [SkipGramDataset]