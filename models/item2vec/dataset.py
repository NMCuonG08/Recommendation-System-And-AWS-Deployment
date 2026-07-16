"""SkipGram IterableDataset over item-id sequences (JSONL).

Ported from the reference `src/model_item2vec/dataset.py`. Pair generation +
negative sampling logic is unchanged; the Ray/DDP sharding helpers
(`get_process_info`, `torch.distributed`) are retained so the dataset shards
correctly across Ray Train workers. On a single worker (`ddp=False`) it
behaves as a plain iterable dataset.
"""

import json
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.distributed import get_rank, get_world_size
from torch.utils.data import IterableDataset, get_worker_info
from tqdm.auto import tqdm


class SkipGramDataset(IterableDataset):
    """Dataset for training a SkipGram model using sequences of item indices."""

    def __init__(
        self,
        sequences_fp: str,
        interacted: defaultdict = None,
        item_freq: defaultdict = None,
        window_size: int = 2,
        negative_samples: int = 5,
        id_to_idx: dict = None,
        ddp: bool = False,
    ):
        """Initialize the SkipGram dataset.

        Args:
            sequences_fp: Path to sequences of item indices in JSONL format.
            interacted: dict[idx -> set[idx]] of items co-occurring in the same
                sequence; excluded from negative sampling. Reuse the training
                set's for validation so val negatives match train's distribution.
            item_freq: dict[idx -> int] of item frequencies for negative sampling.
                Reuse the training set's for validation.
            window_size: Context window size for positive pair generation.
            negative_samples: Number of negative samples per positive pair.
            id_to_idx: Mapping from item IDs (str) to indices (int). If None, the
                dataset builds its own mapping from the sequences it reads.
            ddp: Whether Ray Train / DDP is active. When true, `__iter__` shards
                sequences across workers + ranks so each worker sees a disjoint
                slice. Defaults to False (single-process iterable dataset).

        Note:
            Passing `id_to_idx=None` (e.g. for the overfit sanity batch) makes the
            dataset self-contained; passing the full IDMapper mapping keeps the
            embedding index space aligned across train/val/full-idm.
        """
        if not sequences_fp.endswith(".jsonl"):
            raise ValueError("sequences_fp must be a .jsonl file")
        self.sequences_fp = sequences_fp
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.ddp = ddp

        self.interacted = deepcopy(interacted) if interacted is not None else defaultdict(set)
        self.item_freq = deepcopy(item_freq) if item_freq is not None else defaultdict(int)

        # id <-> index mappings (built or reused).
        self.id_to_idx = dict() if id_to_idx is None else dict(id_to_idx)
        self.idx_to_id = {v: k for k, v in self.id_to_idx.items()} if id_to_idx is not None else {}

        self.num_targets = 0  # total items across all sequences
        self._build_from_sequences()

    def _build_from_sequences(self) -> None:
        """Read sequences, extend id<->idx mappings, populate interacted/item_freq."""
        logger.info(f"Processing sequences: {self.sequences_fp}")
        self.sequences = []
        with open(self.sequences_fp, "r") as f:
            for line in tqdm(f, desc="Building interactions"):
                seq = json.loads(line)
                self.sequences.append(seq)

                for item in seq:
                    idx = self.id_to_idx.get(item)
                    if idx is None:
                        idx = len(self.id_to_idx)
                        self.id_to_idx[item] = idx
                        self.idx_to_id[idx] = item
                    self.num_targets += 1

                seq_idx_set = {self.id_to_idx[id_] for id_ in seq}
                for idx in seq_idx_set:
                    self.interacted[idx].update(seq_idx_set)
                    self.item_freq[idx] += 1

        self.num_sequences = len(self.sequences)
        self.vocab_size = len(self.id_to_idx)

        # Frequency-based negative sampling distribution (smoothed by ^0.75).
        items, frequencies = zip(*self.item_freq.items())
        self.item_freq_array = np.zeros(self.vocab_size)
        self.item_freq_array[np.array(items)] = frequencies
        self.items = np.arange(self.vocab_size)
        sampling_probs = self.item_freq_array**0.75
        self.sampling_probs = sampling_probs / sampling_probs.sum()

    def get_process_info(self) -> tuple[int, int]:
        """Retrieve process info for sharding the dataset across workers.

        Returns:
            (num_replicas, rank): When `ddp` is False, returns (1, 0) so every
            sequence is yielded. When DDP/Ray Train is active, combines the
            DataLoader worker id with the torch.distributed process rank into a
            global rank over `num_replicas = num_workers * world_size` slices.
        """
        if not self.ddp:
            return 1, 0

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        world_size = get_world_size()
        process_rank = get_rank()

        num_replicas = num_workers * world_size
        rank = process_rank * num_workers + worker_id
        return num_replicas, rank

    def __iter__(self):
        """Iterate over the dataset, yielding (target, context, label) pairs.

        Shards sequences across the current worker/rank when DDP is active so
        each Ray Train worker trains on a disjoint slice of the data.
        """
        num_replicas, rank = self.get_process_info()
        idx = 0
        for seq in self.sequences:
            for i in range(len(seq)):
                if idx % num_replicas != rank:
                    idx += 1
                    continue
                yield self._get_item(seq, i)
                idx += 1

    def _get_item(self, sequence: list, i: int) -> dict:
        """Generate positive + negative pairs for the item at position `i`."""
        sequence = [self.id_to_idx[item] for item in sequence]
        target_item = sequence[i]

        positive_pairs = []
        labels = []

        # Positive pairs within the window.
        start = max(i - self.window_size, 0)
        end = min(i + self.window_size + 1, len(sequence))
        for j in range(start, end):
            if i != j:
                positive_pairs.append((target_item, sequence[j]))
                labels.append(1)

        # Negative samples per positive pair.
        negative_pairs = []
        for tgt, _ in positive_pairs:
            neg_probs = deepcopy(self.sampling_probs)
            neg_probs[list(self.interacted[tgt])] = 0
            total = neg_probs.sum()
            if total == 0:
                # Everything is masked (tiny vocab / very popular target) —
                # fall back to a uniform distribution over the whole vocab.
                neg_probs = np.ones(len(neg_probs))
                n_avail = len(neg_probs)
            else:
                n_avail = int((neg_probs > 0).sum())
            neg_probs /= neg_probs.sum()

            # `replace=False` needs at least `negative_samples` candidates; when a
            # popular target has co-occurred with almost the whole vocab, allow
            # replacement so sampling never crashes.
            replace = n_avail < self.negative_samples
            negative_items = np.random.choice(
                self.items, size=self.negative_samples, p=neg_probs, replace=replace
            )
            for neg_item in negative_items:
                negative_pairs.append((tgt, neg_item))
                labels.append(0)

        pairs = positive_pairs + negative_pairs
        return {
            "target_items": torch.tensor([p[0] for p in pairs], dtype=torch.long),
            "context_items": torch.tensor([p[1] for p in pairs], dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }

    def collate_fn(self, batch: list) -> dict:
        """Collate a list of per-item dicts into one dict of concatenated tensors."""
        return {
            "target_items": torch.cat([r["target_items"] for r in batch], dim=0),
            "context_items": torch.cat([r["context_items"] for r in batch], dim=0),
            "labels": torch.cat([r["labels"] for r in batch], dim=0),
        }

    @classmethod
    def get_default_loss_fn(cls) -> nn.Module:
        """Default loss function: binary cross entropy."""
        return nn.BCELoss()