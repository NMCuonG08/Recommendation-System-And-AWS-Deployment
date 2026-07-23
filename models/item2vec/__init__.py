"""Item2Vec (SkipGram) model family — local single-machine training."""

from models.item2vec.dataset import SkipGramDataset
from models.item2vec.model import SkipGram

# The trainer pulls in mlflow / ray / evidently (heavy, training-only deps).
# Import it lazily so the model + dataset classes stay usable in lean
# environments — e.g. loading a checkpoint's embedding on a host without the
# full MLOps stack. `train.py` imports LitSkipGram directly from the trainer
# module and runs in an env where the heavy deps are installed.
try:
    from models.item2vec.trainer import LitSkipGram
except ImportError:
    LitSkipGram = None  # type: ignore[assignment]

__all__ = ["SkipGram", "SkipGramDataset", "LitSkipGram"]