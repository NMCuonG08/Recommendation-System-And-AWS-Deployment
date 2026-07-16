"""Item2Vec evaluation + plotting — derive charts from trained checkpoints.

Run (from repo root):
    PYTHONPATH=. uv run python -m models.item2vec.evaluate
    PYTHONPATH=. uv run python -m models.item2vec.evaluate --config configs/item2vec.yaml

Reads the two Lightning checkpoints produced by `models.item2vec.train`:
  - `models/output/item2vec/final_model/best-checkpoint.ckpt`     (final model)
  - `models/output/item2vec/checkpoints/overfit/best-checkpoint.ckpt` (overfit sanity)

Builds a reproducible validation pair set (seeded negative sampling, same
`interacted` / `item_freq` reuse as training), computes classification metrics
(BCE val_loss, ROC-AUC, PR-AUC, precision/recall @ 0.5), extracts item
embeddings for t-SNE / PCA / similarity-heatmap / nearest-neighbor analysis,
and writes all figures + a `metrics.json` under
`models/output/item2vec/reports/`.

If `models/output/item2vec/reports/hp_search_results.json` exists (written by
the fixed `train.py` HP search loop), its per-trial table is also plotted as a
trial val_loss bar chart.

Note: Lightning was trained with `logger=False`, so per-epoch loss curves were
never persisted. This script reports end-of-training metrics from the
checkpoints directly rather than a training history.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from feature.id_mapper import IDMapper
from models.item2vec.dataset import SkipGramDataset
from models.item2vec.model import SkipGram

# Project root (repo root) so paths resolve regardless of cwd.
ROOT = Path(__file__).resolve().parents[2]

SEED = 42
DEFAULT_CONFIG = ROOT / "configs" / "item2vec.yaml"
DEFAULT_OUT = ROOT / "models" / "output" / "item2vec"
DATA_DIR = ROOT / "feature" / "output" / "engineer"


def _resolve_ckpt(dir_path: Path, base_name: str = "best-checkpoint") -> Path:
    """Pick the newest `best-checkpoint*.ckpt` in a dir.

    Lightning's ModelCheckpoint appends `-v{N}` to avoid clobbering an existing
    file with the same name, so a re-run final-training checkpoint can land as
    `best-checkpoint-v2.ckpt` while a stale `best-checkpoint.ckpt` remains.
    Resolve by mtime so we always load the latest produced checkpoint.
    """
    candidates = sorted(
        dir_path.glob(f"{base_name}*.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        # Fall back to the canonical name so the error message is intuitive.
        return dir_path / f"{base_name}.ckpt"
    return candidates[-1]

FIG_DPI = 120


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def set_seed(seed: int = SEED) -> None:
    """Seed Python / NumPy / Torch for reproducible negative sampling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(yaml_path: Path) -> dict[str, Any]:
    """Load a YAML config (env placeholders not resolved — only paths used)."""
    import yaml

    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


def read_best_val_loss(ckpt_path: Path) -> float | None:
    """Read the best monitored val_loss from a Lightning checkpoint's callback state."""
    if not ckpt_path.exists():
        return None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for cb in ck.get("callbacks", {}).values():
        if isinstance(cb, dict) and "best_model_score" in cb:
            score = cb["best_model_score"]
            return float(score.item()) if hasattr(score, "item") else float(score)
    return None


def load_model_from_checkpoint(
    ckpt_path: Path,
) -> tuple[SkipGram, dict[str, Any], np.ndarray]:
    """Load a SkipGram model + its hparams + the embedding weight from a Lightning ckpt.

    Args:
        ckpt_path: Path to a Lightning `best-checkpoint.ckpt`.

    Returns:
        (model, meta, emb_weight) where meta holds epoch/global_step/lr/l2/dim
        and emb_weight is the (vocab+1, D) embedding tensor (last row = padding).
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck.get("hyper_parameters", {})
    state = ck.get("state_dict", {})
    emb_weight = state["skipgram_model.embeddings.weight"].clone()
    num_items, embedding_dim = emb_weight.shape[0] - 1, emb_weight.shape[1]

    model = SkipGram(num_items, embedding_dim)
    model.load_state_dict({"embeddings.weight": emb_weight}, strict=False)
    model.eval()

    meta = {
        "epoch": int(ck.get("epoch", -1)),
        "global_step": int(ck.get("global_step", -1)),
        "learning_rate": float(hp.get("learning_rate", float("nan"))),
        "l2_reg": float(hp.get("l2_reg", float("nan"))),
        "embedding_dim": int(embedding_dim),
        "num_items": int(num_items),
        "best_val_loss": read_best_val_loss(ckpt_path),
    }
    return model, meta, emb_weight.numpy()


# ---------------------------------------------------------------------------
# Validation pair collection (reproducible)
# ---------------------------------------------------------------------------


def collect_val_pairs(
    cfg: dict[str, Any], idm: IDMapper
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a reproducible validation (target, context, label) pair set.

    Reuses the train dataset's `interacted` / `item_freq` (so val negatives are
    drawn from the same distribution as during training), then seeds NumPy
    before iterating so negative sampling is deterministic across runs.

    Args:
        cfg: Parsed item2vec config (uses `training.window_size`,
            `training.num_negative_samples`, `data.*` filenames).
        idm: Loaded IDMapper (item -> index).

    Returns:
        (targets, contexts, labels) as int64 / float32 NumPy arrays.
    """
    t = cfg["training"]
    d = cfg["data"]

    # Build the train dataset only to harvest interacted + item_freq + idm mapping.
    train_ds = SkipGramDataset(
        str(DATA_DIR / d["sequences_file"]),
        window_size=t["window_size"],
        negative_samples=t["num_negative_samples"],
        id_to_idx=idm.item_to_index,
        ddp=False,
    )

    set_seed(SEED)
    val_ds = SkipGramDataset(
        str(DATA_DIR / d["val_sequences_file"]),
        interacted=train_ds.interacted,
        item_freq=train_ds.item_freq,
        window_size=t["window_size"],
        negative_samples=t["num_negative_samples"],
        id_to_idx=idm.item_to_index,
        ddp=False,
    )
    loader = DataLoader(
        val_ds,
        batch_size=t["batch_size"],
        shuffle=False,
        collate_fn=val_ds.collate_fn,
        num_workers=0,
    )
    targets, contexts, labels = [], [], []
    for batch in loader:
        targets.append(batch["target_items"].numpy())
        contexts.append(batch["context_items"].numpy())
        labels.append(batch["labels"].numpy())
    return (
        np.concatenate(targets).astype(np.int64),
        np.concatenate(contexts).astype(np.int64),
        np.concatenate(labels).astype(np.float32),
    )


def run_preds(
    model: SkipGram, targets: np.ndarray, contexts: np.ndarray
) -> np.ndarray:
    """Run the model on a pair set and return predictions as float32."""
    with torch.no_grad():
        return (
            model(
                torch.tensor(targets, dtype=torch.long),
                torch.tensor(contexts, dtype=torch.long),
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )


def compute_metrics(
    model: SkipGram, targets: np.ndarray, contexts: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Compute BCE val_loss + ROC-AUC + PR-AUC + precision/recall @ 0.5."""
    preds = run_preds(model, targets, contexts)
    loss = nn.BCELoss()(torch.tensor(preds), torch.tensor(labels)).item()
    pred_bin = (preds >= 0.5).astype(np.int64)
    prec = float(precision_score(labels, pred_bin, zero_division=0))
    rec = float(recall_score(labels, pred_bin, zero_division=0))
    return {
        "val_loss": float(loss),
        "num_pairs": int(labels.shape[0]),
        "num_pos": int((labels == 1).sum()),
        "num_neg": int((labels == 0).sum()),
        "mean_pred": float(preds.mean()),
        "roc_auc": float(roc_auc_score(labels, preds)),
        "avg_precision": float(average_precision_score(labels, preds)),
        "precision_at_0.5": prec,
        "recall_at_0.5": rec,
        "f1_at_0.5": float(2 * prec * rec / max(prec + rec, 1e-12)),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_roc(labels: np.ndarray, preds: np.ndarray, out_dir: Path) -> Path:
    fpr, tpr, _ = roc_curve(labels, preds)
    auc = roc_auc_score(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Validation ROC — item2vec SkipGram")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "roc_curve.png")


def plot_pr(labels: np.ndarray, preds: np.ndarray, out_dir: Path) -> Path:
    prec, rec, _ = precision_recall_curve(labels, preds)
    ap = average_precision_score(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, label=f"PR (AP = {ap:.4f})", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Validation Precision-Recall — item2vec SkipGram")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "pr_curve.png")


def plot_score_distribution(
    labels: np.ndarray, preds: np.ndarray, out_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(preds[labels == 1], bins=50, alpha=0.6, label="positive (label=1)", color="tab:green")
    ax.hist(preds[labels == 0], bins=50, alpha=0.6, label="negative (label=0)", color="tab:red")
    ax.set_xlabel("Predicted similarity (sigmoid of dot product)")
    ax.set_ylabel("Count")
    ax.set_title("Validation score distribution by label")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out_dir, "score_distribution.png")


def plot_loss_comparison(
    final_loss: float, overfit_loss: float | None, out_dir: Path
) -> Path:
    overfit_val = overfit_loss if overfit_loss is not None else float("nan")
    fig, ax = plt.subplots(figsize=(6, 4))
    names = ["Overfit sanity\n(1 batch)", "Final model\n(val set)"]
    values = [overfit_val, final_loss]
    bars = ax.bar(names, values, color=["tab:orange", "tab:blue"])
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom"
        )
    ax.set_ylabel("BCE val_loss")
    ax.set_title("Overfit sanity vs final model val_loss")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "loss_comparison.png")


def plot_hp_search(hp_results_fp: Path, out_dir: Path) -> Path | None:
    """Bar chart of per-trial val_loss from the HP search results JSON."""
    if not hp_results_fp.exists():
        return None
    with open(hp_results_fp) as f:
        data = json.load(f)
    trials = data.get("trials", [])
    trials = [t for t in trials if t.get("val_loss") is not None]
    if not trials:
        return None
    labels_x = [f"t{t['trial']}" for t in trials]
    vals = [t["val_loss"] for t in trials]
    best_idx = int(np.argmin(vals))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:red" if i == best_idx else "tab:blue" for i in range(len(vals))]
    ax.bar(labels_x, vals, color=colors)
    ax.set_xlabel("Trial")
    ax.set_ylabel("best val_loss (BCE)")
    ax.set_title(
        f"HP search — {len(vals)} trials (best=t{trials[best_idx]['trial']}, "
        f"val_loss={vals[best_idx]:.4f})"
    )
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, out_dir, "hp_search_val_loss.png")


def plot_embedding_tsne(emb: np.ndarray, freq: np.ndarray, out_dir: Path) -> Path:
    tsne = TSNE(
        n_components=2, perplexity=30, random_state=SEED, init="pca", learning_rate="auto"
    )
    proj = tsne.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.log1p(freq), cmap="viridis", s=8, alpha=0.7)
    plt.colorbar(sc, label="log(1 + item frequency)")
    ax.set_title(f"Item embedding t-SNE (V={emb.shape[0]}, D={emb.shape[1]})")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    return _save(fig, out_dir, "embedding_tsne.png")


def plot_embedding_pca(emb: np.ndarray, freq: np.ndarray, out_dir: Path) -> Path:
    pca = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.log1p(freq), cmap="viridis", s=8, alpha=0.7)
    plt.colorbar(sc, label="log(1 + item frequency)")
    ax.set_title(
        f"Item embedding PCA (D={emb.shape[1]} -> 2; var={pca.explained_variance_ratio_.sum():.3f})"
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})")
    return _save(fig, out_dir, "embedding_pca.png")


def plot_similarity_heatmap(emb: np.ndarray, top_idx: np.ndarray, out_dir: Path) -> Path:
    """Cosine-similarity heatmap among the top-N most frequent items."""
    sub = emb[top_idx]
    sub_norm = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-12)
    sim = sub_norm @ sub_norm.T
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"Item-item cosine similarity (top {len(top_idx)} frequent items)")
    ax.set_xlabel("item rank (by frequency)")
    ax.set_ylabel("item rank (by frequency)")
    plt.colorbar(im, label="cosine similarity")
    return _save(fig, out_dir, "similarity_heatmap.png")


def nearest_neighbors_table(
    emb: np.ndarray, top_idx: np.ndarray, k: int = 10
) -> list[dict[str, Any]]:
    """For each of the top items, return its top-k nearest items by cosine sim."""
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    rows = []
    for item in top_idx[:5]:
        sims = norm @ norm[item]
        order = np.argsort(-sims)
        nn_idx = [int(j) for j in order if j != item][:k]
        rows.append(
            {
                "item_idx": int(item),
                "neighbors": [
                    {"idx": int(j), "sim": float(sims[j])} for j in nn_idx
                ],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate item2vec checkpoints + plot.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    out_dir = Path(args.out)
    fig_dir = out_dir / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    hp_results_fp = out_dir / "reports" / "hp_search_results.json"

    final_ckpt = _resolve_ckpt(out_dir / "final_model")
    overfit_ckpt = _resolve_ckpt(out_dir / "checkpoints" / "overfit")

    print(f"Loading final checkpoint: {final_ckpt}")
    final_model, final_meta, emb_weight = load_model_from_checkpoint(final_ckpt)
    print(f"  final: {final_meta}")

    print(f"Loading overfit checkpoint: {overfit_ckpt}")
    overfit_loss = read_best_val_loss(overfit_ckpt)
    print(f"  overfit best_val_loss: {overfit_loss}")

    print("Loading IDMapper + building validation pairs...")
    idm = IDMapper().load(str(DATA_DIR / cfg["data"]["idm_file"]))
    targets, contexts, labels = collect_val_pairs(cfg, idm)
    print(
        f"  val pairs: {len(labels)} "
        f"(pos={int((labels == 1).sum())}, neg={int((labels == 0).sum())})"
    )

    print("Computing final-model metrics...")
    metrics = compute_metrics(final_model, targets, contexts, labels)
    print(f"  {metrics}")

    print("Plotting curves...")
    preds = run_preds(final_model, targets, contexts)
    plot_roc(labels, preds, fig_dir)
    plot_pr(labels, preds, fig_dir)
    plot_score_distribution(labels, preds, fig_dir)
    plot_loss_comparison(metrics["val_loss"], overfit_loss, fig_dir)
    plot_hp_search(hp_results_fp, fig_dir)

    print("Extracting embeddings + plotting embedding figures...")
    emb = emb_weight[:-1]  # drop padding row (last idx)
    # Frequency per item index from the train dataset.
    train_ds = SkipGramDataset(
        str(DATA_DIR / cfg["data"]["sequences_file"]),
        window_size=cfg["training"]["window_size"],
        negative_samples=cfg["training"]["num_negative_samples"],
        id_to_idx=idm.item_to_index,
        ddp=False,
    )
    freq = np.array(
        [train_ds.item_freq.get(i, 0) for i in range(emb.shape[0])], dtype=np.float32
    )
    plot_embedding_tsne(emb, freq, fig_dir)
    plot_embedding_pca(emb, freq, fig_dir)
    top_idx = np.argsort(-freq)[:25]
    plot_similarity_heatmap(emb, top_idx, fig_dir)
    nn_rows = nearest_neighbors_table(emb, top_idx, k=10)

    report_payload = {
        "final_model": final_meta,
        "overfit_best_val_loss": overfit_loss,
        "metrics": metrics,
        "vocab_size": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "num_train_sequences": int(train_ds.num_sequences),
        "num_val_sequences": int(len(labels)),
        "nearest_neighbors": nn_rows,
        "figures": sorted(str(p.relative_to(ROOT)) for p in fig_dir.glob("*.png")),
    }
    metrics_fp = out_dir / "reports" / "metrics.json"
    with open(metrics_fp, "w") as f:
        json.dump(report_payload, f, indent=2)
    print(f"Saved metrics: {metrics_fp}")
    print(f"Figures dir: {fig_dir}")


if __name__ == "__main__":
    main()