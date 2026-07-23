"""Standalone GRU ranker training (no Ray/MLflow/Evidently) — local-pro path.

Produces the disk artifacts that convert2onnx_and_build_triton.py reads via
its disk-fallback path:
  models/output/ranking_sequence/final_model/best-checkpoint.ckpt  (Lightning
      checkpoint whose state_dict has `model.*` keys — Ranker weights)
  models/output/ranking_sequence/final_model/idm.json
  models/output/ranking_sequence/final_model/item_metadata_pipeline.dill

Mirrors models/ranking_sequence/train.py train_func minus Ray/MLflow: loads the
frozen-init Item2Vec item embedding from disk, builds item features via the
fitted item_metadata_pipeline, trains a GRU Ranker with weighted BCE + val
ROC-AUC, and saves the best checkpoint.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import dill
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from torch.utils.data import DataLoader
from torchmetrics import AUROC

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from feature.id_mapper import IDMapper  # noqa: E402
from models.ranking_sequence.dataset import UserItemBinaryDFDataset  # noqa: E402
from models.ranking_sequence.model import Ranker  # noqa: E402
from src.data_prep_utils import chunk_transform  # noqa: E402

ENGINEER_DIR = ROOT / "feature" / "output" / "engineer"
FINAL_DIR = ROOT / "models" / "output" / "ranking_sequence" / "final_model"
ITEM2VEC_CKPT = ROOT / "models" / "output" / "item2vec" / "final_model" / "best-checkpoint.ckpt"

UCOL, ICOL, RCOL, TSCOL = "user_indice", "item_indice", "rating", "event_timestamp"
REQUIRED_COLS = ["title", "genres", "movie_rating_cnt_90d", "movie_rating_avg_prev_rating_90d",
                 "movie_rating_cnt_30d", "movie_rating_avg_prev_rating_30d",
                 "movie_rating_cnt_7d", "movie_rating_avg_prev_rating_7d"]
HP = dict(max_epochs=5, batch_size=256, lr=1e-3, l2=1e-5, dropout=0.2,
          neg_to_pos_ratio=3, ts_bucket_size=10, bucket_emb_dim=16)


def load_item2vec_embedding() -> torch.Tensor:
    # The checkpoint pickles the full SkipGram nn.Module (stored as a
    # hyperparameter), so weights_only=True would require allowlisting every
    # submodule. The item2vec package __init__ imports the model class lazily
    # (no mlflow/ray/evidently needed), so weights_only=False is safe here on
    # this trusted local checkpoint and reconstructs SkipGram directly.
    ck = torch.load(ITEM2VEC_CKPT, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    key = "skipgram_model.embeddings.weight" if "skipgram_model.embeddings.weight" in state else "embeddings.weight"
    return state[key].detach().clone()


def build_item_features(df: pd.DataFrame, pipeline) -> np.ndarray:
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return chunk_transform(df, pipeline, chunk_size=10000).astype(np.float32)


def collate(batch):
    return {
        "user": torch.stack([x["user"] for x in batch]),
        "item_sequence": torch.stack([x["item_sequence"] for x in batch]),
        "item": torch.stack([x["item"] for x in batch]),
        "rating": torch.stack([x["rating"] for x in batch]),
        "item_sequence_ts_bucket": torch.stack([x["item_sequence_ts_bucket"] for x in batch]),
        "item_feature": torch.stack([x["item_feature"] for x in batch]),
    }


class LitRankerMini(L.LightningModule):
    def __init__(self, model: Ranker, learning_rate: float, l2_reg: float, neg_to_pos_ratio: int):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.val_roc_auc_metric = AUROC(task="binary")

    def training_step(self, batch, batch_idx):
        labels = batch["rating"].float()
        preds = self.model.forward(
            batch["user"], batch["item_sequence"], batch["item_sequence_ts_bucket"],
            batch["item_feature"], batch["item"],
        ).view(labels.shape)
        weights = torch.where(labels == 1, float(self.neg_to_pos_ratio), 1.0)
        loss = nn.BCELoss(weights)(preds, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["rating"]
        preds = self.model.forward(
            batch["user"], batch["item_sequence"], batch["item_sequence_ts_bucket"],
            batch["item_feature"], batch["item"],
        ).view(labels.shape)
        weights = torch.where(labels == 1, float(self.neg_to_pos_ratio), 1.0)
        loss = nn.BCELoss(weights)(preds, labels)
        self.val_roc_auc_metric.update(preds, labels.int())
        self.log("val_roc_auc", self.val_roc_auc_metric.compute(), on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_roc_auc", self.val_roc_auc_metric.compute())
        self.val_roc_auc_metric.reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.3, patience=2)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(ENGINEER_DIR / "train_features_neg_df.parquet").drop(columns=["userId", "movieId"], errors="ignore")
    val_df = pd.read_parquet(ENGINEER_DIR / "val_features_neg_df.parquet").drop(columns=["userId", "movieId"], errors="ignore")
    logger.info("train={} val={}", train_df.shape, val_df.shape)

    with open(ENGINEER_DIR / "item_metadata_pipeline.dill", "rb") as f:
        pipeline = dill.load(f)
    train_feats = build_item_features(train_df, pipeline)
    val_feats = build_item_features(val_df, pipeline)
    logger.info("train_feats={} val_feats={}", train_feats.shape, val_feats.shape)

    idm = IDMapper().load(ENGINEER_DIR / "idm.json")
    num_users = len(idm.user_to_index) + 1
    num_items = len(idm.item_to_index) + 1
    logger.info("num_users={} num_items={}", num_users, num_items)

    emb_weight = load_item2vec_embedding()
    emb_dim = int(emb_weight.shape[1])
    new_emb = nn.Embedding(num_items, emb_dim, padding_idx=num_items - 1)
    if emb_weight.shape[0] < num_items:
        pad = torch.zeros(num_items - emb_weight.shape[0], emb_dim)
        emb_weight = torch.cat([emb_weight, pad], dim=0)
    elif emb_weight.shape[0] > num_items:
        emb_weight = emb_weight[:num_items]
    new_emb.weight.data.copy_(emb_weight)
    logger.info("item embedding: {}", new_emb.weight.shape)

    model = Ranker(
        num_users, num_items, emb_dim,
        item_sequence_ts_bucket_size=HP["ts_bucket_size"],
        bucket_embedding_dim=HP["bucket_emb_dim"],
        item_feature_size=int(train_feats.shape[1]),
        item_embedding=new_emb, dropout=HP["dropout"],
    )

    train_loader = DataLoader(
        UserItemBinaryDFDataset(train_df, UCOL, ICOL, RCOL, TSCOL, item_feature=train_feats),
        batch_size=HP["batch_size"], shuffle=True, num_workers=0, collate_fn=collate,
    )
    val_loader = DataLoader(
        UserItemBinaryDFDataset(val_df, UCOL, ICOL, RCOL, TSCOL, item_feature=val_feats),
        batch_size=HP["batch_size"], shuffle=False, num_workers=0, collate_fn=collate,
    )

    lit = LitRankerMini(model, HP["lr"], HP["l2"], HP["neg_to_pos_ratio"])
    ckpt_cb = ModelCheckpoint(dirpath=str(FINAL_DIR), filename="best-checkpoint",
                              save_top_k=1, monitor="val_roc_auc", mode="max")
    trainer = L.Trainer(
        max_epochs=HP["max_epochs"], accelerator="cpu", devices=1,
        callbacks=[ckpt_cb, EarlyStopping(monitor="val_roc_auc", patience=1, mode="max", verbose=True)],
        logger=False, enable_checkpointing=True,
    )
    trainer.fit(lit, train_loader, val_loader)
    logger.info("best ckpt: {}", ckpt_cb.best_model_path)
    logger.info("val_roc_auc={:.4f}", trainer.callback_metrics.get("val_roc_auc", torch.tensor(0.0)).item())

    # Ensure the canonical best-checkpoint.ckpt filename exists.
    best = Path(ckpt_cb.best_model_path)
    dst = FINAL_DIR / "best-checkpoint.ckpt"
    if best != dst and best.exists():
        shutil.copyfile(best, dst)
    # Copy artifacts convert2onnx reads from final_model/.
    shutil.copyfile(ENGINEER_DIR / "idm.json", FINAL_DIR / "id_mapper.json")
    shutil.copyfile(ENGINEER_DIR / "item_metadata_pipeline.dill", FINAL_DIR / "item_metadata_pipeline.dill")
    logger.info("Done. Artifacts in {}", FINAL_DIR)


if __name__ == "__main__":
    main()