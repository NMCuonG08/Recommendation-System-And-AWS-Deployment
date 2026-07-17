"""LightningModule wrapping the GRU `Ranker` for Ray Train + MLflow training.

Ported from the reference `src/model_ranking_sequence/trainer.py`, adapted for
this port:

  - `logger=False` compatibility: classification/ranking Evidently reports +
    metrics are logged via the active MLflow run (`mlflow.active_run()`) rather
    than a Lightning `MLFlowLogger` — same pattern as `models.item2vec.trainer`.
  - The reference's `model_ranking_sequence.viz.color_scheme` Evidently option
    is dropped (no `viz` module in this port); reports render with defaults.
  - `_log_ranking_metrics` is wrapped in a best-effort try/except so a missing
    Evidently recsys metric (or any eval-time error) cannot fail training.
  - Device resolution uses Lightning's `self.device` instead of the reference's
    hardcoded `accelerator` string.
"""

from __future__ import annotations

import os
from typing import Any

import lightning as L
import mlflow
import numpy as np
import pandas as pd
import torch
from evidently.metric_preset import ClassificationPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from loguru import logger
from torch import nn
from torchmetrics import AUROC

from models.ranking_sequence.model import Ranker


class LitRanker(L.LightningModule):
    """Lightning module training the GRU `Ranker` with weighted BCE + ROC-AUC."""

    def __init__(
        self,
        model: Ranker,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
        evaluate_ranking: bool = False,
        idm: Any | None = None,
        all_items_indices: Any | None = None,
        all_items_features: Any | None = None,
        args: Any | None = None,
        neg_to_pos_ratio: int = 3,
        checkpoint_callback: Any | None = None,
    ):
        """Initialize the LitRanker.

        Args:
            model: The GRU `Ranker` to train.
            learning_rate: Adam learning rate.
            l2_reg: Adam weight decay (L2).
            log_dir: Directory for Evidently HTML reports.
            evaluate_ranking: If True (final training only), also compute
                offline ranking metrics (NDCG/F-beta/personalization) via
                `model.recommend()` over all candidate items.
            idm: IDMapper (for ranking eval ID resolution).
            all_items_indices: Candidate item indices (for ranking eval).
            all_items_features: Candidate item feature matrix (for ranking eval).
            args: Namespace-like object carrying `top_K`, `top_k`, column names,
                `timestamp_col` (for ranking eval).
            neg_to_pos_ratio: Positive-class weight for the weighted BCE loss
                (counteracts the negative-sample majority).
            checkpoint_callback: Optional ModelCheckpoint; if provided, the best
                checkpoint is reloaded before eval so reported metrics reflect
                the best-val model.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir
        self.evaluate_ranking = evaluate_ranking
        self.idm = idm
        self.all_items_indices = all_items_indices
        self.all_items_features = all_items_features
        self.args = args
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.checkpoint_callback = checkpoint_callback

        self.val_roc_auc_metric = AUROC(task="binary")

    # ------------------------------------------------------------------ loss

    def _get_loss_fn(self, weights: torch.Tensor) -> nn.Module:
        return nn.BCELoss(weights)

    def _get_device(self) -> torch.device:
        return self.device

    # -------------------------------------------------------------- training

    def training_step(self, batch, batch_idx):
        labels = batch["rating"].float()
        predictions = self.model.forward(
            batch["user"],
            batch["item_sequence"],
            batch["item_sequence_ts_bucket"],
            batch["item_feature"],
            batch["item"],
        ).view(labels.shape)
        weights = torch.where(labels == 1, float(self.neg_to_pos_ratio), 1.0)
        loss = self._get_loss_fn(weights)(predictions, labels)

        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["rating"]
        predictions = self.model.forward(
            batch["user"],
            batch["item_sequence"],
            batch["item_sequence_ts_bucket"],
            batch["item_feature"],
            batch["item"],
        ).view(labels.shape)
        weights = torch.where(labels == 1, float(self.neg_to_pos_ratio), 1.0)
        loss = self._get_loss_fn(weights)(predictions, labels)

        self.val_roc_auc_metric.update(predictions, labels.int())
        self.log(
            "val_roc_auc",
            self.val_roc_auc_metric.compute(),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)
        self.log("val_roc_auc", self.val_roc_auc_metric.compute(), sync_dist=True)
        self.val_roc_auc_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.3, patience=2
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # -------------------------------------------------------------- fit hooks

    def on_fit_end(self):
        """Reload best checkpoint (if any) and log Evidently reports to MLflow."""
        if self.checkpoint_callback is not None:
            try:
                logger.info(
                    f"Loading best model from {self.checkpoint_callback.best_model_path}..."
                )
                self.model = LitRanker.load_from_checkpoint(
                    self.checkpoint_callback.best_model_path, model=self.model
                ).model
            except Exception as e:
                logger.warning(f"Could not reload best checkpoint: {e}")
        self.model = self.model.to(self._get_device())
        try:
            logger.info("Logging classification metrics...")
            self._log_classification_metrics()
        except Exception as e:
            logger.warning(f"Classification metrics logging failed: {e}")
        if self.evaluate_ranking:
            try:
                logger.info("Logging ranking metrics...")
                self._log_ranking_metrics()
            except Exception as e:
                logger.warning(f"Ranking metrics logging failed: {e}")

    # ------------------------------------------------------------- eval logs

    def _log_classification_metrics(self):
        """Generate + log an Evidently classification report on the val set."""
        self.model.eval()
        val_loader = self.trainer.val_dataloaders
        device = self._get_device()

        labels, classifications = [], []
        for batch in val_loader:
            preds = self.model.predict(
                batch["user"].to(device),
                batch["item_sequence"].to(device),
                batch["item_sequence_ts_bucket"].to(device),
                batch["item_feature"].to(device),
                batch["item"].to(device),
            ).view(-1)
            labels.extend(batch["rating"].cpu().detach().numpy())
            classifications.extend(preds.cpu().detach().numpy())

        eval_df = pd.DataFrame(
            {"labels": labels, "classification_proba": classifications}
        ).assign(label=lambda d: d["labels"].gt(0).astype(int))
        self.eval_classification_df = eval_df

        report = Report(metrics=[ClassificationPreset()])
        report.run(
            reference_data=None,
            current_data=eval_df[["label", "classification_proba"]],
            column_mapping=ColumnMapping(
                target="label", prediction="classification_proba"
            ),
        )
        os.makedirs(self.log_dir, exist_ok=True)
        html_fp = os.path.join(self.log_dir, "evidently_report_classification.html")
        report.save_html(html_fp)
        logger.info(f"Saved Evidently classification report: {html_fp}")

        if mlflow.active_run():
            try:
                mlflow.log_artifact(html_fp)
            except Exception as e:
                logger.warning(f"Failed to log classification artifact: {e}")
            try:
                for m in report.as_dict()["metrics"]:
                    if m["metric"] == "ClassificationQualityMetric":
                        mlflow.log_metric(
                            "val_roc_auc",
                            float(m["result"]["current"]["roc_auc"]),
                        )
            except Exception as e:
                logger.warning(f"Failed to log classification metrics: {e}")

    def _log_ranking_metrics(self):
        """Generate + log offline ranking metrics (NDCG/F-beta/personalization)."""
        from evidently.metrics import FBetaTopKMetric, NDCGKMetric, PersonalizationMetric

        ts = self.args.timestamp_col
        rc = self.args.rating_col
        uc = self.args.user_col
        ic = self.args.item_col
        K = int(getattr(self.args, "top_K", None) or getattr(self.args, "top_k", 10))
        k = int(getattr(self.args, "top_k", K))
        idm = self.idm

        val_loaders = self.trainer.val_dataloaders
        ds = val_loaders[0].dataset if isinstance(val_loaders, list) else val_loaders.dataset
        df = ds.df.copy()

        if df[uc].dtype != "int64" and hasattr(idm, "get_user_index"):
            df[uc] = df[uc].map({u: idm.get_user_index(u) for u in df[uc].unique()}).astype("int64")
        if df[ic].dtype != "int64" and hasattr(idm, "get_item_index"):
            df[ic] = df[ic].map({i: idm.get_item_index(i) for i in df[ic].unique()}).astype("int64")

        to_rec = df.sort_values(ts).drop_duplicates(subset=[uc])
        if len(to_rec) > 3000:
            to_rec = to_rec.iloc[np.random.choice(len(to_rec), 3000, replace=False)]

        user_ids = to_rec[uc].values
        item_sequences = np.stack(to_rec["item_sequence"].values)
        bucket_col = [c for c in to_rec.columns if "ts_bucket" in c.lower()][0]
        item_ts_buckets = np.stack(to_rec[bucket_col].values)

        device = self._get_device()
        item_features = torch.tensor(self.all_items_features, device=device)
        item_indices = torch.tensor(self.all_items_indices, device=device)

        self.model.eval()
        recs = []
        for i in range(0, len(user_ids), 1024):
            with torch.no_grad():
                rec_batch = self.model.recommend(
                    torch.tensor(user_ids[i : i + 1024], device=device),
                    torch.tensor(item_sequences[i : i + 1024], device=device),
                    torch.tensor(item_ts_buckets[i : i + 1024], device=device),
                    item_features,
                    item_indices,
                    k=K,
                    batch_size=K,
                )
            recs.append(rec_batch.cpu().numpy() if isinstance(rec_batch, torch.Tensor) else rec_batch)
        recs = np.vstack(recs)  # [num_users, K]

        def personalization_at_k(arr):
            if len(arr) < 2:
                return 1.0
            total, count = 0.0, 0
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    a, b = set(arr[i]), set(arr[j])
                    if not a and not b:
                        continue
                    total += len(a & b) / len(a | b)
                    count += 1
            return 1 - (total / count) if count else 1.0

        personalization = personalization_at_k(recs)

        rec_df = pd.DataFrame(
            {
                uc: user_ids.repeat(K),
                ic: recs.flatten(),
                "rec_ranking": np.tile(np.arange(1, K + 1), len(user_ids)),
            }
        )
        label_df = df[[uc, ic, rc, ts]].groupby([uc, ic], as_index=False).first()
        eval_df = pd.merge(rec_df, label_df, on=[uc, ic], how="left")
        eval_df[rc] = eval_df[rc].fillna(0.0)
        self.eval_ranking_df = eval_df

        report = Report(
            metrics=[NDCGKMetric(k=k), FBetaTopKMetric(k=k), PersonalizationMetric(k=k)]
        )
        report.run(
            reference_data=None,
            current_data=eval_df,
            column_mapping=ColumnMapping(
                recommendations_type="rank",
                target=rc,
                prediction="rec_ranking",
                item_id=ic,
                user_id=uc,
            ),
        )
        os.makedirs(self.log_dir, exist_ok=True)
        html_fp = os.path.join(self.log_dir, "evidently_report_ranking.html")
        report.save_html(html_fp)
        logger.info(f"Saved Evidently ranking report: {html_fp}")

        if mlflow.active_run():
            try:
                mlflow.log_metric("val_personalization_at_K", float(personalization))
                mlflow.log_artifact(html_fp)
                for m in report.as_dict()["metrics"]:
                    mt = m["metric"].replace("@", "_at_")
                    if mt == "PersonalizationMetric":
                        mlflow.log_metric(f"val_{mt}", float(m["result"]["current_value"]))
                    else:
                        for step, val in m["result"]["current"].items():
                            try:
                                mlflow.log_metric(
                                    f"val_{mt}_as_step", float(val), step=int(step)
                                )
                            except (ValueError, TypeError):
                                # Non-integer step (e.g. NaN k) — log without step.
                                mlflow.log_metric(f"val_{mt}", float(val))
            except Exception as e:
                logger.warning(f"Failed to log ranking metrics: {e}")