"""GRU sequence ranker training entrypoint — Ray Train + MLflow + Evidently.

Run:
    uv run python -m models.ranking_sequence.train --config configs/ranking_sequence.yaml

Ported back from the reference `src/model_ranking_sequence/main.py` (full MLOps):

  1. HP search — a manual loop of `num_samples` `TorchTrainer.fit()` runs
     sampling `learning_rate` / `l2_reg` (loguniform) + `dropout` (uniform);
     each trial logs params/metrics to MLflow experiment
     `ranking/hyperparameter_tuning`, and its best val_roc_auc is read back from
     the trial's Lightning checkpoint. (Replaces the reference's `tune.run` —
     Ray 2.55's v2 `TorchTrainer` is not a Tune trainable, so the manual loop is
     the v2-compatible equivalent, matching `models.item2vec.train`.)
  2. Final training — best trial's params → a final `TorchTrainer` run that logs
     to MLflow experiment `ranking/final_model`, saves the IDMapper + the item
     metadata pipeline + the PyTorch `Ranker` to the Model Registry
     (`ranking_sequence_rating`), and tags the best val_roc_auc version as
     `champion`.

The ranker consumes the **Item2Vec champion** item embeddings (frozen) loaded
from the MLflow Model Registry (`item2vec_skipgram` champion version), with a
fallback to the newest local `models/output/item2vec/final_model/best-checkpoint*
.ckpt` when the registry is unreachable.

Config is env-driven (`${VAR}` resolved from process env / `.env`) so the same
config runs local-pro (RAY_ADDRESS=local, MLflow via docker-compose) and AWS
(RAY_ADDRESS=auto on an EKS KubeRay head pod, MLflow in-cluster). See
`docs/eks-deploy.md` for the EKS path. As with `models.item2vec.train`, the
engineer data dir is absolutized against the repo root on local-pro because Ray
2.55's working_dir packaging honors `.gitignore` (which excludes
`feature/output/**/*.{jsonl,json,parquet}`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

import dill
import lightning as L
import mlflow
import numpy as np
import pandas as pd
import ray
import torch
import torch.nn as nn
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from mlflow.tracking import MlflowClient
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader

from feature.id_mapper import IDMapper
from models.ranking_sequence.dataset import UserItemBinaryDFDataset
from models.ranking_sequence.model import Ranker
from models.ranking_sequence.trainer import LitRanker
from src.data_prep_utils import chunk_transform

# loguru -> stderr only, INFO level.
logger.remove()
logger.add(sys.stderr, level="INFO")

ROOT = Path(__file__).resolve().parents[2]

# Load .env (now that ROOT is defined) BEFORE any config/env resolution so
# `${MLFLOW_*}` / `${RAY_ADDRESS}` placeholders resolve to the docker-compose
# MLflow + local Ray even when the shell has not exported them. Without this,
# an unset MLFLOW_TRACKING_URI resolves to "" and the Ray worker silently
# writes to a throwaway file:mlruns in its temp CWD instead of docker MLflow.
load_dotenv(ROOT / ".env")

_ENV_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _resolve_env(value: Any) -> Any:
    """Recursively replace `${VAR}` placeholders in a config value with env vars."""
    if isinstance(value, str):
        return _ENV_PLACEHOLDER.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the YAML config, resolve `${VAR}` placeholders, absolutize output paths."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = _resolve_env(cfg)

    for key in ("checkpoint_dir", "final_checkpoint_dir", "log_dir", "storage_path"):
        cfg["output"][key] = str((ROOT / cfg["output"][key]).resolve())
    return cfg


# ---------------------------------------------------------------------------
# Item2Vec champion embedding loading.
# ---------------------------------------------------------------------------


def _find_champion_version(model_name: str, tracking_uri: str, tag_key: str) -> str | None:
    """Return the version tagged `champion=true` for `model_name`, else the latest."""
    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    for v in versions:
        if str(v.tags.get(tag_key, "")).lower() == "true":
            return v.version
    return max(versions, key=lambda v: int(v.version)).version


def _load_champion_from_checkpoint(ckpt_path: Path) -> torch.Tensor:
    """Extract the item2vec embedding weight from a Lightning checkpoint on disk."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", {})
    # item2vec port's LitSkipGram stores the embedding under this key.
    for key in ("skipgram_model.embeddings.weight", "embeddings.weight"):
        if key in state:
            return state[key].detach().clone()
    raise KeyError(f"embedding weight not found in {ckpt_path} (keys={list(state)})")


def load_champion_embedding(cfg: Dict[str, Any]) -> torch.Tensor:
    """Load the Item2Vec champion item-embedding weight (MLflow registry, disk fallback).

    Args:
        cfg: Flat `train_loop_config` dict (carries `item2vec_model_name`,
            `champion_tag_key`, `mlflow_tracking_uri`).

    Returns:
        The `(vocab+1, embedding_dim)` embedding weight tensor.
    """
    model_name = cfg["item2vec_model_name"]
    tag_key = cfg["champion_tag_key"]
    tracking_uri = cfg["mlflow_tracking_uri"]

    # Primary: MLflow Model Registry champion.
    try:
        version = _find_champion_version(model_name, tracking_uri, tag_key)
        if version is not None:
            uri = f"models:/{model_name}/{version}"
            logger.info(f"Loading Item2Vec champion from MLflow: {uri}")
            champ = mlflow.pytorch.load_model(model_uri=uri, map_location="cpu")
            emb = getattr(champ, "embeddings", None)
            if emb is not None and hasattr(emb, "weight"):
                logger.info(f"Loaded champion embedding from MLflow: {emb.weight.shape}")
                return emb.weight.detach().clone()
            logger.warning("Champion model has no .embeddings.weight attr; falling back to disk.")
    except Exception as e:
        logger.warning(f"MLflow champion load failed ({e}); falling back to disk checkpoint.")

    # Fallback: newest local item2vec final checkpoint.
    final_dir = ROOT / "models" / "output" / "item2vec" / "final_model"
    candidates = sorted(final_dir.glob("best-checkpoint*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No Item2Vec champion in MLflow and no checkpoint under {final_dir}."
        )
    ckpt = candidates[-1]
    logger.info(f"Loading Item2Vec embedding from disk checkpoint: {ckpt}")
    return _load_champion_from_checkpoint(ckpt)


# ---------------------------------------------------------------------------
# Ray Train per-worker training loop (HP search trials + final training).
# ---------------------------------------------------------------------------


def _read_best_metric(ckpt_path: Path) -> float | None:
    """Read the best monitored metric (val_roc_auc) from a Lightning checkpoint."""
    if not ckpt_path.exists():
        return None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for cb in ck.get("callbacks", {}).values():
        if isinstance(cb, dict) and "best_model_score" in cb:
            score = cb["best_model_score"]
            return float(score.item()) if hasattr(score, "item") else float(score)
    return None


def _collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack a list of dataset dicts into a batched dict of tensors."""
    return {
        "user": torch.stack([x["user"] for x in batch]),
        "item_sequence": torch.stack([x["item_sequence"] for x in batch]),
        "item": torch.stack([x["item"] for x in batch]),
        "rating": torch.stack([x["rating"] for x in batch]),
        "item_sequence_ts_bucket": torch.stack([x["item_sequence_ts_bucket"] for x in batch]),
        "item_feature": torch.stack([x["item_feature"] for x in batch]),
    }


def _train_loop_config(cfg: Dict[str, Any], checkpoint_dir: str) -> Dict[str, Any]:
    """Build the flat train_loop_config dict consumed by `train_func`."""
    t = cfg["training"]
    d = cfg["dataset"]
    tr = cfg["trainer"]
    return {
        "data_path": cfg["data"]["engineer_dir"],
        "train_data_file": cfg["data"]["train_data_file"],
        "val_data_file": cfg["data"]["val_data_file"],
        "idm_file": cfg["data"]["idm_file"],
        "item_metadata_pipeline_file": cfg["data"]["item_metadata_pipeline_file"],
        "checkpoint_dir": checkpoint_dir,
        "run_name": cfg["experiment"]["run_name"],
        "dataset_version": cfg["experiment"]["dataset_version"],
        "mlflow_tracking_uri": cfg["mlflow"]["tracking_uri"],
        "item2vec_model_name": cfg["item2vec"]["model_name"],
        "champion_tag_key": cfg["item2vec"]["champion_tag_key"],
        # dataset cols
        "user_col": d["user_col"],
        "item_col": d["item_col"],
        "rating_col": d["rating_col"],
        "timestamp_col": d["timestamp_col"],
        "required_columns": d["required_columns"],
        # training
        "max_epochs": t["max_epochs"],
        "batch_size": t["batch_size"],
        "num_workers": t["num_workers"],
        "learning_rate": t["learning_rate"]["loguniform"][0],
        "l2_reg": t["l2_reg"]["loguniform"][0],
        "dropout": t["dropout"]["uniform"][0],
        "neg_to_pos_ratio": t["neg_to_pos_ratio"],
        "top_k": t["top_k"],
        "item_sequence_ts_bucket_size": t["item_sequence_ts_bucket_size"],
        "bucket_embedding_dim": t["bucket_embedding_dim"],
        # trainer
        "accelerator": tr["accelerator"],
        "checkpoint_filename": tr["checkpoint"]["filename"],
        "checkpoint_save_top_k": tr["checkpoint"]["save_top_k"],
        "checkpoint_monitor": tr["checkpoint"]["monitor"],
        "checkpoint_mode": tr["checkpoint"]["mode"],
        "early_stopping_monitor": tr["early_stopping"]["monitor"],
        "early_stopping_patience": tr["early_stopping"]["patience"],
        "early_stopping_mode": tr["early_stopping"]["mode"],
        "early_stopping_verbose": tr["early_stopping"]["verbose"],
    }


def _build_item_features(
    df: pd.DataFrame, pipeline, required_columns: list[str]
) -> np.ndarray:
    """Fill missing required columns then apply the item-metadata pipeline."""
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0.0
    return chunk_transform(df, pipeline, chunk_size=10000).astype(np.float32)


def train_func(config: Dict[str, Any], is_final_training: bool = False) -> None:
    """Train one GRU `Ranker` with the given (flat) config inside a Ray worker."""
    from feature.id_mapper import IDMapper  # imported in-worker for Ray runtime_env

    logger.info(f"cwd={os.getcwd()} is_final_training={is_final_training}")

    # --- MLflow init -------------------------------------------------------
    mlflow_run = None
    model_name = f"{config['run_name']}_sequence_rating"
    try:
        mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
        base = config["run_name"]
        experiment_name = f"{base}/final_model" if is_final_training else f"{base}/hyperparameter_tuning"
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        tags = {
            "phase": "final_training" if is_final_training else "tuning",
            "model_type": "sequence_rating",
            "dataset_version": config["dataset_version"],
        }
        run_name = "Final Model - SequenceRanker" if is_final_training else "Tuning Run - SequenceRanker"
        mlflow_run = mlflow.start_run(run_name=run_name, tags=tags)
        mlflow.log_params(
            {
                f"{'final' if is_final_training else 'tuning'}.{k}": v
                for k, v in config.items()
                if k in ("learning_rate", "l2_reg", "dropout", "batch_size", "max_epochs")
            }
        )
    except Exception as e:
        logger.warning(f"MLflow init failed: {e}")

    try:
        data_path = config["data_path"]
        train_df = pd.read_parquet(os.path.join(data_path, config["train_data_file"]))
        val_df = pd.read_parquet(os.path.join(data_path, config["val_data_file"]))
        # Drop raw IDs; the indice columns are what the model uses.
        train_df = train_df.drop(columns=["userId", "movieId"], errors="ignore")
        val_df = val_df.drop(columns=["userId", "movieId"], errors="ignore")

        with open(os.path.join(data_path, config["item_metadata_pipeline_file"]), "rb") as f:
            item_metadata_pipeline = dill.load(f)

        required_columns = config["required_columns"]
        train_item_features = _build_item_features(train_df, item_metadata_pipeline, required_columns)
        val_item_features = _build_item_features(val_df, item_metadata_pipeline, required_columns)
        logger.info(
            f"train rows={len(train_df)} feats={train_item_features.shape}; "
            f"val rows={len(val_df)} feats={val_item_features.shape}"
        )

        idm = IDMapper().load(os.path.join(data_path, config["idm_file"]))
        num_users = len(idm.user_to_index) + 1
        num_items = len(idm.item_to_index) + 1
        logger.info(f"num_users={num_users} num_items={num_items}")

        # --- Frozen Item2Vec champion item embedding -----------------------
        emb_weight = load_champion_embedding(config)
        emb_dim = int(emb_weight.shape[1])
        new_emb = nn.Embedding(num_items, emb_dim, padding_idx=num_items - 1)
        if emb_weight.shape[0] < num_items:
            pad = torch.zeros(num_items - emb_weight.shape[0], emb_dim)
            emb_weight = torch.cat([emb_weight, pad], dim=0)
        elif emb_weight.shape[0] > num_items:
            emb_weight = emb_weight[:num_items]
        new_emb.weight.data.copy_(emb_weight)
        logger.info(f"ranker item embedding: {new_emb.weight.shape} (frozen item2vec)")

        # --- all-items feature matrix (for ranking eval in final training) -
        all_items = list(idm.item_to_index.values()) + [num_items - 1]
        all_items_df = pd.DataFrame({config["item_col"]: all_items})
        item_meta = pd.concat([train_df, val_df]).drop_duplicates(subset=[config["item_col"]])
        all_items_df = all_items_df.merge(item_meta, on=config["item_col"], how="left")
        for col in all_items_df.columns:
            if all_items_df[col].dtype == object:
                all_items_df[col] = all_items_df[col].fillna("")
            else:
                all_items_df[col] = all_items_df[col].fillna(0.0)
        all_items_features = _build_item_features(
            all_items_df, item_metadata_pipeline, required_columns
        )
        all_items_indices = all_items_df[config["item_col"]].values.astype(np.int64)
        logger.info(f"all_items_features={all_items_features.shape}")

        # --- DataLoaders ----------------------------------------------------
        ucol, icol, rcol, tscol = (
            config["user_col"], config["item_col"], config["rating_col"], config["timestamp_col"]
        )
        train_loader = DataLoader(
            UserItemBinaryDFDataset(train_df, ucol, icol, rcol, tscol, item_feature=train_item_features),
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["num_workers"]),
            collate_fn=_collate_fn,
        )
        val_loader = DataLoader(
            UserItemBinaryDFDataset(val_df, ucol, icol, rcol, tscol, item_feature=val_item_features),
            batch_size=int(config["batch_size"]),
            shuffle=False,
            num_workers=int(config["num_workers"]),
            collate_fn=_collate_fn,
        )

        # --- Model + LitRanker ---------------------------------------------
        model = Ranker(
            num_users,
            num_items,
            emb_dim,
            item_sequence_ts_bucket_size=int(config["item_sequence_ts_bucket_size"]),
            bucket_embedding_dim=int(config["bucket_embedding_dim"]),
            item_feature_size=int(train_item_features.shape[1]),
            item_embedding=new_emb,
            dropout=float(config["dropout"]),
        )
        args_ns = type("ArgsNS", (), {})()
        for k, v in config.items():
            setattr(args_ns, k, v)
        args_ns.top_K = int(config["top_k"])

        # trial dir: final training writes directly under checkpoint_dir; trials
        # under trial_{id} (lazy resolution — get_trial_id raises outside Tune).
        if is_final_training:
            trial_suffix = ""
        else:
            trial_id = config.get("trial_id")
            if trial_id is None:
                trial_id = train.get_context().get_trial_id()
            trial_suffix = f"trial_{trial_id}"
        trial_dir = (
            os.path.join(config["checkpoint_dir"], trial_suffix)
            if trial_suffix
            else config["checkpoint_dir"]
        )
        os.makedirs(trial_dir, exist_ok=True)

        ckpt_cb = ModelCheckpoint(
            dirpath=trial_dir,
            filename=config["checkpoint_filename"],
            save_top_k=config["checkpoint_save_top_k"],
            monitor=config["checkpoint_monitor"],
            mode=config["checkpoint_mode"],
        )
        lit = LitRanker(
            model=model,
            learning_rate=float(config["learning_rate"]),
            l2_reg=float(config["l2_reg"]),
            log_dir=trial_dir,
            evaluate_ranking=is_final_training,
            idm=idm,
            all_items_indices=all_items_indices,
            all_items_features=all_items_features,
            args=args_ns,
            neg_to_pos_ratio=int(config["neg_to_pos_ratio"]),
            checkpoint_callback=ckpt_cb,
        )

        trainer = L.Trainer(
            max_epochs=int(config["max_epochs"]),
            accelerator=config["accelerator"],
            callbacks=[
                ckpt_cb,
                EarlyStopping(
                    monitor=config["early_stopping_monitor"],
                    patience=int(config["early_stopping_patience"]),
                    mode=config["early_stopping_mode"],
                    verbose=config["early_stopping_verbose"],
                ),
            ],
            logger=False,
        )
        trainer.fit(lit, train_loader, val_loader)

        val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf"))).item()
        val_roc_auc = trainer.callback_metrics.get("val_roc_auc", torch.tensor(0.0)).item()
        logger.info(f"Training completed: val_loss={val_loss:.6f} val_roc_auc={val_roc_auc:.6f}")

        if mlflow_run:
            try:
                mlflow.log_metric("final.val_loss" if is_final_training else "tuning.val_loss", float(val_loss))
                mlflow.log_metric(
                    "final.val_roc_auc" if is_final_training else "tuning.val_roc_auc",
                    float(val_roc_auc),
                )
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")

        # --- Final: log model to Model Registry + tag champion -------------
        if is_final_training:
            _log_final_ranker_to_mlflow(config, lit, idm, item_metadata_pipeline, model_name, val_roc_auc)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if mlflow_run:
            try:
                mlflow.end_run()
            except Exception:
                pass


def _log_final_ranker_to_mlflow(
    config: Dict[str, Any],
    lit: LitRanker,
    idm: IDMapper,
    item_metadata_pipeline,
    model_name: str,
    val_roc_auc: float,
) -> None:
    """Log the Ranker + artifacts to the MLflow Model Registry and tag champion (max ROC-AUC)."""
    try:
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, TensorSpec

        in_schema = Schema(
            [
                TensorSpec(name="user", type=np.dtype(np.int64), shape=(-1,)),
                TensorSpec(name="item_sequence", type=np.dtype(np.int64), shape=(-1, -1)),
                TensorSpec(name="item_sequence_ts_bucket", type=np.dtype(np.int64), shape=(-1, -1)),
                TensorSpec(name="item_feature", type=np.dtype(np.float32), shape=(-1, -1)),
                TensorSpec(name="target_item", type=np.dtype(np.int64), shape=(-1,)),
            ]
        )
        out_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1, 1))])
        signature = ModelSignature(inputs=in_schema, outputs=out_schema)

        mlflow.pytorch.log_model(
            pytorch_model=lit.model,
            artifact_path="sequence_rating_model",
            registered_model_name=model_name,
            signature=signature,
            metadata={
                "model_type": "SequenceRatingPrediction",
                "task": "rating-prediction",
                "framework": "pytorch",
            },
        )
        logger.info("Ranker model logged to MLflow Model Registry.")

        # id_mapper + item_metadata pipeline as artifacts.
        log_dir = lit.log_dir
        idm_path = os.path.join(log_dir, "id_mapper.json")
        idm.save(idm_path)
        mlflow.log_artifact(idm_path, artifact_path="id_mapper")
        pipe_path = os.path.join(log_dir, "item_metadata_pipeline.dill")
        with open(pipe_path, "wb") as f:
            dill.dump(item_metadata_pipeline, f)
        mlflow.log_artifact(pipe_path, artifact_path="item_metadata_pipeline")

        # Champion tagging (maximize val_roc_auc).
        client = MlflowClient(tracking_uri=config["mlflow_tracking_uri"])
        latest = client.get_latest_versions(model_name, stages=["None"])
        if not latest:
            return
        this_version = latest[0]
        is_champion = True
        worse = []
        for v in client.search_model_versions(f"name='{model_name}'"):
            if v.run_id == this_version.run_id:
                continue
            try:
                other = client.get_run(v.run_id).data.metrics.get("final.val_roc_auc")
                if other is not None and other > val_roc_auc:
                    is_champion = False
                    break
                if other is not None:
                    worse.append(v)
            except Exception:
                pass
        if is_champion:
            client.set_model_version_tag(
                name=model_name, version=this_version.version,
                key=config["champion_tag_key"], value="true",
            )
            logger.info(f"Model version {this_version.version} is CHAMPION (val_roc_auc={val_roc_auc:.4f})")
            for v in worse:
                try:
                    client.delete_model_version_tag(
                        name=model_name, version=v.version, key=config["champion_tag_key"]
                    )
                    logger.info(f"Removed champion tag from version {v.version}")
                except Exception as e:
                    logger.warning(f"Failed to remove champion tag from v{v.version}: {e}")
        else:
            logger.info(f"Model version {this_version.version} is NOT champion (val_roc_auc={val_roc_auc:.4f})")
    except Exception as e:
        logger.error(f"Failed to log final ranker to MLflow: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GRU sequence ranker training (Ray + MLflow).")
    parser.add_argument("--config", default=str(ROOT / "configs" / "ranking_sequence.yaml"))
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Skip the HP search and run final training only, loading best params "
        "from models/output/ranking_sequence/reports/hp_search_results.json. "
        "Use to re-register the model in MLflow after a backend reset without "
        "re-running the full HP search.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")
    for key in ("checkpoint_dir", "final_checkpoint_dir", "log_dir", "storage_path"):
        os.makedirs(cfg["output"][key], exist_ok=True)

    mcfg = cfg["mlflow"]
    raw_address = cfg["ray"]["address"].strip()
    ray_address = None if raw_address.lower() in ("", "local") else raw_address

    ray_env = {
        "working_dir": str(ROOT),
        "excludes": [
            ".venv/", ".git/", "data/", "notebooks/",
            "models/output/", "mlruns/",
            "*.ckpt", "*.pyc",
        ],
        "env_vars": {
            "PYTHONPATH": str(ROOT),
            "MLFLOW_S3_ENDPOINT_URL": mcfg["s3_endpoint_url"],
            "AWS_ACCESS_KEY_ID": mcfg["aws_access_key_id"],
            "AWS_SECRET_ACCESS_KEY": mcfg["aws_secret_access_key"],
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            "MLFLOW_TRACKING_URI": mcfg["tracking_uri"],
            "MLFLOW_S3_IGNORE_TLS": str(mcfg["s3_ignore_tls"]).lower(),
        },
    }
    ray.init(address=ray_address, runtime_env=ray_env)

    use_gpu = cfg["trainer"]["use_gpu"]
    is_local = ray_address is None
    data_path = (
        str((ROOT / cfg["data"]["engineer_dir"]).resolve())
        if is_local
        else cfg["data"]["engineer_dir"]
    )

    lr_range = cfg["training"]["learning_rate"]["loguniform"]
    l2_range = cfg["training"]["l2_reg"]["loguniform"]
    dropout_range = cfg["training"]["dropout"]["uniform"]
    num_samples = cfg["experiment"]["tune_config"]["num_samples"]
    tune_metric = cfg["experiment"]["tune_config"]["metric"]  # "val_roc_auc"
    tune_mode = cfg["experiment"]["tune_config"]["mode"]       # "max"

    best_val_roc_auc = -float("inf")
    best_params: Dict[str, Any] = {}

    if args.final_only:
        # Re-use a completed HP search's best params (no re-tuning).
        reports_dir = ROOT / "models" / "output" / "ranking_sequence" / "reports"
        hp_path = reports_dir / "hp_search_results.json"
        if not hp_path.exists():
            raise FileNotFoundError(
                f"--final-only requires {hp_path}; run a full HP search first."
            )
        with open(hp_path) as f:
            hp = json.load(f)
        best_params = dict(hp["best_params"])
        best_val_roc_auc = float(hp.get("best_val_roc_auc", -float("inf")))
        logger.info(
            f"--final-only: loaded best params from {hp_path}: "
            f"{best_params} (val_roc_auc={best_val_roc_auc})"
        )
    else:
        rng = np.random.default_rng(seed=42)
        trial_results: list[Dict[str, Any]] = []

        logger.info(f"=== Manual HP search: {num_samples} trials (metric={tune_metric}, mode={tune_mode}) ===")
        for i in range(num_samples):
            lr = float(10 ** rng.uniform(np.log10(lr_range[0]), np.log10(lr_range[1])))
            l2 = float(10 ** rng.uniform(np.log10(l2_range[0]), np.log10(l2_range[1])))
            dropout = float(rng.uniform(dropout_range[0], dropout_range[1]))
            logger.info(f"--- Trial {i + 1}/{num_samples}: lr={lr:.6f} l2={l2:.6f} dropout={dropout:.3f} ---")

            trial_config = _train_loop_config(cfg, cfg["output"]["checkpoint_dir"])
            trial_config["data_path"] = data_path
            trial_config["trial_id"] = i
            trial_config["learning_rate"] = lr
            trial_config["l2_reg"] = l2
            trial_config["dropout"] = dropout

            trial_trainer = TorchTrainer(
                train_loop_per_worker=train_func,
                train_loop_config=trial_config,
                scaling_config=ScalingConfig(num_workers=1, use_gpu=use_gpu),
            )
            try:
                trial_trainer.fit()
            except Exception as e:
                logger.error(f"Trial {i + 1} failed: {e}")
                trial_results.append(
                    {"trial": i, "learning_rate": lr, "l2_reg": l2, "dropout": dropout,
                     "val_roc_auc": None, "error": str(e)}
                )
                continue

            trial_ckpt = (
                Path(cfg["output"]["checkpoint_dir"]) / f"trial_{i}" / cfg["trainer"]["checkpoint"]["filename"]
            ).with_suffix(".ckpt")
            val_roc_auc = _read_best_metric(trial_ckpt)
            logger.info(f"Trial {i + 1} val_roc_auc={val_roc_auc}")
            trial_results.append(
                {"trial": i, "learning_rate": lr, "l2_reg": l2, "dropout": dropout, "val_roc_auc": val_roc_auc}
            )
            if val_roc_auc is not None and val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                best_params = {"learning_rate": lr, "l2_reg": l2, "dropout": dropout}

        reports_dir = ROOT / "models" / "output" / "ranking_sequence" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "hp_search_results.json", "w") as f:
            json.dump(
                {"metric": tune_metric, "mode": tune_mode, "trials": trial_results,
                 "best_val_roc_auc": best_val_roc_auc, "best_params": best_params},
                f, indent=2,
            )
        logger.info(f"HP search done. best_val_roc_auc={best_val_roc_auc} best_params={best_params}")

        if not best_params:
            raise RuntimeError("All HP trials failed — cannot run final training.")

    logger.info("Starting final training with best parameters...")
    final_config = _train_loop_config(cfg, cfg["output"]["final_checkpoint_dir"])
    final_config["data_path"] = data_path
    final_config.update(best_params)

    final_trainer = TorchTrainer(
        train_loop_per_worker=lambda c: train_func(c, is_final_training=True),
        train_loop_config=final_config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=use_gpu),
    )
    final_trainer.fit()
    logger.info("Final training completed!")
    logger.info(f"Final ranker checkpoint saved at: {cfg['output']['final_checkpoint_dir']}")


if __name__ == "__main__":
    main()