"""Item2Vec training entrypoint — Ray Train + MLflow + Evidently.

Run:
    uv run python -m models.item2vec.train --config configs/item2vec.yaml
    uv run python -m models.item2vec.train --config configs/item2vec.yaml --overfit

Ported back from the reference `src/model_item2vec/main.py` (full MLOps):

  1. (opt-in, --overfit) Overfit sanity check — train on one batch
     (`batch_sequences_overfit.jsonl`) for many epochs; expect val_loss -> ~0.
     Off by default to match the reference flow.
  2. HP search — a manual loop of `num_samples` `TorchTrainer.fit()` runs
     sampling `embedding_dim` (choice), `learning_rate` / `l2_reg` (loguniform);
     each trial logs params/metrics to MLflow experiment
     `item2vec/hyperparameter_tuning`, and its best val_loss is read back from
     the trial's Lightning checkpoint. (Replaces a Ray `Tuner`-based search —
     Ray 2.55's v2 `TorchTrainer` is not a Tune trainable, so `Tuner(trainer)`
     raises; the manual loop is the v2-compatible equivalent.)
  3. Final training — best trial's params → a final `TorchTrainer` run that logs
     to MLflow experiment `item2vec/final_model`, saves `idm.json` + the
     TorchScript model to the MLflow Model Registry (`item2vec_skipgram`), and
     tags the best val_loss version as `champion`.

Config is env-driven (`${VAR}` resolved from process env / `.env`) so the same
config runs local-pro (RAY_ADDRESS=local, MLflow via docker-compose) and AWS
(RAY_ADDRESS=auto on an EKS KubeRay head pod, MLflow in-cluster via the
`infra/mlflow-stack` helm chart). See `docs/eks-deploy.md` for the EKS path.

Note: on local-pro the engineer data dir is absolutized against the repo root
before being shipped to workers, because Ray 2.55's working_dir packaging
honors `.gitignore` (which excludes `feature/output/**/*.{jsonl,json}`) and the
shipped package would otherwise lack the data. On EKS the path stays relative
and ships via runtime_env.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

import lightning as L
import mlflow
import numpy as np
import psutil
import ray
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from loguru import logger
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from torch.utils.data import DataLoader

from feature.id_mapper import IDMapper
from models.item2vec.dataset import SkipGramDataset
from models.item2vec.model import SkipGram
from models.item2vec.trainer import LitSkipGram

# loguru -> stderr only, INFO level.
logger.remove()
logger.add(sys.stderr, level="INFO")

# Project root (so `feature/output/...` and `models/output/...` resolve from cwd).
ROOT = Path(__file__).resolve().parents[2]

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
    """Load the YAML config, resolve `${VAR}` placeholders, and absolutize paths."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg = _resolve_env(cfg)

    # Convert loguniform ranges to float (Ray Tune needs floats).
    for param in ("learning_rate", "l2_reg"):
        spec = cfg["training"].get(param)
        if isinstance(spec, dict) and "loguniform" in spec:
            spec["loguniform"] = [float(x) for x in spec["loguniform"]]

    # Driver-side output paths resolve against ROOT (driver runs from repo root,
    # so cwd == ROOT locally and == /app on the EKS head pod). The data path
    # (`engineer_dir`) is intentionally left relative so Ray workers can resolve
    # it against their own `cwd` (= the shipped working_dir, where the data is
    # shipped via runtime_env) — absolutizing it against the driver's ROOT would
    # point workers at a path that doesn't exist on their filesystem.
    for key in ("checkpoint_dir", "final_checkpoint_dir", "log_dir", "storage_path"):
        cfg["output"][key] = str((ROOT / cfg["output"][key]).resolve())
    return cfg


# ---------------------------------------------------------------------------
# Ray Train per-worker training loop (HP search trials + final training).
# ---------------------------------------------------------------------------


def train_func(config: Dict[str, Any], is_final_training: bool = False) -> None:
    """Train one SkipGram model with the given (flat) config.

    Runs inside a Ray Train worker. Initializes MLflow, builds loaders + model,
    fits a Lightning Trainer, logs params/metrics, and — for the final run —
    logs the IDMapper + TorchScript model to the Model Registry and tags the
    best val_loss version as champion.

    Args:
        config: Flat train_loop_config dict (see `main()` for keys).
        is_final_training: Whether this is the final (best-params) run.
    """
    from feature.id_mapper import IDMapper  # imported in-worker for Ray runtime_env

    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Train function config: {config}")

    # Initialize MLflow.
    mlflow_run = None
    try:
        mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
        base_experiment = config["run_name"]
        experiment_name = (
            f"{base_experiment}/final_model"
            if is_final_training
            else f"{base_experiment}/hyperparameter_tuning"
        )

        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to create experiment {experiment_name}: {str(e)}")
            try:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment with ID: {experiment_id}")
            except Exception as e:
                logger.error(f"Failed to create experiment with new ID: {str(e)}")

        mlflow.set_experiment(experiment_name)
        tags = {
            "phase": "final_training" if is_final_training else "tuning",
            "model_type": config["model_type"],
            "dataset_version": config["dataset_version"],
        }
        run_name = "Final Model - SkipGram" if is_final_training else "Hyperparameter Tuning"
        mlflow_run = mlflow.start_run(run_name=run_name, tags=tags)

        params_to_log = {
            f"{'final' if is_final_training else 'tuning'}.{k}": v
            for k, v in config.items()
            if k
            in [
                "max_epochs",
                "batch_size",
                "num_negative_samples",
                "window_size",
                "num_workers",
                "learning_rate",
                "l2_reg",
                "embedding_dim",
            ]
        }
        mlflow.log_params(params_to_log)
        mlflow.log_params(
            {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": (
                    torch.version.cuda if torch.cuda.is_available() else "N/A"
                ),
                "num_cpus": psutil.cpu_count(),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            }
        )
    except Exception as e:
        logger.warning(f"Failed to initialize MLflow: {str(e)}")

    try:
        logger.info(f"GPU available: {torch.cuda.is_available()}")
        data_path = config["data_path"]
        sequences_fp = os.path.join(data_path, config["sequences_file"])
        val_sequences_fp = os.path.join(data_path, config["val_sequences_file"])
        idm_fp = os.path.join(data_path, config["idm_file"])

        idm = IDMapper().load(idm_fp)
        dataset = SkipGramDataset(
            sequences_fp,
            window_size=config["window_size"],
            negative_samples=config["num_negative_samples"],
            id_to_idx=idm.item_to_index,
            ddp=True,
        )
        val_dataset = SkipGramDataset(
            val_sequences_fp,
            interacted=dataset.interacted,
            item_freq=dataset.item_freq,
            window_size=config["window_size"],
            negative_samples=config["num_negative_samples"],
            id_to_idx=idm.item_to_index,
            ddp=True,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            collate_fn=dataset.collate_fn,
            num_workers=config["num_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=False,
            collate_fn=val_dataset.collate_fn,
            num_workers=config["num_workers"],
        )

        model = SkipGram(dataset.vocab_size, config["embedding_dim"])
        if is_final_training:
            trial_suffix = ""  # final training writes directly under final_checkpoint_dir
        else:
            # Lazy: `config.get('trial_id', <default>)` would eagerly evaluate the
            # default (Python evaluates dict.get's default arg before the lookup),
            # and `train.get_context().get_trial_id()` raises outside a Tune
            # session. The manual HP loop always sets `trial_id`, so resolve it
            # lazily and only fall back to the Ray context when it is absent.
            trial_id = config.get("trial_id")
            if trial_id is None:
                trial_id = train.get_context().get_trial_id()
            trial_suffix = f"trial_{trial_id}"
        trial_dir = os.path.join(config["checkpoint_dir"], trial_suffix) if trial_suffix else config["checkpoint_dir"]
        lit_model = LitSkipGram(
            model,
            learning_rate=config["learning_rate"],
            l2_reg=config["l2_reg"],
            log_dir=trial_dir,
        )

        start_time = time.time()
        trainer = L.Trainer(
            max_epochs=config["max_epochs"],
            accelerator=config["accelerator"],
            callbacks=[
                ModelCheckpoint(
                    dirpath=trial_dir,
                    filename=config["checkpoint_filename"],
                    save_top_k=config["checkpoint_save_top_k"],
                    monitor=config["checkpoint_monitor"],
                    mode=config["checkpoint_mode"],
                ),
                EarlyStopping(
                    monitor=config["early_stopping_monitor"],
                    patience=config["early_stopping_patience"],
                    mode=config["early_stopping_mode"],
                    verbose=config["early_stopping_verbose"],
                ),
            ],
            logger=False,
        )

        trainer.fit(lit_model, train_loader, val_loader)
        training_time = time.time() - start_time
        val_loss = trainer.callback_metrics.get(
            "val_loss", torch.tensor(float("inf"))
        ).item()

        if not is_final_training:
            train.report({"val_loss": val_loss})

        if mlflow_run:
            try:
                metrics_to_log = {
                    "val_loss": val_loss,
                    "train_loss": trainer.callback_metrics.get(
                        "train_loss", torch.tensor(float("inf"))
                    ).item(),
                    "training_time_seconds": training_time,
                }
                if torch.cuda.is_available():
                    metrics_to_log["gpu_memory_allocated_mb"] = (
                        torch.cuda.memory_allocated() / (1024**2)
                    )
                    metrics_to_log["gpu_memory_reserved_mb"] = (
                        torch.cuda.memory_reserved() / (1024**2)
                    )
                metrics_to_log["cpu_memory_usage_percent"] = psutil.Process().memory_percent()

                metrics_to_log = {
                    f"{'final' if is_final_training else 'tuning'}.{k}": v
                    for k, v in metrics_to_log.items()
                }
                for metric_name, metric_value in metrics_to_log.items():
                    try:
                        mlflow.log_metric(metric_name, metric_value)
                        logger.info(f"Logged metric {metric_name}: {metric_value}")
                    except Exception as e:
                        logger.warning(f"Failed to log metric {metric_name}: {str(e)}")

                if is_final_training:
                    _log_final_model_to_mlflow(config, lit_model, idm, val_loss)
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {str(e)}")

        logger.info(f"Training completed, val_loss: {val_loss}")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        if mlflow_run:
            try:
                mlflow.end_run()
            except Exception:
                pass


def _log_final_model_to_mlflow(
    config: Dict[str, Any], lit_model: LitSkipGram, idm: IDMapper, val_loss: float
) -> None:
    """Log IDMapper + TorchScript SkipGram to the MLflow Model Registry + tag champion."""
    try:
        idm_temp_path = os.path.join(config["checkpoint_dir"], "id_mapper.json")
        idm.save(idm_temp_path)
        mlflow.log_artifact(idm_temp_path, artifact_path="id_mapper")
        logger.info(f"IDMapper saved and logged as artifact at {idm_temp_path}")
        # Also drop an `idm.json` copy next to the checkpoint so the embedding
        # index space is recoverable from disk without the engineer output dir.
        shutil.copyfile(
            idm_temp_path, os.path.join(config["checkpoint_dir"], "idm.json")
        )

        import numpy as np
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, TensorSpec

        input_schema = Schema(
            [
                TensorSpec(name="target_items", type=np.dtype(np.int64), shape=(-1,)),
                TensorSpec(name="context_items", type=np.dtype(np.int64), shape=(-1,)),
            ]
        )
        output_schema = Schema([TensorSpec(type=np.dtype(np.float32), shape=(-1,))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        model_metadata = {
            "model_type": config["model_type"],
            "task": "item-embedding",
            "framework": "pytorch",
            "description": (
                "SkipGram model for learning item embeddings from interaction sequences.\n"
                "Model Details:\n"
                "- Architecture: Single embedding layer with Xavier uniform initialization\n"
                "- Input: Pairs of target and context item indices\n"
                "- Output: Similarity score between items (0-1)\n"
                "- Training: Negative sampling with frequency-based sampling\n"
                "Typical Use Cases:\n"
                "- Item recommendation\n"
                "- Similar item finding\n"
                "- Interaction sequence analysis\n"
                "Input Format:\n"
                "- target_items: Tensor of item indices (int64)\n"
                "- context_items: Tensor of context item indices (int64)\n"
                "Output Format:\n"
                "- Tensor of similarity scores (float32)\n"
                "Additional Artifacts:\n"
                "- id_mapper/id_mapper.json: Mapping of item IDs to indices"
            ),
            "hyperparameters": {
                "embedding_dim": config["embedding_dim"],
                "learning_rate": config["learning_rate"],
                "l2_reg": config["l2_reg"],
            },
        }

        scripted_model = torch.jit.script(lit_model.skipgram_model)
        mlflow.pytorch.log_model(
            pytorch_model=scripted_model,
            artifact_path="skipgram_model",
            registered_model_name=f"{config['run_name']}_skipgram",
            signature=signature,
            metadata=model_metadata,
        )
        logger.info("TorchScript model logged successfully.")

        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(
            f"{config['run_name']}_skipgram", stages=["None"]
        )
        client.update_model_version(
            name=f"{config['run_name']}_skipgram",
            version=latest_versions[0].version,
            description=model_metadata["description"],
        )
        this_version = latest_versions[0]
        this_val_loss = val_loss
        is_champion = True
        worse_versions = []

        for v in client.search_model_versions(f"name='{config['run_name']}_skipgram'"):
            if v.run_id == this_version.run_id:
                continue
            try:
                run_data = client.get_run(v.run_id).data
                other_val_loss = run_data.metrics.get("final.val_loss") or run_data.metrics.get(
                    "tuning.val_loss"
                )
                if other_val_loss is not None and other_val_loss < this_val_loss:
                    is_champion = False
                    break
                if other_val_loss is not None:
                    worse_versions.append(v)
            except Exception as e:
                logger.warning(f"Cannot load run data for version {v.version}: {str(e)}")

        if is_champion:
            client.set_model_version_tag(
                name=f"{config['run_name']}_skipgram",
                version=this_version.version,
                key="champion",
                value="true",
            )
            logger.info(
                f"Model version {this_version.version} is now CHAMPION (val_loss = {this_val_loss:.4f})"
            )
            for v in worse_versions:
                try:
                    client.delete_model_version_tag(
                        name=f"{config['run_name']}_skipgram",
                        version=v.version,
                        key="champion",
                    )
                    logger.info(f"Removed champion tag from version {v.version}")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove champion tag from version {v.version}: {str(e)}"
                    )
        else:
            logger.info(
                f"Model version {this_version.version} is NOT champion (val_loss = {this_val_loss:.4f})"
            )
    except Exception as e:
        logger.error(f"Error logging model or IDMapper to MLflow: {str(e)}")


# ---------------------------------------------------------------------------
# Opt-in overfit sanity check (no Ray / no MLflow — plain local Lightning).
# ---------------------------------------------------------------------------


def run_overfit(cfg: Dict[str, Any]) -> None:
    """Sanity check: overfit a single batch; warn if val_loss does not drop near zero."""
    o = cfg["overfit"]
    d = cfg["data"]
    t = cfg["training"]
    out = cfg["output"]
    logger.info("=== Overfit sanity check ===")

    sequences_fp = os.path.join(d["engineer_dir"], d["overfit_sequences_file"])
    dataset = SkipGramDataset(
        sequences_fp,
        window_size=t["window_size"],
        negative_samples=t["num_negative_samples"],
        id_to_idx=None,  # self-contained mapping for the overfit batch
    )
    train_loader = DataLoader(
        dataset, batch_size=o["batch_size"], shuffle=False, drop_last=False,
        collate_fn=dataset.collate_fn, num_workers=t["num_workers"],
    )
    val_loader = DataLoader(
        dataset, batch_size=o["batch_size"], shuffle=False, drop_last=False,
        collate_fn=dataset.collate_fn, num_workers=t["num_workers"],
    )

    model = SkipGram(dataset.vocab_size, cfg["model"]["embedding_dim"]["choice"][0])
    lit_model = LitSkipGram(
        model,
        learning_rate=t["learning_rate"]["loguniform"][0],
        l2_reg=t["l2_reg"]["loguniform"][0],
        log_dir=os.path.join(out["checkpoint_dir"], "overfit"),
    )

    checkpoint_dir = os.path.join(out["checkpoint_dir"], "overfit")
    os.makedirs(checkpoint_dir, exist_ok=True)
    tr = cfg["trainer"]
    trainer = L.Trainer(
        max_epochs=o["max_epochs"],
        accelerator=tr["accelerator"],
        callbacks=[
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=tr["checkpoint"]["filename"],
                save_top_k=tr["checkpoint"]["save_top_k"],
                monitor=tr["checkpoint"]["monitor"],
                mode=tr["checkpoint"]["mode"],
            ),
            EarlyStopping(
                monitor=tr["early_stopping"]["monitor"],
                patience=tr["early_stopping"]["patience"],
                mode=tr["early_stopping"]["mode"],
            ),
        ],
        logger=False,
        enable_progress_bar=True,
    )
    trainer.fit(lit_model, train_loader, val_loader)
    val_loss = trainer.callback_metrics.get("val_loss", torch.tensor(float("inf"))).item()

    if val_loss > 0.1:
        logger.warning(
            f"Overfit val_loss={val_loss:.4f} > 0.1 — model did NOT converge on the "
            f"sanity batch. Check data / lr / model before full training."
        )
    else:
        logger.info(f"Overfit check passed (val_loss={val_loss:.4f} <= 0.1).")


# ---------------------------------------------------------------------------
# Main: Ray init → Tune → best trial → final training.
# ---------------------------------------------------------------------------


def _train_loop_config(cfg: Dict[str, Any], checkpoint_dir: str) -> Dict[str, Any]:
    """Build the flat train_loop_config dict consumed by `train_func`."""
    t = cfg["training"]
    tr = cfg["trainer"]
    return {
        "data_path": cfg["data"]["engineer_dir"],
        "sequences_file": cfg["data"]["sequences_file"],
        "val_sequences_file": cfg["data"]["val_sequences_file"],
        "idm_file": cfg["data"]["idm_file"],
        "checkpoint_dir": checkpoint_dir,
        "run_name": cfg["experiment"]["run_name"],
        "dataset_version": cfg["experiment"]["dataset_version"],
        "model_type": cfg["model"]["type"],
        "mlflow_tracking_uri": cfg["mlflow"]["tracking_uri"],
        "max_epochs": t["max_epochs"],
        "batch_size": t["batch_size"],
        "num_negative_samples": t["num_negative_samples"],
        "window_size": t["window_size"],
        "num_workers": t["num_workers"],
        "embedding_dim": cfg["model"]["embedding_dim"]["choice"][0],
        "learning_rate": t["learning_rate"]["loguniform"][0],
        "l2_reg": t["l2_reg"]["loguniform"][0],
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


def _read_best_val_loss(ckpt_path: Path) -> float | None:
    """Read the best monitored val_loss from a Lightning checkpoint's callback state.

    Ray Train v2 does not surface per-trial metrics via `Result.metrics_dataframe`
    when running outside Tune, so the trial's `ModelCheckpoint` callback state
    (the lowest monitored `val_loss`) is the reliable source.

    Args:
        ckpt_path: Path to a Lightning `best-checkpoint.ckpt`.

    Returns:
        The best val_loss as a float, or None if the checkpoint / field is absent.
    """
    if not ckpt_path.exists():
        return None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for cb in ck.get("callbacks", {}).values():
        if isinstance(cb, dict) and "best_model_score" in cb:
            score = cb["best_model_score"]
            return float(score.item()) if hasattr(score, "item") else float(score)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Item2Vec training (Ray Tune + MLflow).")
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "item2vec.yaml"),
        help="Path to the item2vec YAML config.",
    )
    parser.add_argument(
        "--overfit", action="store_true",
        help="Run the single-batch overfit sanity check before Ray Tune.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    for key in ("checkpoint_dir", "final_checkpoint_dir", "log_dir", "storage_path"):
        os.makedirs(cfg["output"][key], exist_ok=True)

    if args.overfit:
        run_overfit(cfg)

    # Ray cluster init (local single-machine or EKS KubeRay via RAY_ADDRESS).
    # `RAY_ADDRESS=local` (or empty) → start an in-process local cluster
    # (ray.init with address=None); a `ray://...` address → Ray Client to EKS.
    mcfg = cfg["mlflow"]
    raw_address = cfg["ray"]["address"].strip()
    ray_address = None if raw_address.lower() in ("", "local") else raw_address

    # Ship the repo root so workers can import `models.item2vec` / `feature`.
    # `feature/output/` is intentionally NOT excluded — the engineer artifacts
    # (jsonl sequences + idm.json, a few MB) must ship to workers via runtime_env
    # so `train_func` can read them at the working_dir-relative `engineer_dir`
    # path. Heavy/local-only dirs are still excluded to keep the upload small.
    ray_env = {
        "working_dir": str(ROOT),
        "excludes": [
            ".venv/", ".git/", "data/", "notebooks/",
            "models/output/", "mlruns/",
            "*.parquet", "*.ckpt", "*.pyc",
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
    is_local = ray_address is None  # local-pro; workers share the driver FS.

    # Ray 2.55's working_dir packaging honors .gitignore, which excludes
    # `feature/output/**/*.{jsonl,json}`; the shipped package therefore lacks
    # the data, so workers can't read the relative `engineer_dir`. Absolutize
    # it against ROOT for local runs — workers on the same host read the host
    # path directly. (EKS path unchanged: data ships via runtime_env to remote
    # workers, and the head pod's working_dir is not filtered by this repo's
    # .gitignore.)
    data_path = (
        str((ROOT / cfg["data"]["engineer_dir"]).resolve())
        if is_local
        else cfg["data"]["engineer_dir"]
    )

    emb_choices = cfg["model"]["embedding_dim"]["choice"]
    lr_range = cfg["training"]["learning_rate"]["loguniform"]
    l2_range = cfg["training"]["l2_reg"]["loguniform"]
    num_samples = cfg["experiment"]["tune_config"]["num_samples"]
    tune_metric = cfg["experiment"]["tune_config"]["metric"]  # "val_loss"
    tune_mode = cfg["experiment"]["tune_config"]["mode"]       # "min"

    rng = np.random.default_rng(seed=42)
    best_val_loss = float("inf")
    best_params: Dict[str, Any] = {}
    trial_results: list[Dict[str, Any]] = []

    logger.info(f"=== Manual HP search: {num_samples} trials (metric={tune_metric}, mode={tune_mode}) ===")
    for i in range(num_samples):
        emb_dim = int(rng.choice(emb_choices))
        lr = float(10 ** rng.uniform(np.log10(lr_range[0]), np.log10(lr_range[1])))
        l2 = float(10 ** rng.uniform(np.log10(l2_range[0]), np.log10(l2_range[1])))
        logger.info(f"--- Trial {i + 1}/{num_samples}: emb_dim={emb_dim} lr={lr:.6f} l2={l2:.6f} ---")

        trial_config = _train_loop_config(cfg, cfg["output"]["checkpoint_dir"])
        trial_config["data_path"] = data_path
        trial_config["trial_id"] = i
        trial_config["embedding_dim"] = emb_dim
        trial_config["learning_rate"] = lr
        trial_config["l2_reg"] = l2

        trial_trainer = TorchTrainer(
            train_loop_per_worker=train_func,
            train_loop_config=trial_config,
            scaling_config=ScalingConfig(num_workers=1, use_gpu=use_gpu),
        )
        try:
            trial_trainer.fit()
        except Exception as e:
            logger.error(f"Trial {i + 1} failed: {str(e)}")
            trial_results.append(
                {"trial": i, "embedding_dim": emb_dim, "learning_rate": lr,
                 "l2_reg": l2, "val_loss": None, "error": str(e)}
            )
            continue

        # Read the trial's best val_loss from its Lightning checkpoint.
        # (Ray Train v2's Result.metrics_dataframe is None without Tune, so
        # the checkpoint's ModelCheckpoint callback state is the source.)
        trial_ckpt = (
            Path(cfg["output"]["checkpoint_dir"])
            / f"trial_{i}"
            / cfg["trainer"]["checkpoint"]["filename"]
        ).with_suffix(".ckpt")
        val_loss = _read_best_val_loss(trial_ckpt)
        logger.info(f"Trial {i + 1} val_loss={val_loss}")
        trial_results.append(
            {"trial": i, "embedding_dim": emb_dim, "learning_rate": lr,
             "l2_reg": l2, "val_loss": val_loss}
        )
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {"embedding_dim": emb_dim, "learning_rate": lr, "l2_reg": l2}

    # Persist the HP search results for the training report.
    reports_dir = ROOT / "models" / "output" / "item2vec" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "hp_search_results.json", "w") as f:
        json.dump(
            {"metric": tune_metric, "mode": tune_mode, "trials": trial_results,
             "best_val_loss": best_val_loss, "best_params": best_params},
            f, indent=2,
        )
    logger.info(f"HP search done. best_val_loss={best_val_loss} best_params={best_params}")

    if not best_params:
        raise RuntimeError("All HP trials failed — cannot run final training.")

    logger.info("Starting final training with best parameters...")
    final_config = _train_loop_config(cfg, cfg["output"]["final_checkpoint_dir"])
    final_config["data_path"] = data_path
    final_config.update(best_params)

    final_trainer = TorchTrainer(
        train_loop_per_worker=lambda config: train_func(config, is_final_training=True),
        train_loop_config=final_config,
        scaling_config=ScalingConfig(num_workers=1, use_gpu=use_gpu),
    )
    final_trainer.fit()
    logger.info("Final training completed!")
    logger.info(f"Final model checkpoint saved at: {cfg['output']['final_checkpoint_dir']}")


if __name__ == "__main__":
    main()