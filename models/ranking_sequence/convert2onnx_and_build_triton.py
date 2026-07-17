"""Export the trained GRU `Ranker` to ONNX and build the Triton model repository.

Ported back from the reference project's
`src/model_ranking_sequence/convert2onnx_and_build_triton.py`, adapted to this
port's artifacts:

  - The ranker is a Lightning checkpoint
    (`models/output/ranking_sequence/final_model/best-checkpoint.ckpt`), not an
    MLflow-registered model first. We try the MLflow Model Registry champion
    first (mirroring the reference), then fall back to the disk checkpoint so
    the export still runs on local-pro with `docker compose up -d` MLflow down.
  - The `IDMapper` is saved as `id_mapper.json` (JSON, MovieLens int ids), not
    `id_mapper.pkl` (dill, Amazon string ASINs). The Triton Python backend loads
    the JSON directly and parses string tensors back to int — no dependency on
    `feature.id_mapper` inside the Triton container.
  - The item-metadata pipeline is fitted on MovieLens columns
    (`title`, `genres`, `movie_rating_cnt_*`, `movie_rating_avg_prev_rating_*`),
    not the reference's `parent_asin_*`. The `item_pipeline` Triton backend reads
    those fields; the ensemble config wires the matching input names.

Run:
    uv run python -m models.ranking_sequence.convert2onnx_and_build_triton

Produces `models/ranking_sequence/model_repository/` with the same four-model
layout as the reference: `ranker` (ONNX), `id_mapper` (Python), `item_pipeline`
(Python), `ensemble`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import torch
import torch.nn as nn

import dill
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

try:
    import onnx
    from onnx import checker
except ImportError as exc:  # pragma: no cover - explicit, like the reference
    raise ImportError(
        "Please install the 'onnx' package to run this script: uv add onnx"
    ) from exc

# Make the repo root importable for `models.ranking_sequence.model` (the
# in-process MLflow-pytorch loader imports the model class by qualified name).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Load .env (same as train.py) BEFORE resolving MLflow creds so the
# `${MLFLOW_*}` defaults point at the docker-compose MLflow + MinIO.
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — env-driven (same `${VAR}` resolution idea as train.py) so local-pro
# (docker-compose MLflow + MinIO) and AWS real (in-cluster MLflow) share one
# path. Defaults point at the disk artifacts so the export runs even with
# MLflow down.
# ---------------------------------------------------------------------------
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_S3_ENDPOINT_URL = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
# MLflow artifact creds come from env (loaded from `.env` via train.py's
# dotenv path; set MLFLOW_AWS_ACCESS_KEY_ID / MLFLOW_AWS_SECRET_ACCESS_KEY
# there for local-pro MinIO). No hardcoded defaults — the disk fallback path
# does not need them, and a missing MLflow fails the fast probe above.
MLFLOW_AWS_ACCESS_KEY_ID = os.environ.get("MLFLOW_AWS_ACCESS_KEY_ID", "")
MLFLOW_AWS_SECRET_ACCESS_KEY = os.environ.get("MLFLOW_AWS_SECRET_ACCESS_KEY", "")
MLFLOW_S3_IGNORE_TLS = os.environ.get("MLFLOW_S3_IGNORE_TLS", "true")

# Make boto3 / MLflow's artifact client talk to MinIO when local-pro.
os.environ["AWS_ACCESS_KEY_ID"] = MLFLOW_AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = MLFLOW_AWS_SECRET_ACCESS_KEY
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", MLFLOW_S3_ENDPOINT_URL)
os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", MLFLOW_S3_IGNORE_TLS)

MODEL_NAME = os.getenv("MODEL_NAME", "ranking_sequence_rating")
CHAMPION_TAG_KEY = os.getenv("CHAMPION_TAG_KEY", "champion")

FINAL_MODEL_DIR = ROOT / "models" / "output" / "ranking_sequence" / "final_model"
TRITON_REPO = str(ROOT / "models" / "ranking_sequence" / "model_repository")

# Export constants — match the training config exactly.
SEQ_LEN = 10
FEATURE_SIZE: int | None = None  # set dynamically from the item pipeline
ONNX_OPSET = 13

# Triton input dim hints — -1 = variable-length string tensors.
MAX_USER_ID_LEN = -1
MAX_ITEM_ID_LEN = 1
MAX_CATEGORIES_LEN = -1  # free-text genres, variable length

# MovieLens columns the item_metadata_pipeline was fitted on (must match
# configs/ranking_sequence.yaml `dataset.required_columns`).
ITEM_PIPELINE_COLUMNS = [
    "title",
    "genres",
    "movie_rating_cnt_90d",
    "movie_rating_avg_prev_rating_90d",
    "movie_rating_cnt_30d",
    "movie_rating_avg_prev_rating_30d",
    "movie_rating_cnt_7d",
    "movie_rating_avg_prev_rating_7d",
]

# Dynamic axes — identical to the reference (batch + seq_len vary at serve time).
DYNAMIC_AXES = {
    "user_ids": {0: "batch_size"},
    "input_seq": {0: "batch_size", 1: "seq_len"},
    "input_seq_ts_bucket": {0: "batch_size", 1: "seq_len"},
    "item_features": {0: "batch_size"},
    "target_item": {0: "batch_size"},
    "output": {0: "batch_size"},
}


class TritonModelWrapper(nn.Module):
    """Wraps the Ranker for Triton/ONNX, squeezing the [batch, 1] -> [batch] dims
    the ensemble's `id_mapper` emits for `user_ids` / `target_item`.

    The Ranker's `forward` already takes [batch] user_ids / target_item, so
    this matches the reference wrapper 1:1.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        user_ids: torch.Tensor,
        input_seq: torch.Tensor,
        input_seq_ts_bucket: torch.Tensor,
        item_features: torch.Tensor,
        target_item: torch.Tensor,
    ) -> torch.Tensor:
        user_ids = user_ids.squeeze(1)
        target_item = target_item.squeeze(1)
        return self.model(
            user_ids, input_seq, input_seq_ts_bucket, item_features, target_item
        )


# ---------------------------------------------------------------------------
# Model + artifact loading.
# ---------------------------------------------------------------------------


def _mlflow_reachable(timeout: float = 2.0) -> bool:
    """Fast TCP probe so a down local-pro MLflow fails over to disk in seconds,
    instead of burning ~4 min on MLflow's built-in 7-retry backoff."""
    from urllib.parse import urlparse

    try:
        parsed = urlparse(MLFLOW_URI)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5000
    except Exception:
        return False
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_champion_version(model_name: str) -> str | None:
    """Return the version tagged `champion=true` for `model_name`, else latest.

    Returns None when the model is not registered (so the caller can fall back
    to the disk checkpoint) instead of raising — unlike the reference, which
    raises because it assumes MLflow always has the model.
    """
    if not _mlflow_reachable():
        logger.warning("MLflow unreachable at %s; will use disk checkpoint.", MLFLOW_URI)
        return None
    try:
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        versions = client.search_model_versions(f"name = '{model_name}'")
    except Exception as exc:  # MLflow down / unreachable — caller falls back.
        logger.warning("MLflow search failed (%s); will use disk checkpoint.", exc)
        return None
    if not versions:
        return None
    for v in versions:
        if str(v.tags.get(CHAMPION_TAG_KEY, "")).lower() == "true":
            logger.info("Found champion version: %s", v.version)
            return v.version
    latest = max(versions, key=lambda v: int(v.version))
    logger.info("No champion tag found, using latest version: %s", latest.version)
    return latest.version


def _load_ranker_from_checkpoint(ckpt_path: Path) -> nn.Module:
    """Rebuild a `Ranker` from a Lightning checkpoint on disk (MLflow fallback).

    All Ranker dims are inferred from the checkpoint `state_dict` so the export
    does not need the item2vec champion or the config to reconstruct the model
    (the frozen item2vec weights are already in `model.item_embedding.weight`).
    """
    from models.ranking_sequence.model import Ranker

    logger.info("Loading ranker from disk checkpoint: %s", ckpt_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("state_dict", ck)
    # Strip the Lightning `model.` prefix so the dict loads into a bare Ranker.
    ranker_state = {
        k[len("model."):]: v for k, v in state.items() if k.startswith("model.")
    }
    if not ranker_state:
        raise ValueError(f"No `model.*` keys in checkpoint {ckpt_path}")

    item_emb_w = ranker_state["item_embedding.weight"]          # [num_items+1, d]
    user_emb_w = ranker_state["user_embedding.weight"]           # [num_users, d]
    ts_emb_w = ranker_state["item_sequence_ts_bucket_embedding.weight"]
    feat_w = ranker_state["item_feature_tower.0.weight"]         # [d, feat_size]

    num_items = item_emb_w.shape[0] - 1       # +1 padding row
    emb_dim = item_emb_w.shape[1]
    num_users = user_emb_w.shape[0]
    item_sequence_ts_bucket_size = ts_emb_w.shape[0] - 1
    bucket_embedding_dim = ts_emb_w.shape[1]
    item_feature_size = feat_w.shape[1]

    item_embedding = nn.Embedding(num_items + 1, emb_dim, padding_idx=num_items)
    item_embedding.weight.data.copy_(item_emb_w)

    model = Ranker(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=emb_dim,
        item_sequence_ts_bucket_size=item_sequence_ts_bucket_size,
        bucket_embedding_dim=bucket_embedding_dim,
        item_feature_size=item_feature_size,
        item_embedding=item_embedding,
    )
    missing, unexpected = model.load_state_dict(ranker_state, strict=False)
    if missing or unexpected:
        logger.warning("state_dict mismatch — missing=%s unexpected=%s", missing, unexpected)
    model.eval()
    logger.info(
        "Ranker rebuilt: num_users=%d num_items=%d emb_dim=%d feat_size=%d",
        num_users, num_items, emb_dim, item_feature_size,
    )
    return model


def _load_idm_from_disk(idm_path: Path) -> dict:
    """Load the IDMapper JSON as raw dicts for the Triton backend (no `feature`
    package dependency inside the container)."""
    logger.info("Loading IDMapper JSON from: %s", idm_path)
    with open(idm_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pipeline_from_disk(pipe_path: Path):
    """Load the item_metadata_pipeline via dill (same as the reference)."""
    logger.info("Loading item_metadata_pipeline from: %s", pipe_path)
    with open(pipe_path, "rb") as f:
        return dill.load(f)


def validate_pipeline_feature_size(pipeline) -> int:
    """Compute the item-feature width the pipeline emits on a dummy MovieLens row.

    Unlike the reference (hardcoded expected 416 for Amazon ASIN features), the
    MovieLens port derives the size dynamically because TF-IDF title vocabulary
    size depends on the fitted corpus.
    """
    logger.info("Validating item metadata pipeline feature size...")
    dummy = pd.DataFrame(
        {
            "title": ["Toy Story (1995)"],
            "genres": ["Animation|Children|Comedy|Adventure"],
            "movie_rating_cnt_90d": [10],
            "movie_rating_avg_prev_rating_90d": [4.1],
            "movie_rating_cnt_30d": [5],
            "movie_rating_avg_prev_rating_30d": [3.9],
            "movie_rating_cnt_7d": [2],
            "movie_rating_avg_prev_rating_7d": [4.0],
        }
    )
    features = pipeline.transform(dummy)
    feature_size = features.shape[1]
    logger.info("Item metadata pipeline feature size: %d", feature_size)
    return feature_size


def load_model_and_artifacts(model_name: str):
    """Load the Ranker + IDMapper + item pipeline.

    Primary path mirrors the reference: MLflow champion ranker + its run's
    `id_mapper` / `item_metadata_pipeline` artifacts. Fallback (local-pro,
    MLflow down): the disk checkpoint + sibling JSON/dill under `final_model/`.
    """
    global FEATURE_SIZE
    # --- Try MLflow champion first (reference path). ----------------------
    version = find_champion_version(model_name)
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
        logger.info("Loading model from MLflow: %s", model_uri)
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            model = mlflow.pytorch.load_model(model_uri)
            model.eval()
            logger.info("Model loaded from MLflow successfully.")

            client = MlflowClient(tracking_uri=MLFLOW_URI)
            model_version = client.get_model_version(model_name, version)
            run_id = model_version.run_id

            idm_local = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="id_mapper/id_mapper.json"
                )
            )
            pipe_local = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path="item_metadata_pipeline/item_metadata_pipeline.dill",
                )
            )
            idm_data = json.load(open(idm_local, "r", encoding="utf-8"))
            item_pipeline = dill.load(open(pipe_local, "rb"))

            FEATURE_SIZE = validate_pipeline_feature_size(item_pipeline)
            return model, idm_data, item_pipeline
        except Exception as exc:
            logger.warning("MLflow load path failed (%s); falling back to disk.", exc)

    # --- Disk fallback (local-pro). --------------------------------------
    ckpt_path = FINAL_MODEL_DIR / "best-checkpoint.ckpt"
    idm_path = FINAL_MODEL_DIR / "id_mapper.json"
    pipe_path = FINAL_MODEL_DIR / "item_metadata_pipeline.dill"
    for p in (ckpt_path, idm_path, pipe_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Neither MLflow champion nor disk artifact found: {p}"
            )

    model = _load_ranker_from_checkpoint(ckpt_path)
    idm_data = _load_idm_from_disk(idm_path)
    item_pipeline = _load_pipeline_from_disk(pipe_path)

    FEATURE_SIZE = validate_pipeline_feature_size(item_pipeline)
    return model, idm_data, item_pipeline


def print_model_info(model: nn.Module) -> None:
    """Log a summary of the model's architecture and parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("===== Model Summary =====")
    logger.info("Total parameters: %s", f"{total:,}")
    logger.info("Trainable parameters: %s", f"{trainable:,}")


# ---------------------------------------------------------------------------
# ONNX export.
# ---------------------------------------------------------------------------


def export_to_onnx(model: nn.Module, seq_len: int, feature_size: int) -> str:
    """Export the (eval-mode) Ranker to ONNX and validate it with `onnx.checker`.

    Dummy inputs mirror the reference: int64 id tensors, an int64 ts-bucket
    tensor, and an fp32 item-features tensor. `dynamic_axes` lets batch_size
    and seq_len vary at serve time.
    """
    logger.info("Exporting model to ONNX format...")
    wrapper = TritonModelWrapper(model)
    wrapper.eval()
    dummy_inputs = (
        torch.randint(0, 10, (1, 1), dtype=torch.int64),         # user_ids
        torch.randint(0, 10, (1, seq_len), dtype=torch.int64),    # input_seq
        torch.randint(0, seq_len + 1, (1, seq_len), dtype=torch.int64),  # ts bucket
        torch.randn(1, feature_size, dtype=torch.float32),       # item_features
        torch.randint(0, 10, (1, 1), dtype=torch.int64),          # target_item
    )
    onnx_path = os.path.join(TRITON_REPO, "ranker", "1", "model.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            onnx_path,
            export_params=True,
            opset_version=ONNX_OPSET,
            do_constant_folding=True,
            input_names=[
                "user_ids",
                "input_seq",
                "input_seq_ts_bucket",
                "item_features",
                "target_item",
            ],
            output_names=["output"],
            dynamic_axes=DYNAMIC_AXES,
            verbose=False,
            # torch 2.13 defaults to the TorchScript-based dynamo exporter,
            # which needs `onnxscript`; the reference used the classic
            # (TorchScript) exporter, so force it here for parity.
            dynamo=False,
        )
        logger.info("ONNX model exported: %s", onnx_path)
        onnx_model = onnx.load(onnx_path)
        checker.check_model(onnx_model)
        logger.info("ONNX model validation passed")
        return onnx_path
    except Exception as exc:
        logger.error("ONNX export/check failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Triton model repository build.
# ---------------------------------------------------------------------------

# Python-backend script for `id_mapper`: loads id_mapper.json, maps the string
# tensors the gateway sends (MovieLens int ids as strings) to indices, and
# pads/truncates the input sequence to SEQ_LEN (matches the reference's
# convert_to_idx logic, but parses int instead of str ASIN).
_ID_MAPPER_SCRIPT = '''
import json
import logging
import os
import numpy as np
import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 10
PADDING_VALUE = -1


class TritonPythonModel:
    """Triton Python backend mapping MovieLens string ids to dense indices."""

    def __init__(self):
        self.user_to_index = {}
        self.item_to_index = {}
        self.unknown_user_index = -1
        self.unknown_item_index = -1

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(model_dir, "id_mapper.json")
        logger.info(f"Loading IDMapper from: {pkl_path}")
        with open(pkl_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # JSON keys are strings; rebuild with int keys so lookups match int ids.
        self.user_to_index = {int(k): int(v) for k, v in data["user_to_index"].items()}
        self.item_to_index = {int(k): int(v) for k, v in data["item_to_index"].items()}
        self.unknown_user_index = len(self.user_to_index)
        self.unknown_item_index = len(self.item_to_index)
        logger.info("IDMapper initialized: users=%d items=%d",
                    len(self.user_to_index), len(self.item_to_index))

    def _to_index(self, item_id):
        try:
            return self.item_to_index.get(int(item_id), self.unknown_item_index)
        except (TypeError, ValueError):
            return self.unknown_item_index

    def _convert_seq(self, sequence, sequence_length, padding_value):
        if sequence is None or len(sequence) == 0:
            return np.array([padding_value] * sequence_length, dtype=np.int64)
        indices = []
        for item in sequence:
            s = item.decode("utf-8") if isinstance(item, bytes) else str(item)
            if s == "-1" or s == "":
                indices.append(padding_value)
            else:
                try:
                    indices.append(self.item_to_index.get(int(s), self.unknown_item_index))
                except (TypeError, ValueError):
                    indices.append(padding_value)
        pad_needed = sequence_length - len(indices)
        if pad_needed > 0:
            indices = [padding_value] * pad_needed + indices
        elif pad_needed < 0:
            indices = indices[-sequence_length:]
        return np.array(indices, dtype=np.int64)

    def execute(self, requests):
        responses = []
        for request in requests:
            user_ids = pb_utils.get_input_tensor_by_name(request, "user_ids").as_numpy()
            target_items = pb_utils.get_input_tensor_by_name(request, "target_items").as_numpy()
            input_seq = pb_utils.get_input_tensor_by_name(request, "input_seq").as_numpy()

            batch_size = user_ids.shape[0]
            logger.info("Batch size: %d", batch_size)

            user_indices = np.array([
                self.user_to_index.get(
                    int(uid[0].decode("utf-8")) if isinstance(uid[0], bytes) else int(uid[0]),
                    self.unknown_user_index,
                )
                for uid in user_ids
            ], dtype=np.int64).reshape(batch_size, 1)

            target_indices = np.array([
                self._to_index(item[0].decode("utf-8") if isinstance(item[0], bytes) else item[0])
                for item in target_items
            ], dtype=np.int64).reshape(batch_size, 1)

            seq_indices = np.array([
                self._convert_seq(seq, SEQUENCE_LENGTH, PADDING_VALUE)
                for seq in input_seq
            ], dtype=np.int64)

            logger.info("user_indices shape: %s", user_indices.shape)
            logger.info("seq_indices shape: %s", seq_indices.shape)
            logger.info("target_indices shape: %s", target_indices.shape)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("user_indices", user_indices),
                pb_utils.Tensor("seq_indices", seq_indices),
                pb_utils.Tensor("target_indices", target_indices),
            ]))
        return responses

    def finalize(self):
        logger.info("IDMapper finalized")
'''

# Python-backend script for `item_pipeline`: assembles a MovieLens DataFrame
# from the string/numeric tensors the gateway sends and runs the fitted
# item_metadata_pipeline (TF-IDF title + count genres + scaled rating aggregates).
_ITEM_PIPELINE_SCRIPT = '''
import dill
import logging
import os
import numpy as np
import pandas as pd
import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _first_str(arr):
    """Decode the first element of a Triton string tensor column to str."""
    v = arr.reshape(-1)[0]
    return v.decode("utf-8") if isinstance(v, bytes) else str(v)


def _first_scalar(arr, dtype):
    return arr.reshape(-1)[0].astype(dtype)


class TritonPythonModel:
    """Triton Python backend running the item_metadata_pipeline transform."""

    def __init__(self):
        self.pipeline = None

    def initialize(self, args):
        model_dir = os.path.dirname(__file__)
        pkl_path = os.path.join(model_dir, "item_metadata_pipeline.dill")
        logger.info(f"Loading item pipeline from: {pkl_path}")
        with open(pkl_path, "rb") as f:
            self.pipeline = dill.load(f)
        logger.info("Item pipeline initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            titles = pb_utils.get_input_tensor_by_name(request, "titles").as_numpy()
            genres = pb_utils.get_input_tensor_by_name(request, "genres").as_numpy()
            cnt_90d = pb_utils.get_input_tensor_by_name(request, "movie_rating_cnt_90d").as_numpy()
            avg_90d = pb_utils.get_input_tensor_by_name(request, "movie_rating_avg_prev_rating_90d").as_numpy()
            cnt_30d = pb_utils.get_input_tensor_by_name(request, "movie_rating_cnt_30d").as_numpy()
            avg_30d = pb_utils.get_input_tensor_by_name(request, "movie_rating_avg_prev_rating_30d").as_numpy()
            cnt_7d = pb_utils.get_input_tensor_by_name(request, "movie_rating_cnt_7d").as_numpy()
            avg_7d = pb_utils.get_input_tensor_by_name(request, "movie_rating_avg_prev_rating_7d").as_numpy()

            batch_size = titles.shape[0]
            logger.info("Batch size: %d", batch_size)

            df = pd.DataFrame({
                "title": [_first_str(titles[i]) for i in range(batch_size)],
                "genres": [_first_str(genres[i]) for i in range(batch_size)],
                "movie_rating_cnt_90d": [_first_scalar(cnt_90d[i], np.int64) for i in range(batch_size)],
                "movie_rating_avg_prev_rating_90d": [_first_scalar(avg_90d[i], np.float32) for i in range(batch_size)],
                "movie_rating_cnt_30d": [_first_scalar(cnt_30d[i], np.int64) for i in range(batch_size)],
                "movie_rating_avg_prev_rating_30d": [_first_scalar(avg_30d[i], np.float32) for i in range(batch_size)],
                "movie_rating_cnt_7d": [_first_scalar(cnt_7d[i], np.int64) for i in range(batch_size)],
                "movie_rating_avg_prev_rating_7d": [_first_scalar(avg_7d[i], np.float32) for i in range(batch_size)],
            })
            df.fillna(0.0, inplace=True)
            features = self.pipeline.transform(df).astype(np.float32)
            logger.info("item_features shape: %s", features.shape)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("item_features", features)
            ]))
        return responses

    def finalize(self):
        logger.info("Item pipeline finalized")
'''


def _write_id_mapper(repo_path: str, idm_data: dict) -> None:
    """Write the id_mapper Python backend (script + JSON + config)."""
    id_mapper_dir = os.path.join(repo_path, "id_mapper")
    version_dir = os.path.join(id_mapper_dir, "1")
    os.makedirs(version_dir, exist_ok=True)

    with open(os.path.join(version_dir, "model.py"), "w", encoding="utf-8") as f:
        f.write(_ID_MAPPER_SCRIPT)
    with open(os.path.join(version_dir, "id_mapper.json"), "w", encoding="utf-8") as f:
        json.dump(idm_data, f)

    id_mapper_config = f"""
name: "id_mapper"
backend: "python"
max_batch_size: 512
input [
  {{ name: "user_ids", data_type: TYPE_STRING, dims: [{MAX_USER_ID_LEN}] }},
  {{ name: "input_seq", data_type: TYPE_STRING, dims: [-1] }},
  {{ name: "target_items", data_type: TYPE_STRING, dims: [{MAX_ITEM_ID_LEN}] }}
]
output [
  {{ name: "user_indices", data_type: TYPE_INT64, dims: [1] }},
  {{ name: "seq_indices", data_type: TYPE_INT64, dims: [{SEQ_LEN}] }},
  {{ name: "target_indices", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }}
]
instance_group [ {{ kind: KIND_CPU, count: 4 }} ]
"""
    with open(os.path.join(id_mapper_dir, "config.pbtxt"), "w", encoding="utf-8") as f:
        f.write(id_mapper_config)
    logger.info("id_mapper backend created at %s", id_mapper_dir)


def _write_item_pipeline(repo_path: str, item_pipeline) -> None:
    """Write the item_pipeline Python backend (script + dill + config)."""
    item_pipeline_dir = os.path.join(repo_path, "item_pipeline")
    version_dir = os.path.join(item_pipeline_dir, "1")
    os.makedirs(version_dir, exist_ok=True)

    with open(os.path.join(version_dir, "model.py"), "w", encoding="utf-8") as f:
        f.write(_ITEM_PIPELINE_SCRIPT)
    with open(os.path.join(version_dir, "item_metadata_pipeline.dill"), "wb") as f:
        dill.dump(item_pipeline, f)

    item_pipeline_config = f"""
name: "item_pipeline"
backend: "python"
max_batch_size: 512
input [
  {{ name: "titles", data_type: TYPE_STRING, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "genres", data_type: TYPE_STRING, dims: [{MAX_CATEGORIES_LEN}] }},
  {{ name: "movie_rating_cnt_90d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_avg_prev_rating_90d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_cnt_30d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_avg_prev_rating_30d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_cnt_7d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_avg_prev_rating_7d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }}
]
output [
  {{ name: "item_features", data_type: TYPE_FP32, dims: [{FEATURE_SIZE}] }}
]
instance_group [ {{ kind: KIND_CPU, count: 4 }} ]
"""
    with open(os.path.join(item_pipeline_dir, "config.pbtxt"), "w", encoding="utf-8") as f:
        f.write(item_pipeline_config)
    logger.info("item_pipeline backend created at %s", item_pipeline_dir)


def _write_ranker_config(repo_path: str) -> None:
    """Write the ranker (ONNX) config.pbtxt."""
    ranker_config = f"""
name: "ranker"
platform: "onnxruntime_onnx"
max_batch_size: 512
input [
  {{ name: "user_ids", data_type: TYPE_INT64, dims: [1] }},
  {{ name: "input_seq", data_type: TYPE_INT64, dims: [{SEQ_LEN}] }},
  {{ name: "input_seq_ts_bucket", data_type: TYPE_INT64, dims: [{SEQ_LEN}] }},
  {{ name: "item_features", data_type: TYPE_FP32, dims: [{FEATURE_SIZE}] }},
  {{ name: "target_item", data_type: TYPE_INT64, dims: [1] }}
]
output [
  {{ name: "output", data_type: TYPE_FP32, dims: [1] }}
]
instance_group [ {{ kind: KIND_CPU, count: 4 }} ]
"""
    with open(os.path.join(repo_path, "ranker", "config.pbtxt"), "w", encoding="utf-8") as f:
        f.write(ranker_config)
    logger.info("ranker config created at %s/ranker", repo_path)


def _write_ensemble(repo_path: str) -> None:
    """Write the ensemble config wiring id_mapper + item_pipeline + ranker."""
    ensemble_dir = os.path.join(repo_path, "ensemble")
    os.makedirs(ensemble_dir, exist_ok=True)
    ensemble_config = f"""
name: "ensemble"
platform: "ensemble"
max_batch_size: 512
input [
  {{ name: "user_ids", data_type: TYPE_STRING, dims: [{MAX_USER_ID_LEN}] }},
  {{ name: "input_seq", data_type: TYPE_STRING, dims: [-1] }},
  {{ name: "input_seq_ts_bucket", data_type: TYPE_INT64, dims: [{SEQ_LEN}] }},
  {{ name: "target_items", data_type: TYPE_STRING, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "titles", data_type: TYPE_STRING, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "genres", data_type: TYPE_STRING, dims: [{MAX_CATEGORIES_LEN}] }},
  {{ name: "movie_rating_cnt_90d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_avg_prev_rating_90d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_cnt_30d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_avg_prev_rating_30d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_cnt_7d", data_type: TYPE_INT64, dims: [{MAX_ITEM_ID_LEN}] }},
  {{ name: "movie_rating_avg_prev_rating_7d", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }}
]
output [
  {{ name: "output", data_type: TYPE_FP32, dims: [{MAX_ITEM_ID_LEN}] }}
]
ensemble_scheduling {{
  step [
    {{
      model_name: "id_mapper"
      model_version: -1
      input_map {{ key: "user_ids" value: "user_ids" }}
      input_map {{ key: "input_seq" value: "input_seq" }}
      input_map {{ key: "target_items" value: "target_items" }}
      output_map {{ key: "user_indices" value: "user_indices" }}
      output_map {{ key: "seq_indices" value: "seq_indices" }}
      output_map {{ key: "target_indices" value: "target_indices" }}
    }},
    {{
      model_name: "item_pipeline"
      model_version: -1
      input_map {{ key: "titles" value: "titles" }}
      input_map {{ key: "genres" value: "genres" }}
      input_map {{ key: "movie_rating_cnt_90d" value: "movie_rating_cnt_90d" }}
      input_map {{ key: "movie_rating_avg_prev_rating_90d" value: "movie_rating_avg_prev_rating_90d" }}
      input_map {{ key: "movie_rating_cnt_30d" value: "movie_rating_cnt_30d" }}
      input_map {{ key: "movie_rating_avg_prev_rating_30d" value: "movie_rating_avg_prev_rating_30d" }}
      input_map {{ key: "movie_rating_cnt_7d" value: "movie_rating_cnt_7d" }}
      input_map {{ key: "movie_rating_avg_prev_rating_7d" value: "movie_rating_avg_prev_rating_7d" }}
      output_map {{ key: "item_features" value: "item_features" }}
    }},
    {{
      model_name: "ranker"
      model_version: -1
      input_map {{ key: "user_ids" value: "user_indices" }}
      input_map {{ key: "input_seq" value: "seq_indices" }}
      input_map {{ key: "input_seq_ts_bucket" value: "input_seq_ts_bucket" }}
      input_map {{ key: "item_features" value: "item_features" }}
      input_map {{ key: "target_item" value: "target_indices" }}
      output_map {{ key: "output" value: "output" }}
    }}
  ]
}}
"""
    with open(os.path.join(ensemble_dir, "config.pbtxt"), "w", encoding="utf-8") as f:
        f.write(ensemble_config)
    logger.info("ensemble config created at %s", ensemble_dir)


def prepare_triton_repo(onnx_path: str, model_name: str, repo_path: str,
                         idm_data: dict, item_pipeline) -> None:
    """Prepare the full Triton model repository (4 models)."""
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file does not exist: {onnx_path}")

    # Ensure the four model directories exist.
    for name in ("ranker", "id_mapper", "item_pipeline", "ensemble"):
        os.makedirs(os.path.join(repo_path, name, "1"), exist_ok=True)

    _write_ranker_config(repo_path)
    _write_id_mapper(repo_path, idm_data)
    _write_item_pipeline(repo_path, item_pipeline)
    _write_ensemble(repo_path)
    logger.info("Triton ensemble ready at: %s", os.path.abspath(repo_path))


# ---------------------------------------------------------------------------
# Smoke check: load the exported ONNX in ONNX Runtime and run one forward pass
# to confirm the exported graph matches the PyTorch model numerically.
# ---------------------------------------------------------------------------


def _onnx_numerical_check(onnx_path: str, model: nn.Module, seq_len: int,
                          feature_size: int) -> None:
    """Verify the ONNX graph reproduces PyTorch outputs within 1e-5."""
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime not installed; skipping numerical check.")
        return
    logger.info("Running ONNX Runtime numerical check...")
    rng = np.random.default_rng(0)
    user_ids = rng.integers(0, 10, (4, 1), dtype=np.int64)
    input_seq = rng.integers(0, 10, (4, seq_len), dtype=np.int64)
    ts_bucket = rng.integers(0, seq_len + 1, (4, seq_len), dtype=np.int64)
    item_features = rng.standard_normal((4, feature_size)).astype(np.float32)
    target_item = rng.integers(0, 10, (4, 1), dtype=np.int64)

    with torch.no_grad():
        pt_out = (
            TritonModelWrapper(model)(
                torch.from_numpy(user_ids),
                torch.from_numpy(input_seq),
                torch.from_numpy(ts_bucket),
                torch.from_numpy(item_features),
                torch.from_numpy(target_item),
            )
            .numpy()
        )

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(
        ["output"],
        {
            "user_ids": user_ids,
            "input_seq": input_seq,
            "input_seq_ts_bucket": ts_bucket,
            "item_features": item_features,
            "target_item": target_item,
        },
    )[0]
    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    logger.info("max |torch - onnx| = %.2e", max_diff)
    if max_diff > 1e-5:
        raise RuntimeError(f"ONNX numerical mismatch: max_diff={max_diff:.2e}")
    logger.info("ONNX numerical check passed (<=1e-5).")


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    model, idm_data, item_pipeline = load_model_and_artifacts(MODEL_NAME)
    print_model_info(model)
    onnx_path = export_to_onnx(model, SEQ_LEN, FEATURE_SIZE)  # type: ignore[arg-type]
    prepare_triton_repo(onnx_path, MODEL_NAME, TRITON_REPO, idm_data, item_pipeline)
    _onnx_numerical_check(onnx_path, model, SEQ_LEN, FEATURE_SIZE)  # type: ignore[arg-type]
    logger.info("Done. Triton repo: %s", os.path.abspath(TRITON_REPO))


if __name__ == "__main__":
    main()