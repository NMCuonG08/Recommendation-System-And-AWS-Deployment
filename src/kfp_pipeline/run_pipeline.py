"""Kubeflow Pipelines (KFP) end-to-end training pipeline.

Ported from the reference ``src/kfp_pipeline/run_pipeline.py``, adapted to this
port (MovieLens, uv, notebooks via papermill + module entrypoints, in-cluster
MinIO + MLflow + Ray instead of AWS S3/RDS).

Pipeline (7 ops, share a ``data-pvc``):
  1. feature_etl             — papermill feature/etl/003-feature-etl.ipynb
  2. feature_engineer        — papermill feature/engineer/004-features.ipynb
  3. negative_sampling       — papermill feature/engineer/005-negative-sample.ipynb
  4. prep_item2vec           — papermill feature/engineer/006-prep-item2vec.ipynb
  5. train_item2vec          — uv run python -m models.item2vec.train
  6. train_ranking_sequence  — uv run python -m models.ranking_sequence.train
  7. convert_onnx            — uv run python -m models.ranking_sequence.convert2onnx_and_build_triton

Deps chain::

  etl -> engineer -> {negative_sampling, prep_item2vec -> train_item2vec}
       negative_sampling + train_item2vec -> train_ranking_sequence -> convert_onnx

NOTE on inlining: kfp ``func_to_container_op`` serializes each op function's
SOURCE and runs it in the container — module-level names referenced inside the
body are NOT available. So every op body is fully self-contained (imports +
logic inline), exactly like the reference.

Secrets/env (in-cluster, not AWS):
  minio-creds -> AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (MinIO admin/Password1234)
  MLFLOW_TRACKING_URI / MLFLOW_S3_ENDPOINT_URL -> in-cluster services (set directly)
  RAY_ADDRESS=local (single-process in the KFP pod; set ``auto`` to join KubeRay).

Compile::

  uv run python -m src.kfp_pipeline.run_pipeline
  -> src/kfp_pipeline/feature_pipeline.yaml  (upload to Kubeflow UI -> Run)
"""
from __future__ import annotations

import os as _os

import kfp
from kfp import dsl
from kfp.components import func_to_container_op
from kubernetes.client import V1EnvVar, V1EnvVarSource, V1SecretKeySelector

# Base image: reuse the Ray image (uv + torch + mlflow + papermill + all repo
# deps). Build via infra/scripts/build_push.sh (recsys-ray). Override with
# KFP_BASE_IMAGE at compile time for a slimmer image.
BASE_IMAGE = _os.environ.get("KFP_BASE_IMAGE", "nmcuong08/recsys-ray:v1")


def feature_etl_op(output_path: str):
    """Papermill-execute the feature ETL notebook."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nb_path = "/app/feature/etl/003-feature-etl.ipynb"
    print(f"Papermill-executing {nb_path} -> {output_path}")
    if not os.path.exists(nb_path):
        print(f"Error: notebook not found at {nb_path}", file=sys.stderr)
        sys.exit(1)
    subprocess.run(["uv", "run", "papermill", nb_path, output_path], check=True, cwd="/app")
    return (output_path,)


def feature_engineer_op(output_path: str):
    """Papermill-execute the feature-engineering notebook."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nb_path = "/app/feature/engineer/004-features.ipynb"
    print(f"Papermill-executing {nb_path} -> {output_path}")
    if not os.path.exists(nb_path):
        print(f"Error: notebook not found at {nb_path}", file=sys.stderr)
        sys.exit(1)
    subprocess.run(["uv", "run", "papermill", nb_path, output_path], check=True, cwd="/app")
    return (output_path,)


def negative_sampling_op(output_path: str):
    """Papermill-execute the negative-sampling notebook."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nb_path = "/app/feature/engineer/005-negative-sample.ipynb"
    print(f"Papermill-executing {nb_path} -> {output_path}")
    if not os.path.exists(nb_path):
        print(f"Error: notebook not found at {nb_path}", file=sys.stderr)
        sys.exit(1)
    subprocess.run(["uv", "run", "papermill", nb_path, output_path], check=True, cwd="/app")
    return (output_path,)


def prep_item2vec_op(output_path: str):
    """Papermill-execute the item2vec-prep notebook."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nb_path = "/app/feature/engineer/006-prep-item2vec.ipynb"
    print(f"Papermill-executing {nb_path} -> {output_path}")
    if not os.path.exists(nb_path):
        print(f"Error: notebook not found at {nb_path}", file=sys.stderr)
        sys.exit(1)
    subprocess.run(["uv", "run", "papermill", nb_path, output_path], check=True, cwd="/app")
    return (output_path,)


def train_item2vec_op(output_path: str):
    """Train the Item2Vec model (Ray Train + MLflow)."""
    import os
    import subprocess
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print("Running module: models.item2vec.train")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    subprocess.run(["uv", "run", "python", "-m", "models.item2vec.train"],
                   check=True, cwd="/app", env=env)
    return (output_path,)


def train_ranking_sequence_op(output_path: str):
    """Train the GRU ranking sequence model (frozen item2vec emb + MLflow)."""
    import os
    import subprocess
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print("Running module: models.ranking_sequence.train")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    subprocess.run(["uv", "run", "python", "-m", "models.ranking_sequence.train"],
                   check=True, cwd="/app", env=env)
    return (output_path,)


def convert_onnx_op(output_path: str):
    """Export the ranker to ONNX and build the Triton model repository."""
    import os
    import subprocess
    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print("Running module: models.ranking_sequence.convert2onnx_and_build_triton")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    subprocess.run(
        ["uv", "run", "python", "-m", "models.ranking_sequence.convert2onnx_and_build_triton"],
        check=True, cwd="/app", env=env,
    )
    return (output_path,)


feature_etl_c = func_to_container_op(feature_etl_op, base_image=BASE_IMAGE)
feature_engineer_c = func_to_container_op(feature_engineer_op, base_image=BASE_IMAGE)
negative_sampling_c = func_to_container_op(negative_sampling_op, base_image=BASE_IMAGE)
prep_item2vec_c = func_to_container_op(prep_item2vec_op, base_image=BASE_IMAGE)
train_item2vec_c = func_to_container_op(train_item2vec_op, base_image=BASE_IMAGE)
train_ranking_sequence_c = func_to_container_op(train_ranking_sequence_op, base_image=BASE_IMAGE)
convert_onnx_c = func_to_container_op(convert_onnx_op, base_image=BASE_IMAGE)


def _wire_env(task) -> None:
    """Attach PVC + MinIO/MLflow/Ray env to a task (mutates task in place)."""
    pvc = dsl.PipelineVolume(pvc="data-pvc")
    task.add_pvolumes({"/data": pvc}).add_env_variable(
        V1EnvVar(name="PVC_PATH", value="/data")
    )
    for key, env_name in (("accesskey", "AWS_ACCESS_KEY_ID"),
                          ("secretkey", "AWS_SECRET_ACCESS_KEY")):
        task.add_env_variable(
            V1EnvVar(name=env_name,
                     value_from=V1EnvVarSource(
                         secret_key_ref=V1SecretKeySelector(name="minio-creds", key=key)
                     ))
        )
    task.add_env_variable(V1EnvVar(
        name="MLFLOW_TRACKING_URI",
        value="http://mlflow-tracking-service.mlflow.svc.cluster.local:5000"))
    task.add_env_variable(V1EnvVar(
        name="MLFLOW_S3_ENDPOINT_URL",
        value="http://minio-service.mlflow.svc.cluster.local:9000"))
    task.add_env_variable(V1EnvVar(name="AWS_DEFAULT_REGION", value="us-east-1"))
    # Single-process Ray in the KFP pod. Set "auto" to join the KubeRay cluster.
    task.add_env_variable(V1EnvVar(name="RAY_ADDRESS", value="local"))
    task.add_pod_annotation("debug/mount-path", "/data")
    task.set_memory_request("2Gi")
    task.execution_options.caching_strategy.max_cache_staleness = "P0D"


@dsl.pipeline(
    name="Recsys Feature + Training Pipeline",
    description="MovieLens: ETL -> features -> neg-sampling + prep-item2vec -> "
                "train item2vec -> train ranking sequence -> export ONNX + Triton repo",
)
def feature_pipeline():
    """Kubeflow pipeline wiring 7 ops over a shared PVC."""
    etl = feature_etl_c(output_path="/data/papermill-output/003-feature-etl.ipynb")
    _wire_env(etl)

    engineer = feature_engineer_c(output_path="/data/papermill-output/004-features.ipynb")
    _wire_env(engineer)
    engineer.after(etl)

    neg = negative_sampling_c(output_path="/data/papermill-output/005-negative-sample.ipynb")
    _wire_env(neg)
    neg.after(engineer)

    prep = prep_item2vec_c(output_path="/data/papermill-output/006-prep-item2vec.ipynb")
    _wire_env(prep)
    prep.after(engineer)

    train_i2v = train_item2vec_c(output_path="/data/papermill-output/train-item2vec")
    _wire_env(train_i2v)
    train_i2v.after(prep)

    train_rank = train_ranking_sequence_c(
        output_path="/data/papermill-output/train-ranking_sequence")
    _wire_env(train_rank)
    train_rank.after(train_i2v, neg)

    convert = convert_onnx_c(output_path="/data/papermill-output/convert-onnx")
    _wire_env(convert)
    convert.after(train_rank)


if __name__ == "__main__":
    out = "src/kfp_pipeline/feature_pipeline.yaml"
    kfp.compiler.Compiler().compile(pipeline_func=feature_pipeline, package_path=out)
    print(f"Compiled -> {out}")