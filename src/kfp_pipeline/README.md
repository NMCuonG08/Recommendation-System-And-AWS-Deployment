# kfp_pipeline — Kubeflow end-to-end training pipeline (Stage 4)

`run_pipeline.py` compiles a 7-op Kubeflow pipeline that runs the full MovieLens
training flow on the EKS cluster, sharing a `data-pvc`.

## Ops

| # | Op | Runs | After |
|---|----|------|-------|
| 1 | feature_etl | papermill `feature/etl/003-feature-etl.ipynb` | — |
| 2 | feature_engineer | papermill `feature/engineer/004-features.ipynb` | 1 |
| 3 | negative_sampling | papermill `feature/engineer/005-negative-sample.ipynb` | 2 |
| 4 | prep_item2vec | papermill `feature/engineer/006-prep-item2vec.ipynb` | 2 |
| 5 | train_item2vec | `python -m models.item2vec.train` | 4 |
| 6 | train_ranking_sequence | `python -m models.ranking_sequence.train` | 3, 5 |
| 7 | convert_onnx | `python -m models.ranking_sequence.convert2onnx_and_build_triton` | 6 |

## Compile

```bash
uv run python -m src.kfp_pipeline.run_pipeline
# -> src/kfp_pipeline/feature_pipeline.yaml
```

## Upload + Run

1. Deploy Kubeflow manifests (Kind or EKS) — see `docs/eks-deploy.md`.
2. Create the `data-pvc` + `minio-creds` Secret in the Kubeflow namespace:
   ```bash
   kubectl -n kubeflow-user-example-com create secret generic minio-creds \
       --from-literal=accesskey=admin --from-literal=secretkey=Password1234
   ```
3. Open Kubeflow UI (:8000) → Pipelines → Upload `feature_pipeline.yaml` →
   Create Experiment → Run.
4. Verify in MLflow UI: champion `ranking_sequence_rating` + ONNX registered.

## Base image

Reuses `nmcuong08/recsys-ray:v1` (uv + torch + mlflow + papermill + repo deps).
Override at compile time: `KFP_BASE_IMAGE=... uv run python -m src.kfp_pipeline.run_pipeline`.

## Differences vs reference

- **7 ops vs 5**: reference had feature / neg-sampling / prep-item2vec /
  train-item2vec / train-ranking (5). This port adds `feature_etl` (notebook
  `003`) up front and `convert_onnx` (export ONNX + Triton repo) at the end —
  the reference converted ONNX in the Jenkinsfile, not in KFP.
- **Notebooks via papermill**: reference ran standalone scripts
  (`000_feature_pipeline.py`, `010_negative_sample.py`, ...). This port's
  feature steps are notebooks, so ops papermill-execute them. Train steps use
  the repo's `models.*.train` modules (this port has `train.py` modules, the
  reference used `main.py`).
- **In-cluster MinIO + MLflow** (not AWS S3/RDS). `minio-creds` Secret instead
  of `aws-credentials`; `RAY_ADDRESS=local` (single-process in the KFP pod).
