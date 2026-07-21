# 04 — Modeling + Training (item2vec + ranking_sequence)

**Mục tiêu**: train Item2Vec embedding + GRU sequence ranker, export ONNX, build
Triton repo. Training Ray + MLflow; orchestration Kubeflow (reference).

## 1. Hai model — зачем 2 loại

- **Item2Vec** (Stage 1 retrieval): học item embedding 64-dim từ user sequences.
  Dùng cho Qdrant vector search (top-K similar) + làm **frozen embedding input**
  cho ranker.
- **Ranking sequence (GRU)** (Stage 2 rerank): nhận user sequence + candidate item
  features, tính click/rating probability. Rerank top-K candidate từ retrieval.

> Item2Vec = retrieval (broad, cheap, vector相似). Ranker = precision (personal,
> sequence-aware, expensive). Hai tầng = recall + precision, chuẩn recommender.

## 2. item2vec

| File | Vai trò |
|------|---------|
| `models/item2vec/{model.py,dataset.py,trainer.py,train.py,evaluate.py}` | train + eval |
| `configs/item2vec.yaml` | config (Ray Train + MLflow + Evidently) |
| `models/item2vec/_build_item2vec.py` + `007-train-item2vec.ipynb` | build + notebook |
| `models/output/item2vec/` | checkpoints, final_model, reports |

Run:
```bash
uv run python -m models.item2vec.train --config configs/item2vec.yaml
```

Local-pro: `RAY_ADDRESS=local`, MLflow `docker compose up -d` (localhost:5000).
AWS: `RAY_ADDRESS=auto` (EKS KubeRay), MLflow in-cluster → [`eks-deploy.md`](eks-deploy.md).

✅ **Xong.** Champion registered + tagged. Báo cáo: [`item2vec-training-report.md`](item2vec-training-report.md).

## 3. ranking_sequence

| File | Vai trò |
|------|---------|
| `models/ranking_sequence/{model.py,dataset.py,trainer.py,train.py}` | GRU ranker train |
| `configs/ranking_sequence.yaml` (+ `.smoke.yaml`) | config (frozen item2vec emb, Ray, MLflow) |
| `models/ranking_sequence/convert2onnx_and_build_triton.py` | export ONNX + build Triton repo |
| `models/output/ranking_sequence/` | checkpoints, final_model |

Run:
```bash
uv run python -m models.ranking_sequence.train --config configs/ranking_sequence.yaml
uv run python -m models.ranking_sequence.convert2onnx_and_build_triton
```

Output: `models/ranking_sequence/model_repository/` — 4 model:
- `ranker` (ONNX GRU)
- `id_mapper` (Python backend, load `id_mapper.json`)
- `item_pipeline` (Python backend, load `item_metadata_pipeline.dill`)
- `ensemble` (wires 3 trên lại)

✅ **Xong.** Train + ONNX export + Triton repo build OK.

## 4. KFP orchestration

| Reference | Repo này |
|-----------|----------|
| `src/kfp_pipeline/run_pipeline.py` — 5 task qua PVC | ✅ `src/kfp_pipeline/run_pipeline.py` (7 op) |
| `src/kfp_pipeline/feature_pipeline.yaml` — compiled | ✅ compiled (regenerated) |

> ✅ **KFP xong.** 7-op Kubeflow pipeline: ETL → features → neg-sampling +
> prep-item2vec → train item2vec → train ranking → convert ONNX + Triton repo.
> Notebooks chạy qua papermill, train qua `models.*.train` modules. PVC
> `data-pvc` + `minio-creds` Secret + in-cluster MLflow/MinIO. Xem
> [`src/kfp_pipeline/README.md`](../src/kfp_pipeline/README.md).

## 5. Infra (EKS)

| Chart | Vai trò | Trạng thái |
|-------|---------|-----------|
| `infra/mlflow-stack/` | MLflow + Postgres + MinIO (helm) | ✅ xong |
| `infra/ray-cluster/` | KubeRay RayCluster CPU (helm) | ✅ xong |
| `infra/terraform_eks/` | EKS + VPC (terraform) | ✅ xong |
| `infra/images/{mlflow,ray}.Dockerfile` | image build | ✅ xong |
| `infra/scripts/build_push.sh` | push mlflow+ray image | ✅ xong |

Deploy: [`eks-deploy.md`](eks-deploy.md). AWS keys: [`feature.md`](feature.md).

## 6. Trạng thái

✅ **Model + ONNX + KFP orchestration xong.** Train manual (notebook/script) hoặc auto qua Kubeflow pipeline.
