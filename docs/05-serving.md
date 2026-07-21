# 05 — Triton Serving (KServe + Qdrant + API Gateway)

**Mục tiêu**: deploy ONNX ranker lên Triton (KServe), Qdrant vector DB cho
retrieval, FastAPI API Gateway điều phối 2-stage recommend.

## 1. Kiến trúc serve

```
Client -> API Gateway (FastAPI)
            |-> Redis: rec:{movieId} + popular_movie_score (candidate retrieval)
            |-> Feast API: user_features + features_movie (online features)
            |-> Triton (KServe): ensemble = id_mapper + item_pipeline + ranker ONNX
            -> top-10 ranked items
```

Qdrant: index Item2Vec embedding, `src/caching_offline/load_qdrant.py` upsert +
precompute Redis cache (`rec:{movieId}`, `popular_movie_score`).

## 2. File

| File | Vai trò |
|------|---------|
| `api_gateway/{main.py,Dockerfile,deployment.yaml,service.yaml}` | gateway code + image + k8s |
| `feature/feature_store/{main.py,deployment.yaml,service.yaml}` | Feast API + k8s |
| `src/caching_offline/load_qdrant.py` | index Qdrant + precompute Redis |
| `infra/serving-cluster/deploy_kserve.sh` | install KServe + Knative + Istio + cert-manager |
| `infra/serving-cluster/Dockerfile.triton` | Triton image (dill/numpy/pandas/sklearn) |
| `infra/serving-cluster/inferenceservice-triton.yaml` | KServe InferenceService (CPU, MinIO repo) |
| `infra/serving-cluster/kubeconfig-serving.yaml` | serving cluster kubeconfig (placeholder) |
| `infra/qdrant/` | Qdrant helm chart (StatefulSet + Service + ConfigMap) |
| `infra/scripts/build_push_serving.sh` | build+push 3 image (triton/gateway/feast) |
| `docker-compose.yml` | local serving stack (Triton + feast_api + api_gateway + Qdrant) |

## 3. Deploy order

```bash
# 1. Index Qdrant + Redis cache (local-pro trước)
docker compose up -d redis qdrant
uv run python -m src.caching_offline.load_qdrant

# 2. Local serve (docker-compose): Triton + Feast API + Gateway
docker compose up -d triton feast_api api_gateway
curl -X POST http://localhost:8080/recommend \
     -H "Content-Type: application/json" \
     -d '{"user_id": 12345, "current_item_id": 99}'

# 3. EKS deploy (xem infra/serving-cluster/README.md)
DOCKER_USER=nmcuong08 TAG=v1 bash infra/scripts/build_push_serving.sh
aws --endpoint-url http://localhost:9000 s3 sync \
    models/ranking_sequence/model_repository/ s3://recsys-triton-repo/
KUBECONFIG=infra/serving-cluster/kubeconfig-serving.yaml \
    bash infra/serving-cluster/deploy_kserve.sh
# helm install qdrant infra/qdrant -n kubeflow-user-example-com --create-namespace
kubectl apply -f feature/feature_store/deployment.yaml
kubectl apply -f feature/feature_store/service.yaml
kubectl apply -f api_gateway/deployment.yaml
kubectl apply -f api_gateway/service.yaml
kubectl apply -f infra/serving-cluster/inferenceservice-triton.yaml
```

## 4. Trạng thái

🟡 **Code + manifests xong, chưa deploy lên cluster.**
- ✅ api_gateway code + Dockerfile + k8s manifests
- ✅ Feast API code + Dockerfile + k8s manifests
- ✅ Triton model_repository (ONNX + Python backends)
- ✅ Qdrant helm chart (render OK)
- ✅ KServe install script + InferenceService yaml + Dockerfile.triton
- ✅ build_push_serving.sh
- ❌ chưa `helm install qdrant`, chưa `kubectl apply`, chưa verify cluster

## 5. Khác reference

- **CPU free-tier** thay GPU (no GPU node group).
- **MinIO model repo** thay AWS S3 (`AWS_ENDPOINT_URL` + `minio-creds` secret).
- **Qdrant chart lean**: StatefulSet + 2 Service + ConfigMap + SA, bỏ
  ingress/PDB/serviceMonitor/snapshot của reference chart chính thức.
- **Triton image không COPY src**: Python backend load `id_mapper.json` +
  `item_metadata_pipeline.dill` từ model dir, không import `src/`.
