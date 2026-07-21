# serving-cluster — KServe + Triton on the serving EKS cluster

Stage 5 serving manifests. Mirrors the reference `serving-cluster/` directory,
adapted to this port (MovieLens, free-tier EKS, in-cluster MinIO instead of AWS S3).

## Files

| File | Purpose |
|------|---------|
| `deploy_kserve.sh` | Install cert-manager + Knative + Istio + KServe (v0.12.0). |
| `Dockerfile.triton` | Triton server image with `dill`/`numpy`/`pandas`/`sklearn` for the Python backends. |
| `inferenceservice-triton.yaml` | KServe `InferenceService` running the `ensemble` ranker (CPU, MinIO model repo). |
| `kubeconfig-serving.yaml` | Placeholder kubeconfig for the serving cluster. |

## Deploy order

```bash
# 1. Build + push the Triton image (from repo root)
DOCKER_USER=nmcuong08 TAG=v1 bash infra/scripts/build_push_serving.sh

# 2. Sync the model repository to MinIO (after convert2onnx_and_build_triton)
#    bucket recsys-triton-repo — create it first via the MinIO console / mc.
aws --endpoint-url http://localhost:9000 s3 sync \
    models/ranking_sequence/model_repository/ s3://recsys-triton-repo/

# 3. Install KServe onto the serving cluster
KUBECONFIG=infra/serving-cluster/kubeconfig-serving.yaml \
    bash infra/serving-cluster/deploy_kserve.sh

# 4. Create the minio-creds Secret in the kserve namespace
#    (same admin/Password1234 as the mlflow/ray charts)
KUBECONFIG=infra/serving-cluster/kubeconfig-serving.yaml \
    kubectl -n kserve create secret generic minio-creds \
      --from-literal=accesskey=admin --from-literal=secretkey=Password1234

# 5. Deploy Triton
KUBECONFIG=infra/serving-cluster/kubeconfig-serving.yaml \
    kubectl apply -f infra/serving-cluster/inferenceservice-triton.yaml
```

## Verify

```bash
kubectl -n kserve get isvc recsys-triton   # InferenceService goes Ready

curl -X POST http://<api-gateway>/recommend \
     -H "Content-Type: application/json" \
     -d '{"user_id": 12345, "current_item_id": 99}'
```

## Differences vs the reference

- **CPU only.** Reference pinned a GPU image + GPU resources; this port targets
  the free-tier EKS node group, so resources are CPU and the image has no `-gpu` suffix.
- **MinIO model repo.** Reference read `s3://recsys-triton-repo/` from AWS S3
  (region `ap-southeast-1`). This port points the same bucket at the in-cluster
  MinIO via `AWS_ENDPOINT_URL` + `S3_USE_HTTPS=0`, reusing the `minio-creds`
  Secret the mlflow/ray charts already create.
- **No `src` baked in.** The Python backends load `id_mapper.json` +
  `item_metadata_pipeline.dill` from the model dir, so the Triton image only
  adds the pip deps — no `COPY src`.
