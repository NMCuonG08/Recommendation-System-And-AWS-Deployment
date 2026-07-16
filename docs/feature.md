# 007 Item2Vec — AWS keys to fetch

Đẩy 007 (Ray Tune + MLflow + Evidently) lên AWS. Local-pro chạy ngay không cần
key mới (MLflow docker + MinIO `admin`/`Password1234`). Để sang **AWS thật**,
kiếm/set các giá trị dưới đây — phần lớn **reuse** Path B/C đã có.

## Reuse (đã có từ Path B/C — không kiếm lại)

| Var | Lấy ở đâu |
|-----|-----------|
| `AWS_ACCESS_KEY_ID` | IAM → Users → `recsys-dev` → Security credentials |
| `AWS_SECRET_ACCESS_KEY` | cùng chỗ trên |
| `AWS_DEFAULT_REGION` | `ap-southeast-1` (hoặc region của bạn) |
| `S3_BUCKET` | S3 → bucket `recsys-ops` (Path B) |
| `PG_HOST` / `PG_PORT` / `PG_USER` / `PG_PASSWORD` | RDS endpoint + master creds (Path B) |

IAM user cần `AmazonS3FullAccess` (đã có Path B) — MLflow artifact upload.
Không cần DynamoDB cho 007.

## Mới — set trong `.env` (config, không phải API key)

EKS dùng **in-cluster** Postgres + MinIO (helm `infra/mlflow-stack`), không phải
RDS/S3 thật → creds là `admin`/`Password1234` (MinIO pod), không reuse AWS keys.

| Var | EKS value | Ghi chú |
|-----|-----------|---------|
| `MLFLOW_TRACKING_URI` | `http://mlflow-tracking-service.mlflow.svc.cluster.local:5000` | in-cluster MLflow service |
| `MLFLOW_BACKEND_STORE` | *(helm-managed)* | in-cluster Postgres pod, chart tạo sẵn |
| `MLFLOW_ARTIFACT_ROOT` | `s3://mlflow-artifacts` | in-cluster MinIO bucket (initContainer tạo) |
| `MLFLOW_S3_ENDPOINT_URL` | `http://minio-service.mlflow.svc.cluster.local:9000` | in-cluster MinIO service |
| `MLFLOW_AWS_ACCESS_KEY_ID` | `admin` | MinIO pod (match helm values) |
| `MLFLOW_AWS_SECRET_ACCESS_KEY` | `Password1234` | MinIO pod (match helm values) |
| `RAY_ADDRESS` | `auto` | chạy trong head pod → join local cluster |
| `DOCKER_USER` | `<your-docker-hub-username>` | build/push `recsys-mlflow:v1` + `recsys-ray:v1` |

## Việc cần làm trên AWS (infra, không phải key)

**EKS thật** — không phải EC2. Deploy EKS + KubeRay operator + in-cluster MLflow
stack (Postgres + MinIO pods) + Ray cluster, rồi `kubectl exec` vào head pod chạy
train. Code repo đã bake sẵn trong ray image (`/app`), data ship qua Ray
`runtime_env` (vài MB jsonl) — không cần EFS/data-pvc.

Toàn bộ step-by-step (terraform apply → update-kubeconfig → ebs-csi → build/push
images → KubeRay operator → mlflow-stack helm → port-forward → ray-cluster helm →
kubectl exec head pod → verify MLflow UI + champion → teardown): xem
**[`docs/eks-deploy.md`](eks-deploy.md)**.

Trong head pod, set env rồi chạy:
```bash
export RAY_ADDRESS=auto
export MLFLOW_TRACKING_URI=http://mlflow-tracking-service.mlflow.svc.cluster.local:5000
export MLFLOW_S3_ENDPOINT_URL=http://minio-service.mlflow.svc.cluster.local:9000
export MLFLOW_AWS_ACCESS_KEY_ID=admin
export MLFLOW_AWS_SECRET_ACCESS_KEY=Password1234
export AWS_DEFAULT_REGION=ap-southeast-1
python -m models.item2vec.train --config configs/item2vec.yaml
```

Chi tiết: `docs/README.md` "Path D.2" + `docs/eks-deploy.md`.