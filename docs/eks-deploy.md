# 007 Item2Vec — EKS + KubeRay + MLflow deploy guide (Path D.2 real)

Đây là path "mẫu reference" thật: EKS cluster + **KubeRay operator** chạy Ray
cluster + **in-cluster MLflow stack** (Postgres backend + MinIO artifact store)
deploy qua helm. CPU only (`t3.large`), MovieLens small. Không EC2, không RDS/S3
riêng — Postgres + MinIO chạy trong cluster (mirror local-pro `docker-compose.yml`).

Repo code đã bake sẵn trong ray image (`/app`); data (`feature/output/` vài MB
jsonl) ship qua Ray `runtime_env` — không cần EFS/data-pvc.

> **Cost**: EKS control plane ~$0.10/h + t3.large node ~$0.08/h mỗi con. Xong
> test → `terraform destroy` để không mất tiền.

---

## 0. Prerequisites (trên máy bạn)

Cài đủ trước khi bắt đầu:

| Tool | Cài | Kiểm tra |
|------|-----|----------|
| AWS CLI v2 | https://aws.amazon.com/cli/ | `aws --version` |
| kubectl | https://kubernetes.io/docs/tasks/tools/ | `kubectl version --client` |
| helm v3 | https://helm.sh/docs/intro/install/ | `helm version` |
| terraform ≥1.6 | https://developer.hashicorp.com/terraform/downloads | `terraform --version` |
| docker | https://docs.docker.com/get-docker/ | `docker --version` |
| Docker Hub account | https://hub.docker.com/ (free) | `docker login` |

AWS credentials: `aws configure` (access key + secret + region
`ap-southeast-1`). IAM user cần `AmazonEKSClusterPolicy` + quyền tạo VPC/IAM
roles. Easiest learning: `AdministratorAccess` (narrow in production).

Set `DOCKER_USER` (Docker Hub username) — dùng cho build/push image:
```bash
export DOCKER_USER=<your-docker-hub-username>
```

---

## 1. Terraform — tạo EKS cluster

```bash
cd infra/terraform_eks
terraform init
terraform apply
```

Review plan (VPC + EKS 1.30 + t3.large node group min1/max3/desired2 + EBS CSI
IRSA role). Type `yes`. ~10-15 phút.

Output cho lệnh update-kubeconfig:
```
update_kubeconfig_command = aws eks update-kubeconfig --name recsys-eks --region ap-southeast-1
```

Chạy nó:
```bash
aws eks update-kubeconfig --name recsys-eks --region ap-southeast-1
kubectl get nodes    # 2 t3.large nodes Ready
```

### EBS CSI driver

Terraform đã tạo `aws_eks_addon ebs-csi` + IRSA role. Verify:
```bash
kubectl get pods -n kube-system | grep ebs-csi    # ebs-csi-controller-* Running
```

gp3 StorageClass do `infra/mlflow-stack` chart tạo (xem step 4). PVC sẽ bind
tự động nhờ EBS CSI + `WaitForFirstConsumer`.

---

## 2. Build + push images

2 image: `recsys-mlflow:v1` (MLflow server + psycopg2 + boto3) và `recsys-ray:v1`
(Ray 2.44.1 CPU + repo deps + code baked vào `/app`).

```bash
cd ../../   # về repo root
docker login
DOCKER_USER=$DOCKER_USER bash infra/scripts/build_push.sh
```

Script build + push:
- `<DOCKER_USER>/recsys-mlflow:v1` (context `infra/images`, `mlflow.Dockerfile`)
- `<DOCKER_USER>/recsys-ray:v1` (context repo root, `ray.Dockerfile`, COPY `. /app`)

Verify trên Docker Hub: browse https://hub.docker.com/u/<DOCKER_USER> → thấy 2 repo.

> **Rate-limit**: Docker Hub free ~100 pull/6h/anonymous, 200/logged-in. Ray
> cluster 2 pod + MLflow 1 pod = ít pull. Nếu bị limit → dùng ECR private
> (create repo, `docker tag` + push to `<acct>.dkr.ecr.<region>.amazonaws.com/...`,
> set `image.repository` trong helm values + imagePullSecret). Out of scope ở đây.

---

## 3. KubeRay operator

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0
kubectl get pods -l app.kubernetes.io/name=kuberay-operator    # Running
```

---

## 4. MLflow stack (in-cluster Postgres + MinIO + MLflow)

Sửa `infra/mlflow-stack/values.yaml` `mlflow.image` thành image bạn vừa push:
```yaml
mlflow:
  image: <DOCKER_USER>/recsys-mlflow:v1
```

Install:
```bash
helm install mlflow infra/mlflow-stack -n mlflow --create-namespace
kubectl -n mlflow get pods    # postgres + minio + mlflow Running
kubectl -n mlflow get pvc     # Bound (gp3)
```

initContainer `minio/mc` tự tạo bucket `mlflow-artifacts` trong MinIO khi MLflow
pod khởi động.

### Port-forward MLflow UI (từ máy bạn)
```bash
kubectl -n mlflow port-forward svc/mlflow-tracking-service 5000:5000
```
Mở http://localhost:5000 → MLflow UI lên (chưa có experiment cho tới khi train chạy).

---

## 5. Ray cluster

Sửa `infra/ray-cluster/values.yaml` `image.repository` thành image bạn vừa push:
```yaml
image:
  repository: <DOCKER_USER>/recsys-ray
  tag: v1
```

Install:
```bash
helm install ray infra/ray-cluster -n ray --create-namespace
kubectl -n ray get raycluster    # READY
kubectl -n ray get pods          # 1 head + 1 worker Running
```

Ray head service: `raycluster-kuberay-head-svc` (namespace `ray`).

> Chart tạo riêng Secret `minio-creds` (admin/Password1234) trong namespace `ray`
> vì pod ở `ray` ns không reference được Secret ở `mlflow` ns. Cùng giá trị.

---

## 6. Submit 007 training

Code đã bake trong ray image (`/app`). `kubectl exec` vào head pod, set env, chạy:

```bash
HEAD_POD=$(kubectl -n ray get pod -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')

kubectl -n ray exec -it "$HEAD_POD" -- bash -c '
set -e
cd /app
export RAY_ADDRESS=auto
export MLFLOW_TRACKING_URI=http://mlflow-tracking-service.mlflow.svc.cluster.local:5000
export MLFLOW_S3_ENDPOINT_URL=http://minio-service.mlflow.svc.cluster.local:9000
export MLFLOW_AWS_ACCESS_KEY_ID=admin
export MLFLOW_AWS_SECRET_ACCESS_KEY=Password1234
export AWS_DEFAULT_REGION=ap-southeast-1
python -m models.item2vec.train --config configs/item2vec.yaml
'
```

Hoặc dùng RayJob CR (KubeRay) để submit có quản lý lifecycle — cùng env, xem
KubeRay docs. Default ở đây: `kubectl exec`.

What happens:
- Ray Tune chạy `num_samples` trials (mỗi trial log vào MLflow experiment
  `item2vec/hyperparameter_tuning`, chạy trên worker pod).
- Chọn best theo `val_loss` → final training log TorchScript SkipGram + `idm.json`
  vào Model Registry (`item2vec_skipgram`) + tag champion version.
- Evidently classification reports log as artifacts.
- Data `feature/output/` ship qua `runtime_env` (working_dir = `/app`, đã exclude
  `.venv/`, `*.parquet`, `*.ckpt`).

> Lần đầu Ray init chậm (build runtime_env, ship code/data) ~30-60s — bình thường.

### Sanity check trước (optional)
```bash
kubectl -n ray exec -it "$HEAD_POD" -- bash -c '
cd /app && python -m models.item2vec.train --config configs/item2vec.yaml --overfit
'
```
Single-batch overfit, vài giây, verify pipeline chạy trước khi full Tune.

---

## 7. Verify

Giữ `port-forward` MLflow UI (step 4) mở. Sau khi train xong:

- **MLflow UI** http://localhost:5000:
  - Experiments: `item2vec/hyperparameter_tuning` (các trial), `item2vec/final_model`.
  - Models: `item2vec_skipgram` với version được tag `champion`.
- **Worker logs** (trials chạy ở worker pod):
  ```bash
  kubectl -n ray logs -l ray.io/node-type=worker --tail=50
  ```
- **Artifacts** trong MinIO: port-forward MinIO console
  ```bash
  kubectl -n mlflow port-forward svc/minio-service 9001:9001
  ```
  Mở http://localhost:9001, login `admin`/`Password1234`, bucket `mlflow-artifacts`
  → thấy experiment artifacts.

---

## 8. Teardown (tránh phí tiền)

```bash
helm uninstall ray -n ray
helm uninstall mlflow -n mlflow
helm uninstall kuberay-operator
kubectl delete ns ray mlflow

cd infra/terraform_eks
terraform destroy    # type yes — xóa EKS + VPC + IAM roles
```

Verify EKS gone:
```bash
aws eks list-clusters --region ap-southeast-1    # không còn recsys-eks
```

---

## Troubleshooting

- **PVC Pending**: EBS CSI chưa ready → `kubectl get pods -n kube-system | grep ebs-csi`.
  Phải Running trước khi install helm. Terraform đã tạo addon, nếu thiếu:
  `aws eks create-addon --cluster-name recsys-eks --region ap-southeast-1 \
  --name aws-ebs-csi-driver --service-account-role-arn <IRSA-role-arn>`.
- **Pod ImagePullBackOff**: sai `DOCKER_USER` trong helm values, hoặc image private
  → check `kubectl describe pod` event. Ray pod chạy root + privileged (cần `/mnt/ray`).
- **Ray worker OOM**: `t3.large` 8GB. Giảm `configs/item2vec.yaml`
  `experiment.tune_config.num_samples` + `training.batch_size`.
- **MLflow connection refused từ ray pod**: sai namespace DNS. Phải là
  `mlflow-tracking-service.mlflow.svc.cluster.local` (service.ns.svc.cluster.local).
  Ray ở `ray` ns, MLflow ở `mlflow` ns → phải đầy đủ FQDN (đã set trong helm values).
- **Bucket missing**: MLflow initContainer tạo `mlflow-artifacts`. Nếu MinIO chưa
  ready khi MLflow start → initContainer retry đến khi OK. Check
  `kubectl -n mlflow logs <mlflow-pod> -c init-minio-bucket`.

---

## Files map (infra)

| Path | Role |
|------|------|
| `infra/terraform_eks/` | EKS + VPC + EBS CSI IaC |
| `infra/mlflow-stack/` | helm chart: MLflow + Postgres + MinIO |
| `infra/ray-cluster/` | helm chart: KubeRay RayCluster (CPU) |
| `infra/images/mlflow.Dockerfile` | MLflow server image |
| `infra/images/ray.Dockerfile` | Ray + repo deps image (code baked) |
| `infra/scripts/build_push.sh` | build + push both images to Docker Hub |
| `infra/README.md` | infra index |

See also: `docs/README.md` "Path D.2", `docs/feature.md`.