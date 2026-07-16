# infra — EKS + KubeRay + MLflow (Path D.2)

IaC + helm + images cho 007 Item2Vec training trên EKS thật. CPU only
(`t3.large`), MovieLens small. In-cluster Postgres + MinIO (không RDS/S3 riêng).

Full step-by-step deploy: **[`../docs/eks-deploy.md`](../docs/eks-deploy.md)**.

## Layout

```
infra/
├── terraform_eks/        # EKS 1.30 + VPC + t3.large node group + EBS CSI (IRSA)
│   ├── provider.tf
│   ├── variables.tf      # aws_region, cluster_name, node_instance_type, desired_capacity
│   ├── main.tf           # vpc v5 + eks v20 + ebs-csi addon + IRSA role
│   └── outputs.tf        # cluster_endpoint, update_kubeconfig_command
├── mlflow-stack/         # helm: MLflow + Postgres:16 + MinIO (in-cluster)
│   ├── Chart.yaml
│   ├── values.yaml       # images, gp3 StorageClass, MinIO/Postgres creds
│   └── templates/        # storageclass, secret, postgres, minio, mlflow deployments+svc+pvc
├── ray-cluster/          # helm: KubeRay RayCluster (CPU)
│   ├── Chart.yaml
│   ├── values.yaml       # ray image, MLFLOW_* env, MinIO creds via Secret, gp3 PVC
│   └── templates/        # raycluster CR, _helpers.tpl, ray-pvc, ray-worker-pvc, secret
├── images/
│   ├── mlflow.Dockerfile # FROM ghcr.io/mlflow/mlflow:v2.16.2 + psycopg2 + boto3
│   ├── ray.Dockerfile    # FROM rayproject/ray:2.44.1-py311-cpu + repo deps + COPY . /app
│   └── requirements-ray.txt
└── scripts/
    └── build_push.sh     # build + push <DOCKER_USER>/recsys-{mlflow,ray}:v1
```

## Quick start

```bash
# 1. EKS up
cd infra/terraform_eks && terraform init && terraform apply
aws eks update-kubeconfig --name recsys-eks --region ap-southeast-1

# 2. Images (set DOCKER_USER first)
DOCKER_USER=<user> bash infra/scripts/build_push.sh

# 3. KubeRay operator
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm install kuberay-operator kuberay/kuberay-operator --version 1.1.0

# 4. MLflow stack (edit values.yaml mlflow.image first)
helm install mlflow infra/mlflow-stack -n mlflow --create-namespace
kubectl -n mlflow port-forward svc/mlflow-tracking-service 5000:5000

# 5. Ray cluster (edit values.yaml image.repository first)
helm install ray infra/ray-cluster -n ray --create-namespace

# 6. Submit 007 — see docs/eks-deploy.md step 6
```

## Notes

- **MinIO creds** `admin`/`Password1234` — learning only. Production: real S3 + IRSA
  (commented IRSA block in `terraform_eks/main.tf`).
- **gp3 PVC** dynamic via EBS CSI. `WaitForFirstConsumer` so pods schedule before
  volume binds.
- **Cross-namespace secret**: ray pods (`ray` ns) can't ref `mlflow` ns Secret →
  `ray-cluster` chart creates its own `minio-creds` (same values).
- **Data ship via runtime_env**: `feature/output/` (few MB) ships with code to
  workers — no EFS/data-pvc. `train.py` keeps `engineer_dir` relative for worker
  cwd resolution.
- **Teardown**: `docs/eks-deploy.md` step 8. Always `terraform destroy` to stop
  EKS billing.