# 🚀 Báo Cáo & Hướng Dẫn Deployment Toàn Diện — MovieLens Recommender System

Tài liệu này cung cấp quy trình triển khai (Deployment Playbook) từng bước chi tiết cho các phân hệ đã hoàn thiện mã nguồn nhưng chưa deploy lên cụm hạ tầng AWS / Kubernetes.

---

## 🗺️ Tổng Quan Các Phân Hệ Triển Khai

| Stage | Phân Hệ | Mục Tiêu Triển Khai | Công Cụ & File Thực Thi |
|-------|---------|---------------------|------------------------|
| **Stage 3** | **Real-Time CDC** | Đồng bộ rating thời gian thực từ RDS ➔ Feast Online | `infra/scripts/build_push_lambda_cdc.sh`, `infra/terraform_cdc/`, AWS Lambda |
| **Stage 3.1** | **Data Drift Detection** | Giám sát suy giảm phân phối dữ liệu CDC với Evidently AI | `data_pipeline/check_drift/Dockerfile`, `app.py` |
| **Stage 5** | **Serving Cluster** | Deploy Triton Inference, Qdrant, Feast API & Gateway | `infra/scripts/build_push_serving.sh`, `infra/serving-cluster/`, `infra/qdrant/`, `api_gateway/` |
| **Stage 6** | **CI/CD & Monitoring** | Tự động hóa rollout khi model mới + Giám sát hạ tầng | `infra/jenkins-stack/`, `watcher-pod/`, `infra/monitoring/dashboard-config.yaml`, `locustfile.py` |

---

## ⚡ 1. Giai Đoạn 3: Triển Khai Luồng Real-Time CDC trên AWS

### Bước 1.1: Build & Push Lambda CDC Docker Image lên AWS ECR
```bash
# Thiết lập AWS Region và chạy script đóng gói Lambda image
export AWS_REGION=ap-southeast-1
bash infra/scripts/build_push_lambda_cdc.sh
```
*Ghi nhận lại giá trị `image_uri` xuất ra ở cuối script (ví dụ: `123456789012.dkr.ecr.ap-southeast-1.amazonaws.com/recsys-cdc-lambda:latest`).*

### Bước 1.2: Khởi tạo Hạ tầng CDC bằng Terraform
```bash
cd infra/terraform_cdc
terraform init

# Chạy terraform apply với các tham số kết nối thực tế
terraform apply \
  -var="rds_endpoint=recsys-oltp.xxxxxx.ap-southeast-1.rds.amazonaws.com" \
  -var="rds_password=YOUR_RDS_PASSWORD" \
  -var="feast_postgres_uri=postgresql://postgres:YOUR_RDS_PASSWORD@recsys-oltp.xxxxxx.ap-southeast-1.rds.amazonaws.com:5432/registry_feature_store" \
  -var="lambda_image_uri=<LAMBDA_IMAGE_URI_TU_BUOC_1.1>" \
  -var="dms_vpc_subnet_ids=[\"subnet-xxx\",\"subnet-yyy\"]" \
  -var="dms_vpc_security_group_ids=[\"sg-xxx\"]"
```

### Bước 1.3: Cấu hình One-Time trên RDS PostgreSQL & pglogical
Terraform tạo parameter group `recsys-cdc-logical-repl` (`rds.logical_replication=1`). Cần gán parameter group và reboot RDS:

```bash
# Attach Parameter Group và reboot RDS
aws rds modify-db-instance \
  --db-instance-identifier recsys-oltp \
  --db-parameter-group-name recsys-cdc-logical-repl \
  --apply-immediately --region ap-southeast-1

aws rds reboot-db-instance --db-instance-identifier recsys-oltp --region ap-southeast-1
```

Sau khi RDS reboot thành công, dùng `psql` hoặc DBeaver kết nối vào database `recsys_oltp` và thực thi:
```sql
CREATE EXTENSION IF NOT EXISTS pglogical;
CREATE PUBLICATION recsys_cdc_pub FOR TABLE movie_ratings;
```

### Bước 1.4: Khởi động DMS Replication Task
```bash
# Lấy lệnh start task từ Terraform output và chạy
$(terraform output -raw start_task_command)
```

### Bước 1.5: (Tuỳ chọn) Chạy Service Kiểm tra Data Drift (Evidently AI)
```bash
# Build và chạy container check drift
cd data_pipeline/check_drift
docker build -t recsys-drift-checker:latest .
docker run -d --name recsys-drift-checker \
  -e AWS_REGION=ap-southeast-1 \
  -e STREAM_NAME=recsys-cdc \
  -e USE_RDS=1 \
  -e RDS_HOST=recsys-oltp.xxxxxx.ap-southeast-1.rds.amazonaws.com \
  recsys-drift-checker:latest
```

---

## 🏗️ 2. Giai Đoạn 5: Triển Khai Serving Cluster (Triton + Qdrant + Gateway)

### Bước 2.1: Build & Push các Serving Container Images
```bash
# Build và push 3 container images: Triton server, Feast API, API Gateway
export DOCKER_USER=nmcuong08
export TAG=v1
bash infra/scripts/build_push_serving.sh
```

### Bước 2.2: Sync Triton Model Repository lên S3 / MinIO
```bash
# Đẩy toàn bộ cấu trúc ONNX model repository lên S3/MinIO bucket
aws --endpoint-url http://localhost:9000 s3 sync \
    models/ranking_sequence/model_repository/ s3://recsys-triton-repo/
```

### Bước 2.3: Nạp Vector Embeddings vào Qdrant & Redis Cache
```bash
# 1. Khởi chạy Redis & Qdrant local hoặc K8s service
uv run python -m src.caching_offline.load_qdrant
```

### Bước 2.4: Triển khai KServe & Triton Server trên Kubernetes
```bash
# 1. Cài đặt KServe CRD & Knative Serving
bash infra/serving-cluster/deploy_kserve.sh

# 2. Deploy Qdrant Vector Database
helm install qdrant infra/qdrant -n kubeflow-user-example-com --create-namespace

# 3. Deploy Feast API Service
kubectl apply -f feature/feature_store/deployment.yaml
kubectl apply -f feature/feature_store/service.yaml

# 4. Deploy API Gateway
kubectl apply -f api_gateway/deployment.yaml
kubectl apply -f api_gateway/service.yaml

# 5. Deploy Triton InferenceService
kubectl apply -f infra/serving-cluster/inferenceservice-triton.yaml
```

### Bước 2.5: Kiểm thử API Gợi Ý (Verification)
```bash
# Kiểm tra endpoint API Gateway
curl -X POST http://localhost:8080/recommend \
     -H "Content-Type: application/json" \
     -d '{"user_id": 1, "current_item_id": 10}'
```
*Kỳ vọng: API trả về mã `HTTP 200 OK` chứa danh sách 10 bộ phim được gợi ý kèm điểm số xếp hạng.*

---

## ⚙️ 3. Giai Đoạn 6: Triển Khai CI/CD, Watcher Pod & Monitoring

### Bước 3.1: Triển khai Jenkins Stack & MLflow Model Watcher
```bash
# 1. Install Jenkins Helm Stack
helm install jenkins infra/jenkins-stack/ -n jenkins --create-namespace

# 2. Deploy Watcher Pod theo dõi MLflow Champion Model
kubectl apply -f infra/jenkins-stack/watcher-pod/deployment.yaml -n kubeflow-user-example-com
```

### Bước 3.2: Chạy Giả Lập Tải (Load Testing) bằng Locust
```bash
# Chạy Locust test API Gateway với 50 người dùng giả lập
uv run locust -f locustfile.py --host http://localhost:8080 --users 50 --spawn-rate 5
```

### Bước 3.3: Triển khai Kubeflow Dashboard Links & Grafana ConfigMap
```bash
# Apply ConfigMap chứa liên kết điều hướng Kubeflow, MLflow, Jenkins & Grafana
kubectl apply -f infra/monitoring/dashboard-config.yaml
```

---

## 📝 Tổng Kết Trạng Thái Triển Khai

| Phân hệ | Mã nguồn & Config | Trạng thái Triển khai | Hành động khuyến nghị tiếp theo |
|---------|-------------------|----------------------|--------------------------------|
| **Stage 3 (CDC)** | ✅ Hoàn thành (`app.py`, `lambda_function.py`, `terraform_cdc/`) | 🟡 Sẵn sàng deploy | Thực hiện Bước 1.1 đến 1.4 khi sẵn sàng kết nối RDS AWS |
| **Stage 5 (Serving)** | ✅ Hoàn thành (`api_gateway`, `inferenceservice-triton.yaml`, `load_qdrant.py`) | 🟡 Sẵn sàng deploy | Thực hiện Bước 2.1 đến 2.5 để test endpoint trên cluster local hoặc EKS |
| **Stage 6 (CI/CD)** | ✅ Hoàn thành (`Jenkinsfile`, `watcher-pod`, `dashboard-config.yaml`) | 🟡 Sẵn sàng deploy | Helm install Jenkins stack và apply Watcher Pod |
