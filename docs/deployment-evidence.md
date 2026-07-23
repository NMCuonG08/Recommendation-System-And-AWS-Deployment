# 🛠️ Báo Cáo A — Minh Chứng Triển Khai Hạ Tầng (Deployment Evidence)
## Real-Time MovieLens Recommender System & MLOps Platform

---

## 📌 MỤC LỤC BÁO CÁO A
1. [Tổng Quan Kiến Trúc Triển Khai (Deployment Topology)](#1-tổng-quan-kiến-trúc-triển-khai-deployment-topology)
2. [Minh Chứng Chi Tiết 7 Dịch Vụ Core Services](#2-minh-chứng-chi-tiết-7-dịch-vụ-core-services)
   - [2.1. FastAPI Recommendation Gateway](#21-fastapi-recommendation-gateway)
   - [2.2. MLflow Experiment Tracking & Model Registry](#22-mlflow-experiment-tracking--model-registry)
   - [2.3. MinIO S3 Object Storage Console](#23-minio-s3-object-storage-console)
   - [2.4. Qdrant Vector Database Dashboard](#24-qdrant-vector-database-dashboard)
   - [2.5. Feast Feature Store API](#25-feast-feature-store-api)
   - [2.6. Triton Inference Server Metrics](#26-triton-inference-server-metrics)
   - [2.7. Locust Stress & Load Testing Framework](#27-locust-stress--load-testing-framework)
3. [Luồng Suy Luận Hai Giai Đoạn Thực Tế (Two-Stage Serving End-to-End Flow)](#3-luồng-suy-luận-hai-giai-đoạn-thực-tế-two-stage-serving-end-to-end-flow)
4. [Minh Chứng Mã Hạ Tầng dạng Mã (IaC - Infrastructure as Code)](#4-minh-chứng-mã-hạ-tầng-dạng-mã-iac---infrastructure-as-code)
5. [Bảng Đánh Giá & Checklist Triển Khai (Deployment Checklist Matrix)](#5-bảng-đánh-giá--checklist-triển-khai-deployment-checklist-matrix)

---

## 1. TỔNG QUAN KIẾN TRÚC TRIỂN KHAI (DEPLOYMENT TOPOLOGY)

Hệ thống được thiết kế và đóng gói linh hoạt với 2 chế độ triển khai (Deployment Modes) đảm bảo tính linh hoạt tối đa từ môi trường Developer tới Production:

* **Production-Like AWS Cloud Environment**:
  - Máy chủ EC2 Compute (`54.169.84.125`) vận hành FastAPI Gateway, Triton Inference Server, MLflow Server, và Qdrant Vector Database.
  - AWS RDS PostgreSQL cho dữ liệu OLTP + AWS DMS (Database Migration Service) thực hiện Logical CDC.
  - AWS S3 & AWS Lambda Function đồng bộ dữ liệu sự kiện thời gian thực.
* **Local Docker Container Stack**:
  - Sử dụng Docker Compose giả lập toàn bộ hệ sinh thái dịch vụ phụ trợ bao gồm MinIO (S3 API), Feast Feature Store API, Redis Online Store và PostgreSQL OLTP.

```
                    ┌─────────────────────────────────────────┐
                    │      Client Request / API User          │
                    └────────────────────┬────────────────────┘
                                         │ HTTP GET /infer?user_id=1
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │     FastAPI Gateway (Port 8080)         │
                    └──────┬──────────────────┬───────────────┘
                           │                  │
           1. Vector Candidate Generation     │ 2. Get User/Item Online Features
                           │                  │
                           ▼                  ▼
        ┌────────────────────────┐      ┌─────────────────────────┐
        │ Qdrant Vector DB       │      │ Feast Feature Store     │
        │ (Port 6333) - Item2Vec │      │ (Port 8010) - Redis 6379│
        └────────────────────────┘      └─────────────┬───────────┘
                           │                          │
                           └──────────────┬───────────┘
                                          │ 3. Deep Candidate Ranking
                                          ▼
                        ┌───────────────────────────────────┐
                        │ Triton Inference Server           │
                        │ (Port 8000/8002) ONNX Ensemble    │
                        └───────────────────────────────────┘
```

---

## 2. MINH CHỨNG CHI TIẾT 7 DỊCH VỤ CORE SERVICES

---

### 2.1. FastAPI Recommendation Gateway

* **Chức năng**: API Gateway đóng vai trò làm điều phối viên (Coordinator) chính tiếp nhận yêu cầu từ client, thực hiện candidate generation qua Qdrant, lấy online features từ Feast API và gửi sang Triton Server để ranking.
* **Phương thức triển khai**: Docker Container / Python FastAPI Uvicorn service trên EC2 port 8080.
* **Lệnh khởi chạy**:
  ```bash
  uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000
  ```
* **Lệnh kiểm thử & Output thực tế (curl)**:
  ```powershell
  curl.exe "http://54.169.84.125:8080/infer?user_id=1"
  ```
  **Output JSON thực tế (HTTP 200 OK):**
  ```json
  {
    "user_id": 1,
    "recommendations": [
      {"movie_id": 356, "score": 0.948681116104126},
      {"movie_id": 1198, "score": 0.9291552901268005},
      {"movie_id": 858, "score": 0.9279670715332031},
      {"movie_id": 4993, "score": 0.9112963676452637},
      {"movie_id": 5952, "score": 0.9089178442955017}
    ]
  }
  ```
* **Giao diện thực tế (Swagger UI)**:
![FastAPI Gateway Swagger UI](assets/screenshots/swagger-infer.png)

---

### 2.2. MLflow Experiment Tracking & Model Registry

* **Chức năng**: Quản lý phiên chạy huấn luyện, lưu vết tham số (hyperparameters), đường cong loss/accuracy và đăng ký phiên bản mô hình Champion.
* **Phương thức triển khai**: Python MLflow Server gắn với Postgres backend store & MinIO artifact root (`s3://recsys-ops/mlflow/`).
* **Lệnh khởi chạy**:
  ```bash
  mlflow server --host 0.0.0.0 --port 5000 \
    --backend-store-uri postgresql://<user>:<password>@postgres:5432/mlflow \
    --default-artifact-root s3://recsys-ops/mlflow/
  ```
* **Lệnh kiểm thử**:
  ```bash
  curl -I http://54.169.84.125:5000
  # Response: HTTP/1.1 200 OK
  ```
* **Giao diện thực tế**:
![MLflow Experiment Tracking UI](assets/screenshots/mlflow-experiments.png)

---

### 2.3. MinIO S3 Object Storage Console

* **Chức năng**: Giả lập dịch vụ lưu trữ AWS S3 Object Storage, chứa dữ liệu Parquet thô, Feature Store metadata, CDC Delta Logs và MLflow model artifacts.
* **Phương thức triển khai**: MinIO Docker Container (`minio/minio:latest`).
* **Cấu hình Port**: API Port `9000`, Web Console Port `9101` (Host) ➔ `9001` (Container).
* **Cặp tài khoản đăng nhập**: Username: `admin` | Password: `Password1234`.
* **Giao diện thực tế**:
![MinIO S3 Bucket Console](assets/screenshots/minio-bucket.png)

---

### 2.4. Qdrant Vector Database Dashboard

* **Chức năng**: Cơ sở dữ liệu Vector Database chuyên dụng lưu trữ không gian vector nhúng **Item2Vec** (~9.7k bộ phim), hỗ trợ tìm kiếm hàng xóm gần nhất HNSW với độ trễ sub-millisecond cho Stage 1 Candidate Generation.
* **Phương thức triển khai**: Qdrant Docker Container (`qdrant/qdrant:v1.12.4`).
* **Port**: REST API `6333`, gRPC `6433`.
* **Lệnh kiểm thử (REST API)**:
  ```powershell
  curl.exe http://54.169.84.125:6333/collections/movie_embeddings
  ```
* **Giao diện thực tế (Qdrant Web Dashboard)**:
![Qdrant Vector DB Dashboard](assets/screenshots/qdrant-collection.png)

---

### 2.5. Feast Feature Store API

* **Chức năng**: Phục vụ truy vấn đặc trưng thời gian thực (User & Movie online features) từ Redis Online Store cho Gateway với độ trễ siêu thấp.
* **Phương thức triển khai**: FastAPI Service đóng gói từ `feature/feature_store/feature_store_api.Dockerfile`.
* **Port**: `8010` (Host) ➔ `8000` (Container).
* **Lệnh kiểm thử (Swagger UI / Docs)**:
  ```bash
  curl -I http://localhost:8010/docs
  # Response: HTTP/1.1 200 OK
  ```
* **Giao diện thực tế (Feast Swagger UI)**:
![Feast Feature Store Swagger UI](assets/screenshots/feast-swagger.png)

---

### 2.6. Triton Inference Server Metrics

* **Chức năng**: Máy chủ suy luận AI hiệu năng cao từ NVIDIA hỗ trợ ONNX Runtime, Dynamic Batching và Concurrent Model Execution.
* **Phương thức triển khai**: Docker Container (`nvcr.io/nvidia/tritonserver:24.08-py3`).
* **Port**: HTTP `8000`, gRPC `8001`, Prometheus Metrics `8002`.
* **Lệnh kiểm thử Metrics (curl)**:
  ```powershell
  curl.exe http://54.169.84.125:8002/metrics
  ```
* **Giao diện thực tế (Prometheus Metrics Text)**:
![Triton Server Metrics](assets/screenshots/triton-metrics.png)

---

### 2.7. Locust Stress & Load Testing Framework

* **Chức năng**: Mô phỏng tải lớn từ 100 người dùng giả lập đồng thời liên tục gửi yêu cầu `/recommend` để kiểm tra sức chịu tải và đo đạc latency percentiles (p95/p99).
* **Lệnh khởi chạy**:
  ```powershell
  python -m locust -f locustfile.py --web-host 0.0.0.0 --web-port 8089 --host http://54.169.84.125:8080 --users 100 --spawn-rate 10 --autostart
  ```
* **Giao diện thực tế**:
![Locust Stress Testing Dashboard](assets/screenshots/locust-charts.png)

---

## 3. LUỒNG SUY LUẬN HAI GIAI ĐOẠN THỰC TẾ (TWO-STAGE SERVING END-TO-END FLOW)

Xác nhận luồng suy luận hoàn chỉnh từ client tới response qua các bước thực nghiệm:

1. **Bước 1: Client gửi Yêu cầu**:
   Client gửi HTTP request `GET /infer?user_id=1` tới API Gateway port `8080`.
2. **Bước 2: Stage 1 - Candidate Retrieval**:
   API Gateway truy vấn **Qdrant Vector DB** lấy Top 100 movie_ids có Cosine Similarity cao nhất dựa trên vị trí vector nhúng Item2Vec của bộ phim vừa tương tác.
   *(Nếu user chưa có lịch sử, tự động kích hoạt Fallback Popularity từ Redis Cache)*.
3. **Bước 3: Feature Enrichment**:
   API Gateway gửi danh sách 100 movie_ids + `user_id=1` sang **Feast API** (port 8010) để lấy các đặc trưng thời gian thực lưu trữ trong Redis (như `user_avg_rating`, `item_popularity_score`).
4. **Bước 4: Stage 2 - Deep Ranking (Triton Ensemble)**:
   Gateway gửi mảng đặc trưng sang **Triton Inference Server** (gRPC port 8001) qua mô hình ONNX Ensemble để tính xác suất người dùng sẽ bấm thích bộ phim (`score`).
5. **Bước 5: Trả kết quả**:
   Gateway sắp xếp giảm dần theo điểm `score` và trả về JSON Top 10 gợi ý hoàn chỉnh cho client trong thời gian **< 18ms**.

---

## 4. MINH CHỨNG MÃ HẠ TẦNG DẠNG MÃ (IaC - INFRASTRUCTURE AS CODE)

Toàn bộ hạ tầng trên Cloud AWS được định nghĩa bằng mã nguồn Terraform trong thư mục `infra/terraform_ec2` và `infra/terraform_cdc`:

### Kiểm tra Trạng thái Tài nguyên AWS bằng AWS CLI:
```powershell
aws ec2 describe-instances --query "Reservations[*].Instances[*].[InstanceId, State.Name, PublicIpAddress, InstanceType]" --output table
```

**Output thực tế từ AWS CLI:**
```
------------------------------------------------------------------
|                        DescribeInstances                       |
+----------------------+----------+---------------+--------------+
|  i-03694ba3d463e8259 |  running | 54.169.84.125 |  t3.xlarge   |
+----------------------+----------+---------------+--------------+
```

---

## 5. BẢNG ĐÁNH GIÁ & CHECKLIST TRIỂN KHAI (DEPLOYMENT CHECKLIST MATRIX)

| Dịch Vụ / Component | Công Nghệ | Endpoint / Port | Trạng Thái (Status) | Minh Chứng (Screenshot) |
| :--- | :--- | :--- | :--- | :--- |
| **API Gateway** | FastAPI / Uvicorn | `http://54.169.84.125:8080` | 🟢 HEALTHY (200 OK) | `swagger-infer.png` |
| **Model Registry** | MLflow Server | `http://54.169.84.125:5000` | 🟢 HEALTHY (200 OK) | `mlflow-experiments.png` |
| **Object Storage** | MinIO Console | `http://localhost:9101` | 🟢 HEALTHY (200 OK) | `minio-bucket.png` |
| **Vector Search** | Qdrant Vector DB | `http://54.169.84.125:6333` | 🟢 HEALTHY (200 OK) | `qdrant-collection.png` |
| **Feature Store** | Feast Store API | `http://localhost:8010` | 🟢 HEALTHY (200 OK) | `feast-swagger.png` |
| **Model Serving** | Triton Server | `http://54.169.84.125:8002` | 🟢 HEALTHY (200 OK) | `triton-metrics.png` |
| **Stress Testing** | Locust Framework | `http://localhost:8089` | 🟢 HEALTHY (200 OK) | `locust-charts.png` |

---
*Báo cáo Minh chứng Triển khai Hạ tầng được xác thực trực tiếp trên môi trường thử nghiệm.*
