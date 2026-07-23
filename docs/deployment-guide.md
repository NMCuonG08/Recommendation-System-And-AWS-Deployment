# 🚀 Hướng Dẫn Triển Khai Recommender System Từ Đầu Cho Học Viên / Engineers

Tài liệu này hướng dẫn bạn **tự tay triển khai toàn bộ hệ thống Recommendation System** từ đầu đến cuối trên hạ tầng AWS (RDS PostgreSQL, AWS DMS, AWS Lambda, AWS S3, AWS ECR, EC2 Serving, Qdrant Vector DB, Triton Inference Server & API Gateway).

---

## 🛑 Trạng Thái Hiện Tại: Đã Clean sạch 100% Hạ Tầng Cũ (Clean State)
Tất cả tài nguyên AWS (DMS Replication Instance, Endpoints, Lambda, Parameter Groups, RDS Database, EC2 Instance) đã được **`terraform destroy`** và xoá sạch. Bạn sẽ bắt đầu tạo lại từng tài nguyên chính tay mình.

---

## 🛠️ Chuẩn Bị Trước Khi Thực Hiện (Prerequisites)

1. **AWS CLI & Đăng nhập tài khoản AWS**:
   ```powershell
   aws configure
   # Nhập AWS Access Key ID, Secret Access Key, Region: ap-southeast-1, Output format: json
   ```
2. **Kích hoạt Python Virtual Environment**:
   ```powershell
   .venv\Scripts\activate
   ```

---

## 📍 Bước 1: Khởi Tạo Hạ Tầng EC2 Serving & S3 Model Bucket (Terraform EC2)

1. Chuyển vào thư mục `infra/terraform_ec2`:
   ```powershell
   cd infra/terraform_ec2
   terraform init
   terraform apply -auto-approve
   ```
2. Lưu lại thông tin từ Output:
   - `ec2_public_ip`: IP công khai của EC2 Serving Server.
   - `triton_s3_bucket`: Tên S3 Bucket lưu trữ Triton Model Repository (dạng `recsys-triton-repo-<ACCOUNT_ID>`).

---

## 📍 Bước 2: Đồng Bộ Triton Model Repository Lên AWS S3

Thực hiện đồng bộ 9 file mô hình ONNX Ensemble và cấu hình Triton lên S3 Bucket vừa tạo:

Nếu bạn dùng **Git Bash (MINGW64)**:
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws s3 sync models/ranking_sequence/model_repository/ "s3://recsys-triton-repo-${ACCOUNT_ID}/" --region ap-southeast-1
```

Nếu bạn dùng **PowerShell**:
```powershell
$ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
aws s3 sync models/ranking_sequence/model_repository/ "s3://recsys-triton-repo-$ACCOUNT_ID/" --region ap-southeast-1
```

*Kiểm tra danh sách file trên S3:*
```powershell
aws s3 ls "s3://recsys-triton-repo-$ACCOUNT_ID/" --recursive --region ap-southeast-1
```

---

## 📍 Bước 3: Build & Push Lambda CDC Docker Image Lên AWS ECR

1. Đảm bảo Docker Desktop đã bật (hoặc chạy trong bash terminal):
   ```bash
   export AWS_REGION=ap-southeast-1
   bash infra/scripts/build_push_lambda_cdc.sh
   ```
2. Copy đường dẫn image URI xuất ra ở màn hình (ví dụ: `<ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/recsys-cdc-lambda:latest`).

---

## 📍 Bước 4: Tạo RDS PostgreSQL Database Instance (Nếu Chưa Có)

Do ở bước Clean Destroy chúng ta đã xóa sạch RDS cũ, bạn cần chạy 1 lệnh để tạo lại RDS Database Instance trên AWS:

```bash
aws rds create-db-instance \
  --db-instance-identifier recsys-oltp \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 18.1 \
  --allocated-storage 20 \
  --master-username postgres \
  --master-user-password <YOUR_RDS_PASSWORD> \
  --db-name postgres \
  --publicly-accessible \
  --db-parameter-group-name recsys-cdc-logical-repl \
  --vpc-security-group-ids <YOUR_SECURITY_GROUP_ID> \
  --region ap-southeast-1
```

*Lấy Endpoint URL mới sau khi RDS khởi tạo xong (~3-4 phút):*
```bash
aws rds describe-db-instances --db-instance-identifier recsys-oltp --query "DBInstances[0].Endpoint.Address" --output text --region ap-southeast-1
```

---

## 📍 Bước 5: Triển Khai Luồng Real-Time CDC Bằng Terraform CDC

1. Mở file `infra/terraform_cdc/terraform.tfvars` và điền cấu hình:
   ```hcl
   aws_region                  = "ap-southeast-1"
   rds_endpoint                = "<RDS_ENDPOINT_CỦA_BẠN>"
   rds_user                    = "postgres"
   rds_password                = "<YOUR_RDS_PASSWORD>"
   lambda_image_uri            = "<ACCOUNT_ID>.dkr.ecr.ap-southeast-1.amazonaws.com/recsys-cdc-lambda:latest"
   dms_vpc_subnet_ids          = ["<YOUR_SUBNET_ID_1>", "<YOUR_SUBNET_ID_2>"]
   dms_vpc_security_group_ids = ["<YOUR_SECURITY_GROUP_ID>"]
   ```

2. Khởi chạy Terraform CDC:
   ```powershell
   cd infra/terraform_cdc
   terraform init
   terraform apply -auto-approve
   ```

3. Lưu giá trị output:
   - `dms_replication_task_arn`: ARN của DMS Replication Task.
   - `s3_cdc_events_bucket`: Tên S3 Bucket nhận sự kiện CDC (`recsys-cdc-events-<ACCOUNT_ID>`).

---

## 📍 Bước 6: Tạo Bảng Database & Kích Hoạt Logical Replication (RDS)

1. Chạy Python script tạo bảng `movie_ratings` và khởi tạo Native Logical Publication (thay `<RDS_ENDPOINT>` và `<YOUR_RDS_PASSWORD>` tương ứng):
   ```powershell
   cd E:\MachineLearning\Recommendation_System
   .venv\Scripts\python.exe -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:<YOUR_RDS_PASSWORD>@<RDS_ENDPOINT>:5432/postgres'); conn.autocommit=True; cur=conn.cursor(); cur.execute('CREATE TABLE IF NOT EXISTS movie_ratings (user_id INT, movie_id INT, rating FLOAT, timestamp BIGINT, PRIMARY KEY (user_id, movie_id));'); cur.execute('CREATE PUBLICATION recsys_cdc_pub FOR TABLE movie_ratings;'); print('✅ DB Table & Publication Created!')"
   ```

---

## 📍 Bước 7: Khởi Động DMS Task & Kiểm Thử Luồng CDC Real-Time

1. Lấy ARN của DMS Task bằng lệnh CLI:
   ```bash
   TASK_ARN=$(aws dms describe-replication-tasks --query "ReplicationTasks[0].ReplicationTaskArn" --output text --region ap-southeast-1)
   ```

2. Khởi động DMS Replication Task:
   ```bash
   aws dms start-replication-task --replication-task-arn $TASK_ARN --region ap-southeast-1 --start-replication-task-type start-replication
   ```

3. Chèn dữ liệu rating mới vào RDS PostgreSQL để test CDC:
   ```powershell
   .venv\Scripts\python.exe -c "import psycopg2; conn = psycopg2.connect('postgresql://postgres:<YOUR_RDS_PASSWORD>@<RDS_ENDPOINT>:5432/postgres'); conn.autocommit=True; cur=conn.cursor(); cur.execute('INSERT INTO movie_ratings (user_id, movie_id, rating, timestamp) VALUES (9999, 10, 5.0, 1600000000);'); print('✅ Inserted test rating!')"
   ```

4. Kiểm tra kết quả CDC ghi vào S3 & AWS CloudWatch Lambda Logs:

   Nếu dùng **PowerShell**:
   ```powershell
   $ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
   aws s3 ls "s3://recsys-cdc-events-$ACCOUNT_ID/cdc_events/" --recursive --region ap-southeast-1
   aws logs tail /aws/lambda/recsys-cdc-lambda --region ap-southeast-1
   ```

   Nếu dùng **Git Bash (MINGW64)**:
   ```bash
   ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
   aws s3 ls "s3://recsys-cdc-events-${ACCOUNT_ID}/cdc_events/" --recursive --region ap-southeast-1
   aws logs tail /aws/lambda/recsys-cdc-lambda --region ap-southeast-1
   ```

---

## 📍 Bước 8: Triển Chạy Serving Stack Trên AWS EC2 Server

Toàn bộ mô hình Deep Learning Triton Server, Qdrant Vector DB, Redis Cache và FastAPI Gateway sẽ chạy **100% trên AWS EC2 Server**, giúp máy laptop của bạn không bị nóng hay tốn RAM.

1. **Nạp dữ liệu Vector nhúng vào Qdrant DB & Popularity Cache**:
   ```powershell
   .venv\Scripts\python.exe -m src.caching_offline.load_qdrant
   ```

2. **Kết nối vào AWS EC2 Server (Lấy Instance ID tự động, KHÔNG CẦN FILE KEY .pem)**:

   Nếu dùng **Git Bash**:
   ```bash
   INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].InstanceId" --output text --region ap-southeast-1)
   aws ec2-instance-connect ssh --instance-id $INSTANCE_ID --os-user ubuntu --region ap-southeast-1
   ```

   Nếu dùng **PowerShell**:
   ```powershell
   $INSTANCE_ID = (aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[0].Instances[0].InstanceId" --output text --region ap-southeast-1)
   aws ec2-instance-connect ssh --instance-id $INSTANCE_ID --os-user ubuntu --region ap-southeast-1
   ```

3. **Khởi chạy Docker Serving Cluster trên EC2**:
   Khi màn hình đen terminal EC2 xuất hiện, copy & dán khối lệnh này:
   ```bash
   git clone https://github.com/NMCuonG08/Recommendation-System-And-AWS-Deployment.git Recommendation_System
   cd Recommendation_System
   docker compose up -d
   ```

---

## 📍 Bước 9: Kiểm Thử API Gợi Ý Trực Tiếp Qua AWS EC2 Public IP

Gửi HTTP POST request tới EC2 Public IP (`<EC2_PUBLIC_IP>:8080`) để nhận kết quả gợi ý phim từ mô hình Deep Learning chạy trên AWS:

```powershell
# Lấy EC2 IP tự động:
$EC2_IP = (aws ec2 describe-instances --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region ap-southeast-1)

curl -X POST "http://${EC2_IP}:8080/recommend" -H "Content-Type: application/json" -d "{\"user_id\": 1, \"current_item_id\": 10}"
```

**Kỳ vọng kết quả (HTTP 200 OK từ EC2 Server AWS):**
```json
{
  "user_id": 1,
  "recommendations": [
    {"item_id": 318, "score": 0.9421},
    {"item_id": 296, "score": 0.9150},
    {"item_id": 593, "score": 0.8872}
  ]
}
```

---

## 🎯 Tổng Kết

Hệ thống Recommendation System của bạn thiết kế theo chuẩn doanh nghiệp (Production Grade):
- **Real-Time CDC**: RDS PostgreSQL ➔ AWS DMS ➔ S3 ➔ Lambda ➔ Feast Online Store.
- **Deep Learning Serving**: Triton Inference Server (ONNX Ensemble) + Qdrant Vector Search + Redis Cache + FastApi Gateway (chạy 100% trên AWS EC2).
- **Hạ Tầng Tự Động**: 100% Khai báo bằng Infrastructure as Code (Terraform).
