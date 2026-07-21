# 03 — Real-Time CDC (Change Data Capture) trên AWS

**Mục tiêu**: đồng bộ rating mới insert vào RDS `movie_ratings` → Feast online
(DynamoDB) `user_feature_view` trong < 10s, để serving sequence ranker thấy
interaction gần nhất mà không cần batch `feast materialize` lại.

> ✅ **Đã port.** Code + Terraform xong, **chưa apply lên AWS** (user chạy terraform
> + one-time SQL). Ràng buộc: KHÔNG local, đẩy hết AWS.

## 1. Kiến trúc

```
RDS recsys-oltp (movie_ratings)
   │ logical replication (pglogical) + rds.logical_replication=1
   ▼
AWS DMS Replication Instance (dms.t3.medium)
   │ Source endpoint = RDS, Target endpoint = Kinesis
   │ Task CDC mode (ongoing replication)
   ▼
Kinesis Data Stream "recsys-cdc"
   │ trigger (batch=1, window=1s)
   ▼
AWS Lambda (Docker image, ECR)
   │ decode Kinesis record → append interaction → write_to_online_store
   ▼
Feast DynamoDB online (user_feature_view)  ← registry RDS registry_feature_store
```

DMS Kinesis target format: `{"data":{"userId","movieId","rating","timestamp"},"metadata":{operation:"insert"}}`.
Lambda đọc `record["kinesis"]["data"]` → base64 decode → JSON.

## 2. File

| File | Vai trò |
|------|---------|
| `data_pipeline/lambda/lambda_function.py` | handler đọc Kinesis, parse CDC, append sequence, `write_to_online_store` |
| `data_pipeline/lambda/feature_store.yaml` | Feast config (dynamodb online + sql registry) bundled trong image |
| `data_pipeline/lambda/timestamp_bucket.py` | copy `feature/features/timestamp_bucket.py` (bucket nhất quán online/offline) |
| `data_pipeline/lambda/Dockerfile` | `public.ecr.aws/lambda/python:3.11`, COPY 3 file trên |
| `data_pipeline/lambda/requirements.txt` | `feast[aws]`, pandas, tenacity, psycopg2 (bỏ pyspark) |
| `infra/scripts/build_push_lambda_cdc.sh` | build image → ECR `recsys-cdc-lambda`, in `image_uri` |
| `infra/terraform_cdc/` | Kinesis + DMS + Lambda + IAM + Secrets Manager + RDS param group |

## 3. Adapt MovieLens (khác REF Amazon)

- Record fields: `userId, movieId` (camelCase) — không `user_id/parent_asin`.
- Feast entity join key: `userId` (Int64) — `entity_rows=[{"userId": uid}]`.
- Feature names: `user_rating_list_10_recent_movie` (không `..._asin`).
- Bucket: tái dùng `from_ts_to_bucket` (copy file, image self-contained).
- Online store: DynamoDB (REF dùng Redis; ta dùng DynamoDB cho nhất quán với
  `feature/feature_store/feature_store.yaml`).
- Bỏ drift checker + Airflow (Lambda trigger=Kinesis đủ).
- Feast URI từ Secrets Manager (env `REGISTRY_PATH_SECRET_ARN`), không hardcode.

## 4. RDS one-time setup (ngoài terraform, 1 lần)

Terraform tạo parameter group `recsys-cdc-logical-repl` (`rds.logical_replication=1`)
nhưng KHÔNG tự attach vào RDS hiện có (terraform không quản lý RDS đó). Sau apply:

```bash
# Attach param group + reboot (downtime ngắn — làm ngoài giờ)
aws rds modify-db-instance \
  --db-instance-identifier recsys-oltp \
  --db-parameter-group-name recsys-cdc-logical-repl \
  --apply-immediately --region ap-southeast-1
aws rds reboot-db-instance --db-instance-identifier recsys-oltp --region ap-southeast-1
```

Sau reboot, connect RDS (`psql ... recsys-oltp`), chạy:

```sql
CREATE EXTENSION IF NOT EXISTS pglogical;
CREATE PUBLICATION recsys_cdc_pub FOR TABLE movie_ratings;
```

> DMS có thể tự tạo logical slot native; nếu task CDC chạy được mà không cần
> publication thủ công thì bỏ bước `CREATE PUBLICATION`. Test cả 2.

## 5. Provision + start (user chạy)

```bash
# 1. Build + push lambda image → in ra image_uri
AWS_REGION=ap-southeast-1 bash infra/scripts/build_push_lambda_cdc.sh

# 2. Terraform apply (pass sensitive vars qua -var hoặc gitignored .tfvars)
cd infra/terraform_cdc
terraform init
terraform apply \
  -var="rds_endpoint=recsys-oltp.cng4gycq4m4s.ap-southeast-1.rds.amazonaws.com" \
  -var="rds_password=<RDS_PASSWORD>" \
  -var="feast_postgres_uri=postgresql://postgres:<pwd>@recsys-oltp...:5432/registry_feature_store" \
  -var="lambda_image_uri=<uri-from-step-1>" \
  -var="dms_vpc_subnet_ids=[\"subnet-xxx\",\"subnet-yyy\"]" \
  -var="dms_vpc_security_group_ids=[\"sg-xxx\"]"
# Note: subnet/sg phải cùng VPC với RDS, SG cho phép egress tới RDS:5432.

# 3. RDS one-time (mục 4): attach param group + reboot + SQL pglogical + publication.

# 4. Start DMS CDC task (output từ terraform apply):
$(terraform output -raw start_task_command)
```

IAM cần thêm cho AWS key (S3+DynamoDB đã có): Kinesis, DMS, Lambda, ECR,
Secrets Manager, RDS param group. Attach inline policy hoặc dùng role riêng.

## 6. Verify (end-to-end AWS)

1. Insert 1 rating mới vào RDS:
   ```sql
   INSERT INTO movie_ratings ("userId","movieId",rating,timestamp)
   VALUES (999, 42, 5.0, EXTRACT(EPOCH FROM NOW()));
   ```
2. Tail Lambda logs:
   ```bash
   $(terraform output -raw verify_command)
   # expect: "Updated feature userId=999 movieId=42"
   ```
3. DMS metrics CloudWatch → `CDCThroughput > 0`, `SourceLatency`.
4. Feast online query (từ máy có aws creds):
   ```python
   from feast import FeatureStore
   store = FeatureStore(repo_path="feature/feature_store")
   store.get_online_features(
     features=["user_feature_view:user_rating_list_10_recent_movie",
               "user_feature_view:item_sequence_ts_bucket"],
     entity_rows=[{"userId": 999}]).to_dict()
   # expect: "42" trong recent list, bucket 0 (vừa xảy ra)
   ```
5. Trước insert → list không có 42; sau insert (<10s) → có 42.

## 7. Khác reference

- **Terraform** thay console manual (reproducible, machine không cần thao tác).
- **DynamoDB** online thay Redis (nhất quán với feature_store.yaml).
- **Secrets Manager** giữ Feast URI + RDS password, không hardcode.
- **Bỏ** Airflow dags + drift checker (Evidently).
- **DMS dms.t3.medium** không free-tier — `terraform destroy` khi idle để dừng phí.

## 8. Trạng thái

🟡 **Code + Terraform xong, chưa apply AWS.** Cần user:
- [ ] `build_push_lambda_cdc.sh` push image ECR
- [ ] `terraform apply` (pass sensitive vars)
- [ ] RDS one-time: attach param group + reboot + pglogical + publication
- [ ] Start DMS task + verify end-to-end

## 9. Out of scope

- Drift checker (Evidently) — phase 6 nếu cần.
- Airflow orchestration — Lambda trigger đủ.
- `movie_feature_view` update — REF chỉ update user view; movie cnt/avg qua batch
  materialize hoặc bỏ. Thêm logic lambda nếu serving cần.
- Real-time retrain triggers (REF backlog) — giai đoạn sau, cần drift + AUC monitor.