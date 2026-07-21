# 03 — Real-Time CDC (Change Data Capture)

**Mục tiêu**: đồng bộ click/rating mới từ Postgres sang Feast online (Redis)
<10ms qua CDC stream. Reference dùng AWS DMS + Kinesis + Lambda.

> ⚠️ **Chưa port.** Repo này chưa có CDC. Doc này mô tả kế hoạch port từ reference.

## 1. Kiến trúc reference

```
RDS Postgres (pglogical) -> AWS DMS (CDC) -> Kinesis Data Stream -> Lambda -> Feast Redis
```

- **RDS**: `rds.logical_replication=1` + `CREATE EXTENSION pglogical;`
- **DMS**: Replication Instance, Source=RDS, Target=Kinesis, CDC task.
- **Lambda**: Docker image từ `data_pipeline_aws/lambda/`, trigger=Kinesis, ghi Feast Redis.
- **Airflow**: `data_pipeline_aws/airflow/dags/` — `simulate_realtime_insert.py`
  chèn giao dịch mới để test; `reset_dag.py` reset; `preview.py` preview.

## 2. File reference cần port

| Reference | Vai trò |
|-----------|---------|
| `data_pipeline_aws/lambda/lambda_function.py` | handler đọc Kinesis record, parse CDC, write Feast online |
| `data_pipeline_aws/lambda/Dockerfile` | Lambda image (feast + redis client) |
| `data_pipeline_aws/lambda/feature_store.yaml` | Feast config cho Lambda |
| `data_pipeline_aws/airflow/dags/*.py` | Airflow dags simulate/reset/preview |
| `data_pipeline_aws/check_drift/{app.py,Dockerfile}` | drift check service (Stage 6) |
| `data_pipeline_aws/glue.py` | Glue ETL (Stage 2, đã bỏ qua) |

## 3. Kế hoạch port (TODO)

1. Tạo `data_pipeline/lambda/` — port `lambda_function.py` adapt MovieLens:
   - CDC record = insert vào `movie_ratings` (userId, movieId, rating, timestamp).
   - Lambda cập nhật Feast `user_feature_view` online (cộng recent sequence)
     + `movie_feature_view` cnt/avg.
2. Tạo `data_pipeline/lambda/Dockerfile` + `feature_store.yaml` (Redis online).
3. (Tuỳ chọn) Airflow dags `simulate_realtime_insert.py` để test local.
4. **Local thay DMS+Kinesis**: dùng Postgres logical replication + Redis Streams
   hoặc đơn giản hơn — script `simulate_realtime_insert.py` insert Postgres rồi
   trigger Lambda-like function trực tiếp (bỏ Kinesis cho local-pro).

## 4. Verify (khi xong)

1. Chèn 1 rating mới vào Postgres.
2. Check Lambda log nhận record.
3. Truy vấn Feast online — sequence user đã cộng interaction mới.

## 5. Trạng thái

❌ **Chưa làm.** Toàn bộ CDC stack chưa port. Ưu tiên thấp vì Stage 5 serving
và Stage 6 CI/CD quan trọng hơn cho luồng end-to-end. CDC chỉ cần khi muốn
real-time feature update (thay vì batch materialize lại).
