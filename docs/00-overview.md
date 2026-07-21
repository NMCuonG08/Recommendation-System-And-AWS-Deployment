# Lộ trình 6 giai đoạn — Recommender MLOps (MovieLens)

Bản đồ lộ trình, port từ `TUTORIAL_STEP_BY_STEP.md` của reference
`Ecommerce-Recommender-System-On-AWS-With-MLOps`, áp dụng cho repo này
(MovieLens `ml-latest-small`, free-tier EKS, in-cluster MinIO/Postgres/MLflow).

```
[1. Local Setup] -> [2. Batch ETL + Feast] -> [3. Real-Time CDC]
                                                          |
                                                          v
[6. CI/CD + Monitor] <- [5. Triton Serving] <- [4. Kubeflow + Ray]
```

## Trạng thái hiện tại

| # | Giai đoạn | Doc | Trạng thái |
|---|-----------|-----|-----------|
| 1 | Local Setup — env, deps, tests | [01-local-setup.md](01-local-setup.md) | ✅ xong |
| 2 | Batch ETL + Feast — notebooks 001/002, feature/etl+engineer, Feast | [02-batch-etl-feast.md](02-batch-etl-feast.md) | ✅ xong (local) |
| 3 | Real-Time CDC — DMS + Kinesis + Lambda | [03-realtime-cdc.md](03-realtime-cdc.md) | ❌ chưa làm |
| 4 | Modeling + Training — item2vec + ranking_sequence, ONNX, Ray/MLflow, KFP | [04-modeling-training.md](04-modeling-training.md) | ✅ xong |
| 5 | Triton Serving — KServe, Qdrant, API Gateway | [05-serving.md](05-serving.md) | 🟡 manifests xong, chưa deploy lên cluster |
| 6 | CI/CD + Monitoring — Jenkins, watcher, Locust, Grafana | [06-cicd-monitoring.md](06-cicd-monitoring.md) | 🟡 code xong, chưa deploy |

## Tham khảo sâu (deep-dives có sẵn)

- [`README.md`](README.md) — data setup (Stage 1 + 2 local/AWS, Path A–D).
- [`eks-deploy.md`](eks-deploy.md) — Stage 4 EKS + KubeRay + MLflow deploy (Path D.2).
- [`feature.md`](feature.md) — Stage 4 AWS keys checklist.
- [`item2vec-training-report.md`](item2vec-training-report.md) — kết quả train item2vec.

## Khác vs reference

- **Dữ liệu**: MovieLens `ml-latest-small` (610 users, ~9.7k movies, ~100k ratings)
  thay cho Amazon e-commerce reviews. Entity `movieId`/`userId` (int) thay
  `parent_asin`/`user_id` (str).
- **Offline store**: Feast `file` (parquet local) + MinIO thay AWS S3 + Glue Spark.
- **Online store**: Redis thay AWS DynamoDB.
- **Infra**: free-tier EKS, CPU only, in-cluster Postgres + MinIO + MLflow (helm),
  không RDS/S3 thật.
- **CDC (Stage 3)** + **CI/CD (Stage 6)**: reference dùng DMS/Kinesis/Lambda +
  Jenkins/Grafana; repo này chưa port — xem doc tương ứng để biết TODO.
