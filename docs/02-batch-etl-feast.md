# 02 — Batch ETL + Feast Feature Store

**Mục tiêu**: nạp dữ liệu MovieLens vào Postgres (OLTP) + MinIO (S3), chạy ETL
đặc trưng, nạp vào Feast Feature Store (offline=parquet, online=Redis).

## 1. Dịch vụ

| Reference (AWS) | Repo này (local/AWS) |
|------------------|----------------------|
| RDS Postgres `simulate-oltp-db` | Postgres Docker / RDS Path B |
| S3 `recsys-ops` | MinIO Docker / S3 Path B |
| AWS Glue Spark `glue.py` | `feature/etl/003-feature-etl.ipynb` (pandas, không Spark) |
| Feast + Redis online | `feature/feature_store/` (entities, feature_views, materialize) |

## 2. Các bước

1. **Chuẩn bị + upload dữ liệu** (notebooks):
   - `notebooks/001-prepare-dataset.ipynb` — tải Kaggle MovieLens, sinh
     `train.parquet`/`val.parquet` + `raw_meta.parquet`.
   - `notebooks/002-upload-data.ipynb` — merge metadata, temporal holdout,
     upload OLTP rows vào Postgres + holdout parquet lên S3/MinIO.

2. **ETL đặc trưng** (notebooks, không Glue):
   - `feature/etl/003-feature-etl.ipynb` — đặc trưng movie rating cnt/avg
     90d/30d/7d, user sequence recent-10 + ts buckets.
   - `feature/engineer/004-features.ipynb` — gộp item + user features.
   - `feature/engineer/005-negative-sample.ipynb` — popularity negative sampling.
   - `feature/engineer/006-prep-item2vec.ipynb` — prep item2vec corpus.

3. **Feast apply + materialize**:
   ```bash
   cd feature/feature_store
   cp feature_store.local.yaml feature_store.yaml     # local dev
   feast apply
   MATERIALIZE_CHECKPOINT_TIME=$(uv run src/check_oltp_max_timestamp.py \
       | awk -F'<ts>|</ts>' '{print $2}')
   uv run feast materialize 2010-01-01T00:00:00 "$MATERIALIZE_CHECKPOINT_TIME"
   ```

## 3. File cần đọc

| File | Vai trò |
|------|---------|
| `feature/feature_store/entities.py` | entity `movieId`, `userId` |
| `feature/feature_store/feature_views.py` | `movie_feature_view`, `user_feature_view` |
| `feature/feature_store/feature_store.{local,docker}.yaml` | Feast config (Redis online) |
| `feature/feature_store/materialize.py` | helper materialize |
| `src/check_oltp_max_timestamp.py` | lấy checkpoint ts từ OLTP |
| `feature/features/{tfm.py,timestamp_bucket.py,negative_sampling.py}` | transform util |

## 4. Verify

```bash
uv run feast online-lookup user_feature_view --entity-key 1   # từ feature/feature_store
# hoặc qua Feast API (docker compose): POST :8010/user_features {"user_id":1}
```

## 5. Trạng thái

✅ **Xong (local).** Notebooks 001/002, feature/etl+engineer, Feast apply/materialize
đều chạy được local-pro.
❌ **AWS Glue**: không port `glue.py` (dùng notebook pandas thay). Nếu cần Spark
trên AWS → TODO port Glue job.
❌ **k8s deploy Feast API**: manifest `feature/feature_store/{deployment,service}.yaml`
đã viết (Stage 5) nhưng chưa apply lên cluster.

## 6. Khác reference

- ETL pandas trong notebook thay Glue Spark (MovieLens nhỏ ~100k rows).
- Feast config có 3 bản: `local` (Redis localhost), `docker` (Redis compose),
  `aws` (DynamoDB, Path C). Provider `local` + offline `file`.
