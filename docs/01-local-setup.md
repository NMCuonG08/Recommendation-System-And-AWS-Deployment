# 01 — Local Setup (thiết lập môi trường cục bộ)

**Mục tiêu**: cài python env + deps, chạy unit tests, xác minh môi trường ổn.

## 1. Các bước

1. **Python 3.11 + uv**: tạo env, sync deps.
   ```bash
   pip install uv
   uv sync --all-groups          # cài toàn bộ group trong pyproject.toml
   make install                  # = uv sync + install ipykernel
   ```

2. **Pre-commit / style** (tuỳ chọn):
   ```bash
   make precommit
   make style
   ```

## 2. File cần đọc

| File | Vai trò |
|------|---------|
| `pyproject.toml` | deps + group (dev, aws, viz...), ruff/mypy config |
| `Makefile` | lệnh tắt: `install`, `run-notebook`, `kaggle-download` |
| `.env.example` → `.env` | toàn bộ config (PG_*, S3_*, MLFLOW_*, REDIS_*, ...) |

## 3. Verify

```bash
docker compose up -d            # Postgres + MinIO + Redis + Qdrant + MLflow + Triton
docker compose ps               # tất cả healthy
```

> Local services mirror AWS 1:1 (Postgres=OLTP, MinIO=S3, Redis=DynamoDB,
> MLflow tracking). Chi tiết: [`README.md`](README.md) "Path A".

## 4. Trạng thái

✅ **Xong.** `pyproject.toml`, `Makefile`, `.env.example`, `docker-compose.yml`
đã sẵn. Tests: ruff/mypy có config; unit tests cho model/pipeline — xem TODO
ở [`06-cicd-monitoring.md`](06-cicd-monitoring.md) (Jenkins sẽ chạy pytest).

## 5. Khác reference

Reference dùng conda env `recsys_ops` + `uv sync --all-groups`. Repo này dùng `uv`
trực tiếp (uv-managed venv), không cần conda.
