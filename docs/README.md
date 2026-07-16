# Data & Service Setup Guide

This project prepares the Kaggle **MovieLens ml-latest-small** dataset
(`shubhammehta21/movie-lens-small-latest-dataset`) and uploads it to an OLTP database
(Postgres) + object storage (S3), mirroring the reference project's `02-upload-data.ipynb`
flow.

> **Data scope.** MovieLens is a real **user × item × rating × timestamp** interaction
> dataset (610 users, ~9.7k movies, ~100k ratings) — proper collaborative-filtering data,
> unlike the earlier HQ Trivia sample which had no user dimension. `001` produces
> `train.parquet` + `val.parquet` (interactions) + `raw_meta.parquet` (movie metadata);
> `002` merges metadata, applies a temporal holdout, uploads OLTP rows to Postgres and
> the holdout parquet to S3. The pipeline is schema-agnostic, so swapping to another
> recsys dataset later needs only `KAGGLE_DATASET` + Args tweaks.

---

## Two ways to run

| Mode | `DATA_UPLOAD_USE_LOCAL` | Database | Object storage |
|------|------------------------|----------|----------------|
| **Local (learn first)** | `true` | Postgres in Docker | MinIO in Docker |
| **AWS (production-like)** | `false` | Amazon RDS Postgres | Amazon S3 |

Everything is driven by `.env`. Copy `.env.example` → `.env` and fill in.

---

## Path A — Local (Docker, recommended to start)

Local services mirror AWS 1:1, so the notebook code is identical — only env changes.

### Prerequisites
- Docker Desktop (Windows/Mac/Linux).

### 1. Start services

```bash
docker compose up -d
```

This starts:
- **Postgres** on `localhost:5435`, db `raw_data`, user `postgres` / pass `postgres`.
- **MinIO** (S3-compatible) on `http://localhost:9000` (API) and `http://localhost:9001` (web console, login `admin` / `Password1234`).
- A one-shot helper that creates the `recsys-ops` bucket in MinIO.

Check: `docker compose ps` → all healthy. MinIO console: open `http://localhost:9001`.

### 2. `.env` for local

```
DATA_UPLOAD_USE_LOCAL=true
S3_BUCKET=recsys-ops
S3_ENDPOINT=http://localhost:9000
PG_HOST=localhost
PG_PORT=5435
PG_DB=raw_data
PG_USER=
PG_PASSWORD=
```

> The notebook falls back to Docker defaults (`postgres`/`postgres`, MinIO
> `admin`/`Password1234`) when `PG_USER`/`AWS_ACCESS_KEY_ID` are empty, so leaving
> them blank works locally.

### 3. Stop / reset

```bash
docker compose down        # stop, keep data
docker compose down -v     # stop + wipe volumes (fresh start)
```

---

## Path B — AWS (real RDS + S3)

### Prerequisites
- An AWS account.
- Pick a region (e.g. `ap-southeast-1` Singapore — matches `.env` default).

### 1. Create an IAM user + access keys (for S3 / boto3)

1. AWS Console → **IAM** → **Users** → **Create user**. Name e.g. `recsys-dev`.
2. Attach permissions: for learning, **`AmazonS3FullAccess`** (narrow it in production).
3. Finish, open the user → **Security credentials** tab → **Create access key** →
   choose *Application running outside AWS* → copy `Access key ID` + `Secret access key`.
4. Put them in `.env`:

```
AWS_ACCESS_KEY_ID=AKIA...your key...
AWS_SECRET_ACCESS_KEY=...your secret...
AWS_DEFAULT_REGION=ap-southeast-1
```

> ⚠️ Never commit these. `.env` is gitignored. Rotate if leaked.

### 2. Create the S3 bucket

**Console:** S3 → **Create bucket** → name `recsys-ops` (globally unique — append a
suffix if taken, e.g. `recsys-ops-cuong`) → region same as above → keep defaults
→ Create.

**Or CLI:**
```bash
aws s3api create-bucket --bucket recsys-ops --region ap-southeast-1 \
  --create-bucket-configuration LocationConstraint=ap-southeast-1
```

Set in `.env`: `S3_BUCKET=recsys-ops`, and **leave `S3_ENDPOINT` empty** (real S3).

### 3. Create the RDS Postgres instance

AWS Console → **RDS** → **Create database**:

- **Engine:** PostgreSQL → latest 16.x.
- **Templates:** Free tier (if available) or Dev/Test.
- **DB instance identifier:** `recsys-oltp`.
- **Master username / password:** choose and record (→ `PG_USER` / `PG_PASSWORD`).
- **Instance class:** `db.t3.micro` (free/cheapest).
- **Storage:** 20 GB gp2 (default).
- **Connectivity:** Public access = Yes (for learning from your machine); choose
  or create a **security group**.
- **Security group inbound rule:** add **PostgreSQL / TCP 5432 / source = My IP**
  (or `0.0.0.0/0` for learning only — not for production).
  
After creation, find the **Endpoint** (e.g.
`recsys-oltp.xxxxxxx.ap-southeast-1.rds.amazonaws.com`).

Set in `.env`:
```
PG_HOST=recsys-oltp.xxxxxxx.ap-southeast-1.rds.amazonaws.com
PG_PORT=5432
PG_DB=postgres          # or create a DB named raw_data in RDS
PG_USER=<master username>
PG_PASSWORD=<master password>
PG_SCHEMA=public
PG_TABLE=movie_ratings
```

> Note: RDS uses port **5432** (not 5435 like local Docker). Update `PG_PORT`.
> Create a database `raw_data` in RDS first (RDS Console → Databases → your DB →
> Actions → Query, or `psql` / pgAdmin) if you don't want to use the default
> `postgres` DB.

### 4. Switch the flag

```
DATA_UPLOAD_USE_LOCAL=false
```

---

## Path C — AWS Feature Store (DynamoDB + RDS SQL registry)

Path B puts the **OLTP data** on AWS (RDS `raw_data` + S3). Path C puts the
**Feast Feature Store** on AWS too, so online serving reads from DynamoDB and
the feature-view registry lives in RDS (instead of local Redis + sqlite).

`feature/feature_store/feature_store.yaml` is already the AWS config
(`provider: aws`, `online_store: dynamodb`, `registry: sql`). Local mode is
saved at `feature/feature_store/feature_store.local.yaml` — to revert:
`cp feature/feature_store/feature_store.local.yaml feature/feature_store/feature_store.yaml`
and bring Redis up via `docker compose up -d`.

> Offline store stays `file` (reads the local parquet under `feature/output/`).
> That is intentional — feature engineering computes locally; only **serving +
> registry** go to AWS. To push offline to AWS too, switch `offline_store` to
> `spark` and point the `FileSource` paths in `feature_views.py` at `s3://...`
> (see the reference project's `feature_store/feature_store.yaml`).

### C.1 What AWS services does this add?

| Service | Purpose | Extra key needed? |
|---------|---------|--------------------|
| **DynamoDB** | Feast online store (online serving) | **No** — reuses `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`. Just needs IAM `dynamodb:*` permission. Feast creates tables on `feast apply`. |
| **RDS Postgres (2nd db)** | Feast registry (`registry_feature_store`) | **No new key** — reuses the RDS master `PG_USER` / `PG_PASSWORD`. You create one extra empty database. |

So **no new API keys** beyond what Path B already set. The remaining work is
**IAM permissions** + **one extra RDS database** + **one URI**.

### C.2 Give the IAM user DynamoDB permission

The same IAM user whose `AWS_ACCESS_KEY_ID` you put in `.env` (Path B step 1)
needs DynamoDB access in addition to S3.

1. AWS Console → **IAM** → **Users** → your `recsys-dev` user → **Add permissions** →
   **Attach policies directly**.
2. For learning, attach **`AmazonDynamoDBFullAccess`** (narrow it in production —
   e.g. a custom policy scoped to `Resource: arn:aws:dynamodb:<region>:<account>:table/*`).
3. Save. No new access key is generated — the existing key now covers DynamoDB.

> DynamoDB has no endpoint/credential field in `feature_store.yaml`. It picks up
> `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` from the
> environment automatically.

### C.3 Create the `registry_feature_store` database on RDS

Feast's SQL registry needs a dedicated database on the **same** RDS instance
that already hosts `raw_data`. Create it once:

- RDS Console → your DB → **Actions** → **Query** (or connect via `psql` /
  DBeaver using `PG_HOST` / `PG_PORT` / `PG_USER` / `PG_PASSWORD`), then run:

```sql
CREATE DATABASE registry_feature_store;
```

No tables needed — Feast creates its own `feast_metadata` etc. tables on
`feast apply`.

### C.4 Build `POSTGRES_URI_REGISTRY` in `.env`

`feature_store.yaml` reads the registry URI from `${POSTGRES_URI_REGISTRY}`.
Build it from the RDS master creds:

```
REGISTRY_DB=registry_feature_store
POSTGRES_URI_REGISTRY=postgresql://<PG_USER>:<urlencoded PG_PASSWORD>@<PG_HOST>:<PG_PORT>/registry_feature_store
```

- **URL-encode the password** if it contains `@ : / # ? &` etc. (e.g. `p@ss` →
  `p%40ss`). Quick encode: `python -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" 'yourpass'`
- Example:
  `postgresql://postgres:p%40ssw0rd@recsys-oltp.xxxxx.ap-southeast-1.rds.amazonaws.com:5432/registry_feature_store`

### C.5 Apply + materialize

From the repo root, load `.env` into the shell then run Feast from the
`feature/feature_store` dir (the config + `entities.py` + `feature_views.py`
live there):

```bash
# 1. set DATA_UPLOAD_USE_LOCAL=false and re-run 002 so OLTP is on RDS (if not already)
# 2. create the registry db (C.3) once

export $(grep -v '^#' .env | xargs)            # load AWS creds + POSTGRES_URI_REGISTRY + region
cd feature/feature_store
uv run feast apply                              # creates DynamoDB tables + registry schema
# materialize from the local offline parquet -> DynamoDB, up to the OLTP max timestamp
MATERIALIZE_CHECKPOINT_TIME=$(uv run ../../src/check_oltp_max_timestamp.py 2>&1 \
  | awk -F'<ts>|</ts>' '{print $2}')
uv run feast materialize 2010-01-01T00:00:00 "$MATERIALIZE_CHECKPOINT_TIME"
```

> `src/check_oltp_max_timestamp.py` reads `PG_*` from `.env` and prints the max
> `timestamp` of `public.movie_ratings` wrapped in `<ts>…</ts>` tags (the `awk`
> extracts it). If the OLTP table is empty, fall back to a fixed timestamp:
> `uv run feast materialize 2010-01-01T00:00:00 2026-07-16T00:00:00`.

### C.6 Verify

- **DynamoDB:** AWS Console → DynamoDB → **Tables** — you should see
  `feature_store_recsys_*` tables with item counts after materialize.
- **Registry:** connect to `registry_feature_store` on RDS and check Feast's
  metadata tables exist.

---

## Path D — MLflow tracking + Ray (007 Item2Vec training)

Path B puts the OLTP data on AWS; Path C puts the Feast Feature Store on AWS.
Path D puts the **Item2Vec training MLOps** on AWS too — Ray Tune HP search +
MLflow tracking + Model Registry champion tagging + Evidently reports —
mirroring the reference project's `src/model_item2vec/main.py` flow.

`models/item2vec/train.py` drives `configs/item2vec.yaml`, which is **env-driven**
(`${MLFLOW_TRACKING_URI}`, `${RAY_ADDRESS}`, …) so one config runs local-pro
and AWS.

### D.1 Local-pro (MLflow via docker-compose, Ray local)

The `mlflow` + `createmlflowdb` services in `docker-compose.yml` run an MLflow
server on http://localhost:5000 with a Postgres `mlflow` backend DB and a MinIO
artifact store (`s3://recsys-ops/mlflow/`). Ray runs as a local in-process
cluster (`RAY_ADDRESS=local`).

```bash
docker compose up -d                     # starts Postgres, MinIO, Redis, MLflow (+ creates `mlflow` DB)
curl http://localhost:5000               # MLflow UI
export $(grep -v '^#' .env | xargs)      # load MLFLOW_*, RAY_ADDRESS
uv run python -m models.item2vec.train --config configs/item2vec.yaml
```

What happens: Ray Tune runs `num_samples` trials (each logs to MLflow
`item2vec/hyperparameter_tuning`), picks the best by `val_loss`, then a final
training logs the TorchScript SkipGram + `idm.json` to the Model Registry
(`item2vec_skipgram`) and tags the champion version. Evidently classification
reports are logged as artifacts. Add `--overfit` for a single-batch sanity
check first (off by default).

Open http://localhost:5000 → Experiments (`item2vec/hyperparameter_tuning`,
`item2vec/final_model`) + Models (`item2vec_skipgram` with the champion version).

After training, generate figures + a metrics summary:
```bash
PYTHONPATH=. uv run python -m models.item2vec.evaluate --config configs/item2vec.yaml
```
Figures land in `models/output/item2vec/reports/figures/`; the full written
report (architecture, HP search, metrics, embedding analysis, limitations) is
in **[`docs/item2vec-training-report.md`](item2vec-training-report.md)**.

### D.2 AWS real (EKS + KubeRay + in-cluster MLflow)

The real "mẫu reference" path: an EKS cluster with the **KubeRay operator** running
the Ray cluster, and an **in-cluster MLflow stack** (Postgres backend + MinIO
artifact store) deployed via the `infra/mlflow-stack` helm chart. No EC2, no
separate RDS/S3 — Postgres + MinIO run as pods in the cluster (mirror of the
local-pro `docker-compose.yml` MLflow stack). CPU only (`t3.large`), MovieLens
small.

Full step-by-step (prereq tools → `terraform apply` → update-kubeconfig → ebs-csi
→ build/push images → KubeRay operator → mlflow-stack helm → port-forward →
ray-cluster helm → `kubectl exec` head pod to run train.py → verify MLflow UI +
champion → teardown) is in **[`docs/eks-deploy.md`](eks-deploy.md)**.

TL;DR from inside the Ray head pod (the only place these in-cluster DNS names
resolve):

```
RAY_ADDRESS=auto
MLFLOW_TRACKING_URI=http://mlflow-tracking-service.mlflow.svc.cluster.local:5000
MLFLOW_S3_ENDPOINT_URL=http://minio-service.mlflow.svc.cluster.local:9000
MLFLOW_AWS_ACCESS_KEY_ID=admin
MLFLOW_AWS_SECRET_ACCESS_KEY=Password1234
AWS_DEFAULT_REGION=ap-southeast-1
```

Then `python -m models.item2vec.train --config configs/item2vec.yaml`. Trials
run on the EKS Ray cluster; MLflow logs to the in-cluster server with artifacts
on the in-cluster MinIO (`s3://mlflow-artifacts`).

> The local-pro MLflow stack (`docker-compose.yml` `mlflow` service) is the
> 1:1 mirror of the in-cluster `infra/mlflow-stack` (MLflow + Postgres + MinIO),
> so the training code is identical — only env changes. To move to real RDS +
> S3 later, swap the helm values for an RDS backend + S3 artifact root and add
> IRSA (commented in `infra/terraform_eks/main.tf`).

---

## Environment variable reference

| Var | Local value | AWS value |
|-----|-------------|-----------|
| `DATA_UPLOAD_USE_LOCAL` | `true` | `false` |
| `AWS_ACCESS_KEY_ID` | *(empty)* | IAM access key |
| `AWS_SECRET_ACCESS_KEY` | *(empty)* | IAM secret |
| `AWS_DEFAULT_REGION` | `ap-southeast-1` | your region |
| `S3_BUCKET` | `recsys-ops` | your S3 bucket |
| `S3_ENDPOINT` | `http://localhost:9000` | *(empty)* |
| `PG_HOST` | `localhost` | RDS endpoint |
| `PG_PORT` | `5435` | `5432` |
| `PG_DB` | `raw_data` | RDS DB name |
| `PG_USER` | *(empty→postgres)* | RDS master user |
| `PG_PASSWORD` | *(empty→postgres)* | RDS master pass |
| `PG_SCHEMA` | `public` | `public` |
| `PG_TABLE` | `movie_ratings` | `movie_ratings` |
| `REGISTRY_DB` | `registry_feature_store` | `registry_feature_store` |
| `POSTGRES_URI_REGISTRY` | `postgresql://postgres:postgres@localhost:5435/registry_feature_store` | `postgresql://<user>:<urlencoded pass>@<rds-endpoint>:5432/registry_feature_store` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | `http://mlflow-tracking-service.mlflow.svc.cluster.local:5000` |
| `MLFLOW_BACKEND_STORE` | `postgresql://postgres:postgres@localhost:5435/mlflow` | *(in-cluster Postgres, managed by helm)* |
| `MLFLOW_ARTIFACT_ROOT` | `s3://recsys-ops/mlflow` | `s3://mlflow-artifacts` |
| `MLFLOW_S3_ENDPOINT_URL` | `http://localhost:9000` | `http://minio-service.mlflow.svc.cluster.local:9000` |
| `MLFLOW_AWS_ACCESS_KEY_ID` | `admin` (MinIO) | `admin` (in-cluster MinIO) |
| `MLFLOW_AWS_SECRET_ACCESS_KEY` | `Password1234` (MinIO) | `Password1234` (in-cluster MinIO) |
| `RAY_ADDRESS` | `local` | `auto` (inside head pod) / `ray://raycluster-kuberay-head-svc:10001` (outside) |

---

## Run order

```bash
make install          # uv sync --all-groups + register kernel (once)

# Path A: start local services
docker compose up -d

# Jupyter
make run-notebook     # or: uv run jupyter lab --no-browser
```

Then in Jupyter run notebooks in order:

1. **`001-prepare-dataset.ipynb`** — download (Kaggle), inspect, clean, persist
   `notebooks/data/001-prepare-dataset/clean.parquet`.
2. **`002-upload-data.ipynb`** — load clean parquet, split holdout, upload OLTP
   rows to Postgres and holdout parquet to S3 (MinIO or AWS).
3. **`feature/etl/003` → `feature/engineer/004–006`** — feature ETL + engineering
   + negative sampling + Item2Vec sequence prep (local compute).
4. **Feast apply + materialize** (Path C, AWS only) — `feature/feature_store`:
   `feast apply` then `feast materialize` to push features into DynamoDB.
5. **`models/item2vec/007-train-item2vec.ipynb`** — train Item2Vec via Ray Tune +
   MLflow + Evidently (Path D): `docker compose up -d` for MLflow, then
   `uv run python -m models.item2vec.train --config configs/item2vec.yaml`.
   Champion model lands in the MLflow Model Registry (`item2vec_skipgram`).
   Then run `models.item2vec.evaluate` for figures — see
   [`docs/item2vec-training-report.md`](item2vec-training-report.md) for results.

---

## Verifying the upload

**Postgres (local):**
```bash
docker exec -it recsys-postgres psql -U postgres -d raw_data \
  -c "SELECT COUNT(*) FROM public.movie_ratings;"
```

**MinIO:** open `http://localhost:9001`, login `admin` / `Password1234`, browse
bucket `recsys-ops` → `holdout.parquet`.

**AWS S3:**
```bash
aws s3 ls s3://recsys-ops/
```

**AWS RDS:**
```bash
psql "host=<endpoint> port=5432 dbname=raw_data user=<user>" \
  -c "SELECT COUNT(*) FROM public.movie_ratings;"
```

---

## Notes

- `psycopg2` lives in the `features` dependency group. Run `make install`
  (`uv sync --all-groups`) once so the upload notebook can connect to Postgres.
- `001` splits temporally: last 90 days of ratings → validation (cold-start-filtered to
  train users/items), the rest → train. `002` merges movie metadata (title, genres),
  holds out the last 30 days for S3, and pushes the rest to the OLTP table
  `movie_ratings` (PK on userId, movieId, timestamp).
- To swap datasets: change `KAGGLE_DATASET` in `.env` and the column names in the
  notebooks' `Args` (user/item/rating/timestamp). The pipeline is schema-agnostic.