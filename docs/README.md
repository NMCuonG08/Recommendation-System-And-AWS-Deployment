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