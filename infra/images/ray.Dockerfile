# Ray worker image with the 007 training deps + the repo baked at /app.
# Published as <DOCKER_USER>/recsys-ray:v1 via infra/scripts/build_push.sh.
# Base image: Ray 2.44.1, Python 3.11, CPU-only (MovieLens small needs no GPU).
FROM rayproject/ray:2.44.1-py311-cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install deps first (cache layer) — see requirements-ray.txt.
COPY requirements-ray.txt ./
RUN pip install --no-cache-dir -r requirements-ray.txt

# Bake the repo at /app so the head pod can run `python -m models.item2vec.train`
# with working_dir=/app and ship it (minus excludes in train.py runtime_env) to
# workers via Ray GCS. feature/output/engineer/ data ships with it (a few MB).
COPY . /app

WORKDIR /app