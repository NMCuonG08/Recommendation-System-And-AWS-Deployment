# Custom MLflow server image (port of reference mlflow-stack/mlflow.Dockerfile).
# Upstream MLflow image lacks the postgres + S3/MinIO extras this stack needs.
# Published as <DOCKER_USER>/recsys-mlflow:v1 via infra/scripts/build_push.sh.
FROM ghcr.io/mlflow/mlflow:v2.16.2

RUN apt-get -y update && \
    apt-get -y install --no-install-recommends python3-dev build-essential pkg-config && \
    pip install --upgrade pip && \
    pip install psycopg2-binary boto3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# The actual `mlflow server ...` command is supplied by the helm chart's
# Deployment (see infra/mlflow-stack/templates/mlflow-deployment.yaml).
CMD ["bash"]