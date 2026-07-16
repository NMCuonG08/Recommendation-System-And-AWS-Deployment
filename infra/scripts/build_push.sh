#!/usr/bin/env bash
# Build and push the two EKS images (MLflow server + Ray worker) to Docker Hub.
#
#   DOCKER_USER=<your-docker-hub-username> TAG=v1 bash infra/scripts/build_push.sh
#
# Produces:
#   <DOCKER_USER>/recsys-mlflow:<TAG>  (from infra/images/mlflow.Dockerfile)
#   <DOCKER_USER>/recsys-ray:<TAG>     (from infra/images/ray.Dockerfile, repo root context)
set -euo pipefail

DOCKER_USER="${DOCKER_USER:?Set DOCKER_USER to your Docker Hub username, e.g. DOCKER_USER=nmcuong08}"
TAG="${TAG:-v1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGES_DIR="${REPO_ROOT}/infra/images"

echo "==> Building ${DOCKER_USER}/recsys-mlflow:${TAG}"
docker build -t "${DOCKER_USER}/recsys-mlflow:${TAG}" \
    -f "${IMAGES_DIR}/mlflow.Dockerfile" "${IMAGES_DIR}"

echo "==> Building ${DOCKER_USER}/recsys-ray:${TAG} (context: repo root)"
docker build -t "${DOCKER_USER}/recsys-ray:${TAG}" \
    -f "${IMAGES_DIR}/ray.Dockerfile" "${REPO_ROOT}"

echo "==> Logging in to Docker Hub (enter credentials if not already logged in)"
if ! docker info 2>/dev/null | grep -q "Username:"; then
    docker login
fi

echo "==> Pushing ${DOCKER_USER}/recsys-mlflow:${TAG}"
docker push "${DOCKER_USER}/recsys-mlflow:${TAG}"

echo "==> Pushing ${DOCKER_USER}/recsys-ray:${TAG}"
docker push "${DOCKER_USER}/recsys-ray:${TAG}"

echo ""
echo "Done. Set in helm values:"
echo "  infra/mlflow-stack/values.yaml -> mlflow.image: ${DOCKER_USER}/recsys-mlflow:${TAG}"
echo "  infra/ray-cluster/values.yaml -> image.repository: ${DOCKER_USER}/recsys-ray  (tag: ${TAG})"