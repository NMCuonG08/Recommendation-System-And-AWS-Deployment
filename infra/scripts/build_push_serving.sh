#!/usr/bin/env bash
# Build and push the three serving images to Docker Hub:
#   - Triton server (infra/serving-cluster/Dockerfile.triton)
#   - API Gateway   (api_gateway/Dockerfile, repo-root context)
#   - Feast API     (feature/feature_store/feature_store_api.Dockerfile, repo-root context)
#
# Usage:
#   DOCKER_USER=cuongngx TAG=v1 bash infra/scripts/build_push_serving.sh
#   OR on Windows PowerShell:
#   bash infra/scripts/build_push_serving.sh cuongngx v1
#
# Produces:
#   <DOCKER_USER>/recsys-triton:<TAG>
#   <DOCKER_USER>/recsys-api-gateway:<TAG>
#   <DOCKER_USER>/recsys-feature-store-api:<TAG>
#
# After pushing, set these images in:
#   infra/serving-cluster/inferenceservice-triton.yaml  -> recsys-triton
#   api_gateway/deployment.yaml                         -> recsys-api-gateway
#   feature/feature_store/deployment.yaml               -> recsys-feature-store-api
set -euo pipefail

export DOCKER_BUILDKIT=1

DOCKER_USER="${1:-${DOCKER_USER:-cuongngx}}"
TAG="${2:-${TAG:-v1}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "==> Using DOCKER_USER=${DOCKER_USER}, TAG=${TAG} (BuildKit enabled)"
echo "==> Building ${DOCKER_USER}/recsys-triton:${TAG} (context: repo root)"
docker build -t "${DOCKER_USER}/recsys-triton:${TAG}" \
    -f "${REPO_ROOT}/infra/serving-cluster/Dockerfile.triton" "${REPO_ROOT}"

echo "==> Building ${DOCKER_USER}/recsys-api-gateway:${TAG} (context: repo root)"
docker build -t "${DOCKER_USER}/recsys-api-gateway:${TAG}" \
    -f "${REPO_ROOT}/api_gateway/Dockerfile" "${REPO_ROOT}"

echo "==> Building ${DOCKER_USER}/recsys-feature-store-api:${TAG} (context: repo root)"
docker build -t "${DOCKER_USER}/recsys-feature-store-api:${TAG}" \
    -f "${REPO_ROOT}/feature/feature_store/feature_store_api.Dockerfile" "${REPO_ROOT}"

echo "==> Logging in to Docker Hub (enter credentials if not already logged in)"
if ! docker info 2>/dev/null | grep -q "Username:"; then
    docker login
fi

for img in recsys-triton recsys-api-gateway recsys-feature-store-api; do
    echo "==> Pushing ${DOCKER_USER}/${img}:${TAG}"
    docker push "${DOCKER_USER}/${img}:${TAG}"
done

echo ""
echo "Done. Update images in:"
echo "  infra/serving-cluster/inferenceservice-triton.yaml -> ${DOCKER_USER}/recsys-triton:${TAG}"
echo "  api_gateway/deployment.yaml                        -> ${DOCKER_USER}/recsys-api-gateway:${TAG}"
echo "  feature/feature_store/deployment.yaml              -> ${DOCKER_USER}/recsys-feature-store-api:${TAG}"
