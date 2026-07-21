#!/usr/bin/env bash
# Build and push the CDC Lambda image to AWS ECR.
#
#   AWS_REGION=ap-southeast-1 bash infra/scripts/build_push_lambda_cdc.sh
#
# Creates (if missing) ECR repo "recsys-cdc-lambda", builds the image, tags it,
# pushes it, and prints the image URI to feed terraform:
#
#   terraform apply -var="lambda_image_uri=<printed-uri>"
#
# Requires: docker, aws CLI v2 (creds configured with ECR + STS perms).
set -euo pipefail

AWS_REGION="${AWS_REGION:-ap-southeast-1}"
REPO_NAME="${REPO_NAME:-recsys-cdc-lambda}"
TAG="${TAG:-latest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "==> Ensuring ECR repo ${REPO_NAME} exists"
aws ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null

echo "==> Logging in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

IMAGE_URI="${REGISTRY}/${REPO_NAME}:${TAG}"
echo "==> Building ${IMAGE_URI} (context: repo root)"
docker build -t "${IMAGE_URI}" -f "${REPO_ROOT}/data_pipeline/lambda/Dockerfile" "${REPO_ROOT}"

echo "==> Pushing ${IMAGE_URI}"
docker push "${IMAGE_URI}"

echo ""
echo "Done. Feed this to terraform:"
echo "  lambda_image_uri = \"${IMAGE_URI}\""