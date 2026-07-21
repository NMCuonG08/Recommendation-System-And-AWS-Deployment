# jenkins-stack — CI/CD + model promotion watcher (Stage 6)

Lean port of the reference `jenkins-stack/`. Jenkins controller + RBAC + gp3
PVC + Istio VirtualService, plus a watcher pod that polls MLflow and triggers
the Jenkins pipeline when a new champion ranker is registered.

## Files

| Path | Purpose |
|------|---------|
| `Chart.yaml` / `values.yaml` | helm chart (Jenkins controller) |
| `Dockerfile.jenkins` | Jenkins image (docker + kubectl + helm + uv + awscli) |
| `templates/{deployment,service,rbac,pvc,istio-vs}.yaml` | k8s manifests |
| `watcher-pod/watch_promotion.py` | poll MLflow champion tag -> trigger Jenkins |
| `watcher-pod/Dockerfile.watcher` | watcher image (mlflow + requests) |
| `watcher-pod/{deployment,service}.yaml` | watcher k8s manifests |

## Deploy

```bash
# 1. Build + push images (from repo root)
docker build -t nmcuong08/recsys-jenkins:v1 \
    -f infra/jenkins-stack/Dockerfile.jenkins .
docker build -t nmcuong08/recsys-model-watcher:v1 \
    -f infra/jenkins-stack/watcher-pod/Dockerfile.watcher .
docker push nmcuong08/recsys-jenkins:v1 nmcuong08/recsys-model-watcher:v1

# 2. Create the namespace + the Jenkins token Secret (no default — no leak)
kubectl create ns devops-tools
kubectl -n devops-tools create secret generic jenkins-creds \
    --from-literal=jenkins-token=<your-jenkins-api-token>

# 3. Install the chart (needs KServe/Istio from deploy_kserve.sh first)
helm install jenkins infra/jenkins-stack -n devops-tools

# 4. Apply the watcher pod
kubectl apply -f infra/jenkins-stack/watcher-pod/

# 5. In Jenkins: create pipeline job `pipeline_deploy_triton` from the repo
#    Jenkinsfile, register the `aws-credentials` credential (AWS access/secret).
```

## Pipeline (`Jenkinsfile`, repo root)

```
uv sync -> convert ranker to ONNX + Triton repo -> validate repo ->
aws s3 sync model_repository -> kubectl apply inferenceservice-triton + rollout
```

Triggered manually or by the watcher when MLflow tags a new champion
`ranking_sequence_rating` version.

## Differences vs the reference

- **uv instead of conda.** Reference Jenkins image baked a `datn` conda env
  with torch/mlflow/onnx. This port installs uv; the Jenkinsfile runs
  `uv sync --all-groups` against the repo's own `pyproject.toml` — one source
  of truth for deps.
- **gp3 dynamic PVC.** Reference used `local-storage` + hostPath PV (kind
  cluster). This port uses the `gp3` StorageClass from `infra/terraform_eks`
  (EBS CSI) — no static PV template.
- **No hardcoded Jenkins token.** The reference `watch_promotion.py` hardcoded
  `<REDACTED-reference-token>`. This port reads `JENKINS_TOKEN` only from
  the `jenkins-creds` Secret and exits if it is missing.
- **Champion tag, not stage.** The ranker is registered with stage `None` +
  tag `champion=true` (see `models/ranking_sequence/train`), so the watcher
  filters on the tag, not `stage == "production"`.
- **Model name** `ranking_sequence_rating` (run_name `ranking` + suffix),
  not `seq_tune_v1_sequence_rating`.
