# 06 — CI/CD + Monitoring (Jenkins + Locust)

**Mục tiêu**: auto deploy khi code/model mới + giám sát load hệ thống.

## 1. Kiến trúc reference

```
MLflow new champion -> watcher-pod -> webhook Jenkins -> Jenkinsfile (test+build+rollout)
                                                              |
Locust (load test) -> Grafana + Loki + Prometheus <- Triton metrics
```

- **Jenkins**: helm chart `jenkins-stack/`, `Jenkinsfile` (pytest → docker build
  → kubectl rollout serving).
- **Watcher pod**: `jenkins-stack/watcher-pod/watch_promotion.py` poll MLflow,
  khi model version mới đạt chuẩn → trigger Jenkins webhook.
- **Drift check**: `data_pipeline_aws/check_drift/app.py` (Evidently).
- **Monitor**: Grafana `dashboard-config.yaml`, Loki + Prometheus.
- **Load test**: `locustfile.py` gọi API Gateway.

## 2. File reference cần port

| Reference | Vai trò |
|-----------|---------|
| `Jenkinsfile` | CI/CD pipeline (test, build, rollout) |
| `jenkins-stack/{Chart,values,templates/*}` | Jenkins helm + Istio VS + RBAC + PV |
| `jenkins-stack/watcher-pod/{watch_promotion.py,Dockerfile.watcher,deployment.yaml}` | MLflow watcher |
| `locustfile.py` | load test kịch bản |
| `dashboard-config.yaml` | Grafana dashboard JSON |
| `data_pipeline_aws/check_drift/{app.py,Dockerfile}` | drift detection |
| `ui/{main.py,feature_store.yaml}` | demo dashboard UI (tuỳ chọn) |
| `.github/workflows/ci.yml` | GitHub Actions CI |

## 3. Kế hoạch port (TODO)

1. `Jenkinsfile` — pytest (models/item2vec + ranking_sequence) → build 3 serving
   image (build_push_serving.sh) → kubectl rollout api_gateway + feature_store +
   Triton InferenceService.
2. `infra/jenkins-stack/` helm chart (Jenkins + RBAC + PV).
3. `infra/jenkins-stack/watcher-pod/` — port `watch_promotion.py` poll MLflow
   champion tag → trigger Jenkins.
4. `locustfile.py` — load test `POST /recommend`.
5. `dashboard-config.yaml` + Grafana/Loki/Prometheus (helm).
6. (tuỳ chọn) `data_pipeline/check_drift/` port Evidently drift.

## 4. Trạng thái

🟡 **Code xong, chưa deploy lên cluster.**
- ✅ `Jenkinsfile` (uv + path `models/ranking_sequence/`, ensemble model)
- ✅ `locustfile.py` (POST /recommend, int MovieLens ids)
- ✅ `infra/jenkins-stack/` helm (Jenkins + RBAC + gp3 PVC + Istio VS)
- ✅ `infra/jenkins-stack/watcher-pod/` (watch_promotion.py env-driven, no hardcoded token)
- ❌ chưa `helm install`, chưa apply watcher, chưa setup Jenkins job
- ❌ Grafana + Loki + Prometheus dashboard (`dashboard-config.yaml`) — chưa port
- ❌ drift check (`data_pipeline/check_drift/`) — chưa port (tuỳ chọn)
