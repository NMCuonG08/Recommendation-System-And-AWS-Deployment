"""Watch the MLflow Model Registry for a new champion ranker and trigger Jenkins.

Ported from the reference `jenkins-stack/watcher-pod/watch_promotion.py`.

Security difference: the reference hardcoded a Jenkins API token
(`<REDACTED-reference-token>`). This port reads it ONLY from the env /
Secret — no default, no fallback. If `JENKINS_TOKEN` is unset the watcher logs
an error and exits instead of silently using a leaked credential.

Adapted to this port:
  - Registered model name `ranking_sequence_rating` (run_name `ranking` + suffix),
    not `seq_tune_v1_sequence_rating`.
  - Champion tag key `champion` (value `true`), not `stage == "production"`.
    The ranker is registered with stage `None` + a `champion=true` tag (see
    `models/ranking_sequence/train._log_final_ranker_to_mlflow`), so the watcher
    filters on that tag, not on the MLflow stage.
  - Jenkins job `pipeline_deploy_triton` triggers the Jenkinsfile with
    `MODEL_NAME` + `MODEL_VERSION` params.

Env (set via the watcher-pod Secret/Deployment, see deployment.yaml):
  MLFLOW_TRACKING_URI  default http://localhost:5000
  MODEL_NAME           default ranking_sequence_rating
  CHAMPION_TAG_KEY     default champion
  JENKINS_BASE         default http://jenkins-service.devops-tools.svc.cluster.local:8080/jenkins
  JENKINS_JOB          default pipeline_deploy_triton
  JENKINS_USER         default admin
  JENKINS_TOKEN        REQUIRED — no default, no fallback.
  POLL_INTERVAL_SEC    default 10
"""
from __future__ import annotations

import logging
import os
import time
from typing import Tuple

import requests
from mlflow.tracking import MlflowClient


def configure_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_configurations() -> Tuple[str, str, str, str, str, str, str, int]:
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    model_name = os.getenv("MODEL_NAME", "ranking_sequence_rating")
    champion_tag = os.getenv("CHAMPION_TAG_KEY", "champion")
    jenkins_base = os.getenv(
        "JENKINS_BASE",
        "http://jenkins-service.devops-tools.svc.cluster.local:8080/jenkins",
    )
    jenkins_job = os.getenv("JENKINS_JOB", "pipeline_deploy_triton")
    jenkins_user = os.getenv("JENKINS_USER", "admin")
    jenkins_token = os.getenv("JENKINS_TOKEN", "")
    poll_interval = int(os.getenv("POLL_INTERVAL_SEC", "10"))
    return (
        mlflow_uri, model_name, champion_tag,
        jenkins_base, jenkins_job, jenkins_user, jenkins_token, poll_interval,
    )


logger = configure_logging()
(
    MLFLOW_URI, MODEL_NAME, CHAMPION_TAG,
    JENKINS_BASE, JENKINS_JOB, JENKINS_USER, JENKINS_TOKEN,
    POLL_INTERVAL_SEC,
) = load_configurations()

if not JENKINS_TOKEN:
    logger.error(
        "JENKINS_TOKEN is not set. Refusing to start — the reference hardcoded a "
        "token, but this port requires it from the env/Secret. Set JENKINS_TOKEN "
        "in the watcher-pod Secret (see deployment.yaml)."
    )
    raise SystemExit(1)

JENKINS_URL = f"{JENKINS_BASE}/job/{JENKINS_JOB}/buildWithParameters"
client = MlflowClient(tracking_uri=MLFLOW_URI)


def get_jenkins_crumb_and_cookies() -> Tuple[str, str, requests.Session]:
    crumb_url = f"{JENKINS_BASE}/crumbIssuer/api/json"
    session = requests.Session()
    resp = session.get(crumb_url, auth=(JENKINS_USER, JENKINS_TOKEN), timeout=10)
    resp.raise_for_status()
    crumb_json = resp.json()
    return crumb_json["crumbRequestField"], crumb_json["crumb"], session


def trigger_jenkins(model_name: str, version: str) -> bool:
    try:
        crumb_field, crumb, session = get_jenkins_crumb_and_cookies()
        headers = {crumb_field: crumb}
        params = {"MODEL_NAME": model_name, "MODEL_VERSION": version}
        response = session.post(
            JENKINS_URL,
            auth=(JENKINS_USER, JENKINS_TOKEN),
            headers=headers,
            params=params,
            timeout=20,
        )
        ok = response.status_code in (200, 201, 202)
        logger.info(
            "Trigger Jenkins for %s v%s: %s - %s",
            model_name, version, response.status_code, response.text[:200],
        )
        return ok
    except Exception as e:  # noqa: BLE001 — watcher must keep looping
        logger.error("Failed to trigger Jenkins for %s v%s: %s", model_name, version, e)
        return False


def check_model_promotion() -> None:
    """Poll MLflow for champion-tagged versions and trigger Jenkins + tag `deploy`."""
    while True:
        try:
            versions = client.search_model_versions(f"name='{MODEL_NAME}'")
            champions = [
                v for v in versions
                if v.tags.get(CHAMPION_TAG, "").lower() == "true"
            ]
            if champions:
                champions.sort(key=lambda v: int(v.version), reverse=True)
                latest = champions[0]
                if latest.tags.get("deploy", "").lower() != "true":
                    logger.info("Found champion %s v%s", MODEL_NAME, latest.version)
                    if trigger_jenkins(MODEL_NAME, latest.version):
                        client.set_model_version_tag(
                            MODEL_NAME, latest.version, "deploy", "true"
                        )
                        logger.info("Tagged %s v%s deploy=true", MODEL_NAME, latest.version)
                    else:
                        logger.warning("Jenkins trigger failed for %s v%s", MODEL_NAME, latest.version)
                else:
                    logger.info("%s v%s already deployed", MODEL_NAME, latest.version)
            else:
                logger.info("No champion models found for %s", MODEL_NAME)
        except Exception as e:  # noqa: BLE001 — keep polling
            logger.error("Error in promotion check: %s", e)
        time.sleep(POLL_INTERVAL_SEC)


def main() -> None:
    logger.info("Starting model promotion watcher for %s ...", MODEL_NAME)
    check_model_promotion()


if __name__ == "__main__":
    main()
