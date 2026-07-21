// CI/CD pipeline: convert ranker -> ONNX + Triton repo -> upload to S3/MinIO
// -> rollout KServe InferenceService on the serving cluster.
//
// Ported from the reference `Jenkinsfile`, adapted to this port:
//   - uv-managed env (uv sync --all-groups + uv run) instead of conda `datn`.
//   - ranker path models/ranking_sequence/ instead of src/model_ranking_sequence/.
//   - Triton ensemble model name `ensemble` (matches gateway MODEL_NAME).
//   - serving kubeconfig at infra/serving-cluster/kubeconfig-serving.yaml and
//     InferenceService at infra/serving-cluster/inferenceservice-triton.yaml.
//   - S3 upload targets MinIO bucket recsys-triton-repo via MLFLOW_S3_ENDPOINT_URL.
//
// Triggered manually or by the watcher-pod (watch_promotion.py) when MLflow
// registers a new champion ranker version.
pipeline {
  agent any

  environment {
    AWS_DEFAULT_REGION = "ap-southeast-1"
    S3_MODEL_REPO       = "s3://recsys-triton-repo/"
    MLFLOW_S3_ENDPOINT_URL = "http://minio-service.mlflow.svc.cluster.local:9000"
    MLFLOW_TRACKING_URI    = "http://mlflow-tracking-service.mlflow.svc.cluster.local:5000"
    MODEL_NAME         = "${params.MODEL_NAME ?: 'ranking_sequence_rating'}"
    MODEL_VERSION      = "${params.MODEL_VERSION ?: 'latest'}"
  }

  parameters {
    string(name: 'MODEL_NAME',    defaultValue: 'ranking_sequence_rating', description: 'MLflow registered model name (ranker)')
    string(name: 'MODEL_VERSION', defaultValue: 'latest',                 description: 'Model version to deploy (or latest)')
  }

  stages {

    stage('Sync deps') {
      steps {
        sh '''
          uv sync --all-groups
        '''
      }
    }

    stage('Convert to Triton repo') {
      steps {
        sh '''
          uv run python -m models.ranking_sequence.convert2onnx_and_build_triton
        '''
      }
    }

    stage('Test Triton repo + install awscli') {
      steps {
        sh '''
          # Validate model repository structure (4-model ensemble).
          if [ ! -d "./models/ranking_sequence/model_repository" ]; then
            echo "Error: Model repository directory not found"; exit 1
          fi
          if [ ! -f "./models/ranking_sequence/model_repository/ranker/1/model.onnx" ]; then
            echo "Error: ONNX ranker model file not found"; exit 1
          fi
          if [ ! -f "./models/ranking_sequence/model_repository/ensemble/config.pbtxt" ]; then
            echo "Error: ensemble config.pbtxt not found"; exit 1
          fi
          echo "Triton repository structure validated successfully"
          pip install --quiet awscli
        '''
      }
    }

    stage('Upload model repo to S3') {
      steps {
        withCredentials([aws(accessKeyVariable: 'AWS_ACCESS_KEY_ID',
                             secretKeyVariable: 'AWS_SECRET_ACCESS_KEY',
                             credentialsId: 'aws-credentials')]) {
          sh '''
            aws s3 rm ${S3_MODEL_REPO} --recursive || true
            aws s3 sync ./models/ranking_sequence/model_repository/ ${S3_MODEL_REPO}
            touch .keep
            aws s3 cp .keep ${S3_MODEL_REPO}ensemble/1/.keep
          '''
        }
      }
    }

    stage('Deploy to KServe') {
      steps {
        withCredentials([aws(accessKeyVariable: 'AWS_ACCESS_KEY_ID',
                             secretKeyVariable: 'AWS_SECRET_ACCESS_KEY',
                             credentialsId: 'aws-credentials')]) {
          sh '''
            export KUBECONFIG="./infra/serving-cluster/kubeconfig-serving.yaml"
            kubectl apply -f ./infra/serving-cluster/inferenceservice-triton.yaml --validate=false
            sleep 5
            kubectl delete pod -n kserve -l serving.kserve.io/inferenceservice=recsys-triton || true
          '''
        }
      }
    }
  }
}
