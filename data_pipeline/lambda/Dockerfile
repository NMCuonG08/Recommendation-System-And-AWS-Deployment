# CDC Lambda image (Feast online update on Kinesis trigger).
# Build context = repo root (so the script can reference this Dockerfile by path).
#   bash infra/scripts/build_push_lambda_cdc.sh
FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

# Install deps first for layer caching.
COPY data_pipeline/lambda/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bundle the handler + the Feast config + the bucket helper (self-contained).
COPY data_pipeline/lambda/lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY data_pipeline/lambda/feature_store.yaml ${LAMBDA_TASK_ROOT}/
COPY data_pipeline/lambda/timestamp_bucket.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.lambda_handler"]