FROM python:3.10-slim

WORKDIR /app/feature_store

# Feast API serves from the online store (Redis/DynamoDB); offline parquet
# sources are only needed by `materialize.py`, not by the serving container.
COPY feature_store/requirements.txt /app/feature_store/requirements.txt
RUN pip install --no-cache-dir -r /app/feature_store/requirements.txt

COPY feature_store/ /app/feature_store/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]