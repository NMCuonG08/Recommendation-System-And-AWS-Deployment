FROM python:3.10-slim

WORKDIR /app

COPY api_gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api_gateway/main.py /app/main.py

# Triton repo (ensemble) is mounted at runtime, not baked in.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]