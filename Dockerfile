# Serving image: FastAPI + the deployed model bundle baked in.
# Built by CI on every merge to main; tagged with the git commit so a running container
# can always be traced back to its source (see /version and the OCI labels below).
FROM python:3.11-slim

ARG GIT_SHA=unknown
ARG IMAGE_TAG=local
ARG BUILD_DATE=unknown
ARG MODEL_VERSION=unknown

LABEL org.opencontainers.image.title="house-price-api" \
      org.opencontainers.image.description="MLOps Project 4 - house price prediction API" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${IMAGE_TAG}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/OWNER/mlops-house-price-pipeline" \
      hpp.model.version="${MODEL_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODEL_DIR=/app/model GIT_SHA=${GIT_SHA} IMAGE_TAG=${IMAGE_TAG}

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# The bundle produced by the pipeline's deploy stage (deployments/current).
COPY deployments/current /app/model

RUN useradd --create-home --uid 10001 app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["uvicorn", "hpp.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
