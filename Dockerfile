# SecureCodeRAG - reproducible environment
# Build:  docker build -t securecoderag .
# Tests:  docker run --rm securecoderag
# Bench:  docker run --rm securecoderag python -m scripts.run_benchmark
# API:    docker run --rm -p 8000:8000 securecoderag \
#             uvicorn src.api.main:app --host 0.0.0.0 --port 8000
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements.txt -r requirements-dev.txt

COPY . .
RUN mkdir -p logs results data/clean data/poisoned data/processed

# No API keys are baked in. Pass them at runtime, e.g.
#   docker run --env-file .env ...
CMD ["python", "-m", "pytest", "tests", "-q"]

