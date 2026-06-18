FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from pyproject.toml
RUN mkdir -p football_predictor
COPY pyproject.toml football_predictor/
COPY . football_predictor/
RUN pip install --no-cache-dir football_predictor/

# Create directories for persistent data
RUN mkdir -p /app/data /app/models

ENV FOOTBALL_DB_PATH=/app/data/football.db \
    MODELS_DIR=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "football_predictor.interface.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
