FROM python:3.11-slim

WORKDIR /app

# Minimal build deps for Python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies. We deliberately use a slim subset of the
# repo-root requirements so the Docker image stays under HF Space size limits
# (no torch / transformers / peft - this Space only serves the env, not training).
COPY hf_env_space/requirements-runtime.txt requirements-runtime.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-runtime.txt

# Copy the AIC source. We ship only what's needed to run the env service.
COPY aic ./aic
COPY server ./server
COPY openenv.yaml ./openenv.yaml
COPY pyproject.toml ./pyproject.toml
COPY hf_env_space/README.md ./README.md

# Hugging Face Spaces require the app on port 7860 by default.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=5)" || exit 1

CMD ["uvicorn", "aic.server.env_api:app", "--host", "0.0.0.0", "--port", "7860"]
