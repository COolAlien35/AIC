FROM python:3.11-slim

WORKDIR /app

# Minimal build deps for Python wheels + git to pull source
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies. We deliberately use a slim subset of the
# repo-root requirements so the Docker image stays under HF Space size limits
# (no torch / transformers / peft - this Space only serves the env, not training).
COPY requirements-runtime.txt requirements-runtime.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --upgrade setuptools wheel \
    && pip install --no-cache-dir -r requirements-runtime.txt

# Runtime entrypoint that pulls the latest GitHub source and starts uvicorn.
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Hugging Face Spaces require the app on port 7860 by default.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=5)" || exit 1

CMD ["/app/start.sh"]
