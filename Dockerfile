# syntax=docker/dockerfile:1.7
# ---- build: produce a wheel from the source tree -------------------------------------------
FROM python:3.14-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build==1.3.0
COPY pyproject.toml README.md LICENSE ./
COPY callinsights ./callinsights
RUN python -m build --wheel --outdir /dist

# ---- runtime ---------------------------------------------------------------------------------
FROM python:3.14-slim
# "postgres" (default) or "postgres,asr" to bundle local faster-whisper + ffmpeg.
ARG EXTRAS="postgres"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DATA_DIR=/data \
    PORT=8003

RUN if echo "$EXTRAS" | grep -q asr; then \
      apt-get update && apt-get install -y --no-install-recommends ffmpeg \
      && rm -rf /var/lib/apt/lists/*; \
    fi \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home /home/app --create-home app \
 && mkdir -p /data && chown app:app /data

WORKDIR /app
# Hash-pinned runtime dependencies (incl. the postgres driver) first, as a cached layer.
COPY requirements.lock ./
RUN pip install --require-hashes --no-deps -r requirements.lock
COPY --from=build /dist/*.whl /tmp/
RUN WHEEL="$(ls /tmp/*.whl)" \
 && if echo "$EXTRAS" | grep -q asr; then pip install "${WHEEL}[asr]"; \
    else pip install --no-deps "$WHEEL"; fi \
 && rm -f /tmp/*.whl

USER app
VOLUME ["/data"]
EXPOSE 8003
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["callinsights", "healthcheck"]

CMD ["callinsights", "serve"]
