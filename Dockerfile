# Data services (ingest/backfill). Multi-arch base: builds natively on the DGX Spark (aarch64).
# Model serving (M2+) uses NVIDIA's vLLM container for the Spark instead.
FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN uv sync --frozen --no-dev
RUN mkdir -p /app/docs
ENV WS_DATA_DIR=/data
VOLUME /data
ENTRYPOINT ["uv", "run", "--no-sync", "python"]
