# 基础镜像和运行时使用同一 Python 版本，依赖版本以 uv.lock 为准。
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv
WORKDIR /app
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
ENV UV_PYTHON_DOWNLOADS=never \
    UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-emit-project \
        --format requirements-txt --output-file /tmp/requirements.txt \
    && uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python \
        --no-cache --require-hashes --requirements /tmp/requirements.txt

FROM python:3.12-slim-bookworm AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY app ./app

# 数据卷首次创建时继承此目录权限，机器人本身以普通用户运行。
RUN useradd --uid 10001 --create-home bot \
    && mkdir -p /app/data \
    && chown bot:bot /app/data
USER bot
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
