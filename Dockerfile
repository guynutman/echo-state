FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install uv==0.12.3

COPY pyproject.toml README.md ./
COPY echostate ./echostate

RUN uv pip install --system --no-cache --torch-backend=cpu .

COPY experiments ./experiments

RUN useradd --create-home --uid 1000 echostate \
    && mkdir -p /app/output \
    && chown -R echostate:echostate /app

USER echostate
ENV HF_HOME=/home/echostate/.cache/huggingface

ENTRYPOINT ["python", "-m", "echostate.main"]
