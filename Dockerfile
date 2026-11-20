# 基础镜像与 pip 源都做成 ARG，方便在受限网络下换成镜像站：
#   docker build --build-arg BASE_IMAGE=docker.m.daocloud.io/library/python:3.12-slim .
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDING_MODE=hashing \
    HF_ENDPOINT=https://hf-mirror.com

WORKDIR /app

COPY requirements.txt .
RUN pip install -i ${PIP_INDEX_URL} -r requirements.txt

COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY pyproject.toml README.md ./

EXPOSE 8771

# 首次启动：造数 → 清洗入库 → 建索引 → 起服务（data 目录建议挂卷，避免每次重建）
CMD ["sh", "-c", "test -f data/shop.db || (python -m app.data.generator --count 50000 && python -m app.data.etl); exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8771"]
