"""向量化后端。

生产上应该用 BGE 这类中文语义模型；但为了 demo 能离线、零下载地跑起来，
这里做了三级兜底，接口完全一致，随时可切换：

1. fastembed + BAAI/bge-small-zh-v1.5  —— 本地 ONNX，512 维，真实语义；
2. OpenAI 兼容 /embeddings（通义 text-embedding-v3 等）—— 需要 key 与全量重建索引；
3. HashingEmbedder —— 词 + 字符 n-gram 的带符号哈希，零依赖，确定性，
   只能捕捉词面相似度，配合 BM25 的召回已经够 demo 用。
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from app.core.text import char_ngrams, normalize, stable_hash, tokenize

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _add(self, vec: np.ndarray, token: str, weight: float) -> None:
        h = stable_hash(token)
        idx = h % self.dim
        sign = 1.0 if (h >> 15) & 1 else -1.0
        vec[idx] += sign * weight

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        norm_text = normalize(text)
        for token in tokenize(norm_text):
            self._add(vec, "w:" + token, 1.0)
        for gram in char_ngrams(norm_text, 2):
            self._add(vec, "c:" + gram, 0.35)
        for gram in char_ngrams(norm_text, 3):
            self._add(vec, "t:" + gram, 0.2)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class FastEmbedEmbedder:
    """本地 BGE（onnxruntime），首次使用会下载模型（约 95MB）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from fastembed import TextEmbedding  # 延迟导入，没装也不影响启动

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)
        probe = next(iter(self._model.embed(["测试"])))
        self.dim = int(probe.shape[0])

    def embed(self, text: str) -> np.ndarray:
        vec = next(iter(self._model.embed([normalize(text)]))).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        arr = np.vstack([np.asarray(v, dtype=np.float32) for v in self._model.embed([normalize(t) for t in texts])])
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class OpenAICompatEmbedder:
    """OpenAI 兼容的 /embeddings，如 DashScope text-embedding-v3。"""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int = 1024):
        from openai import OpenAI

        self.name = model
        self.dim = dim
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        resp = self._client.embeddings.create(model=self.name, input=[normalize(t)[:2000] for t in texts])
        arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


def build_embedder(mode: str, dim: int = 512, *, base_url: str = "", api_key: str = "",
                   model: str = "", fastembed_model: str = "BAAI/bge-small-zh-v1.5") -> Embedder:
    if mode == "hashing":
        return HashingEmbedder(dim)
    if mode == "openai":
        return OpenAICompatEmbedder(base_url, api_key, model, dim=1024)
    if mode == "fastembed":
        try:
            return FastEmbedEmbedder(fastembed_model)
        except Exception as exc:  # noqa: BLE001 - 没装 fastembed 时别让服务起不来
            logger.warning("fastembed 不可用(%s)，回退 hashing embedder", exc)
            return HashingEmbedder(dim)
    # auto：优先真实语义模型，失败则退回 hashing，保证服务永远能起
    try:
        embedder = FastEmbedEmbedder(fastembed_model)
        logger.info("embedding backend = %s (dim=%s)", embedder.name, embedder.dim)
        return embedder
    except Exception as exc:  # noqa: BLE001
        logger.warning("fastembed 不可用(%s)，回退 hashing embedder", exc)
        return HashingEmbedder(dim)
