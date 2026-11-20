"""文本处理工具：归一化、分词、价格解析、字符 n-gram。

分词用 jieba 的搜索模式（cut_for_search），它会把长词再切出子词，
对"降噪耳机""拍照手机"这类电商短语的召回更友好。
"""

from __future__ import annotations

import logging
import re
import unicodedata
import zlib

import jieba

jieba.setLogLevel(logging.WARNING)
_INITIALIZED = False

STOPWORDS: set[str] = {
    "的", "了", "和", "与", "及", "或", "在", "是", "有", "我", "你", "他", "她", "它", "们",
    "吗", "呢", "吧", "啊", "呀", "哦", "嗯", "这", "那", "个", "就", "都", "也", "还", "很",
    "太", "更", "最", "一", "二", "三", "请", "帮", "想", "要", "买", "个", "款", "台", "部",
    "什么", "怎么", "如何", "可以", "能", "有没有", "推荐", "一下", "一个", "一台", "一款",
    "现在", "最近", "比较", "一点", "点", "得", "地", "着", "过", "把", "被", "给", "对",
}

_PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(万|w|k|千|元|块|rmb|￥|¥)?", re.IGNORECASE)


def normalize(text: str) -> str:
    """全角转半角、去多余空白、统一小写。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def init_tokenizer() -> None:
    global _INITIALIZED
    if not _INITIALIZED:
        jieba.initialize()
        _INITIALIZED = True


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    init_tokenizer()
    text = normalize(text)
    if not text:
        return []
    out: list[str] = []
    for tok in jieba.cut_for_search(text):
        tok = tok.strip().lower()
        if not tok or _PUNCT_RE.match(tok):
            continue
        if not keep_stopwords and tok in STOPWORDS:
            continue
        out.append(tok)
    return out


def char_ngrams(text: str, n: int = 2) -> list[str]:
    text = re.sub(r"\s+", "", normalize(text))
    if len(text) < n:
        return [text] if text else []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def stable_hash(token: str) -> int:
    """跨进程稳定的 hash，不能用内置 hash()（带随机种子）。"""
    return zlib.crc32(token.encode("utf-8"))


def parse_price(text: str) -> float | None:
    """从'5000以内''5k左右''预算一万''￥4,999'里取第一个价格数字。"""
    if not text:
        return None
    cleaned = re.sub(r"[,，\s]", "", normalize(text))
    m = _PRICE_RE.search(cleaned)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit in ("万", "w"):
        value *= 10000
    elif unit in ("k", "千"):
        value *= 1000
    return value


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
