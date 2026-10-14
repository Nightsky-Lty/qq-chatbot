"""长期记忆向量后端：本地特征哈希与可选的 BGE ONNX CPU 推理。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Protocol

from app.config import Settings

logger = logging.getLogger(__name__)
HASH_MODEL_ID = "hash-blake2b-512-v1"


class EmbeddingUnavailable(RuntimeError):
    """模型未部署、加载失败或单次 ONNX 推理失败。"""


def tokens(text: str) -> list[str]:
    """生成适合中英文短聊天的词元；中文使用单字和双字片段。"""
    lowered = text.casefold()
    result = re.findall(r"[a-z0-9_]+", lowered)
    for chunk in re.findall(r"[\u3400-\u9fff]+", lowered):
        result.extend(chunk)
        result.extend(chunk[index:index + 2] for index in range(len(chunk) - 1))
    return result


def sparse_vector(text: str, dimensions: int = 512) -> str:
    """生成 L2 归一化的本地特征哈希向量。"""
    counts: Counter[int] = Counter()
    for token in tokens(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        counts[int.from_bytes(digest, "big") % dimensions] += 1
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    return json.dumps({str(key): value / norm for key, value in counts.items()}, separators=(",", ":"))


def dense_vector(values: object) -> str:
    """将已经归一化的稠密向量紧凑序列化；SQLite 保持后端无关。"""
    return json.dumps([round(float(value), 8) for value in values], separators=(",", ":"))


def vector_similarity(left: str, right: str) -> float:
    """同时计算稀疏字典或稠密数组的点积；输入均已 L2 归一化。"""
    try:
        a = json.loads(left)
        b = json.loads(right)
    except (TypeError, ValueError):
        return 0.0
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return 0.0
        return sum(float(x) * float(y) for x, y in zip(a, b))
    if not isinstance(a, dict) or not isinstance(b, dict):
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(float(value) * float(b.get(key, 0.0)) for key, value in a.items())


class EmbeddingProvider(Protocol):
    model_id: str
    dimensions: int
    available: bool

    async def start(self) -> bool: ...

    async def embed_documents(self, texts: list[str]) -> list[str]: ...

    async def embed_query(self, text: str) -> str: ...


class HashEmbeddingProvider:
    model_id = HASH_MODEL_ID
    dimensions = 512
    available = True

    async def start(self) -> bool:
        return True

    async def embed_documents(self, texts: list[str]) -> list[str]:
        return [sparse_vector(text) for text in texts]

    async def embed_query(self, text: str) -> str:
        return sparse_vector(text)


class BgeOnnxEmbeddingProvider:
    """惰性加载本地 BGE ONNX 模型；推理放在线程中，且一次只执行一个批次。"""

    model_id = "BAAI/bge-small-zh-v1.5-onnx-int8"

    def __init__(self, settings: Settings):
        self.path = Path(settings.embedding_model_path)
        self.file_name = settings.embedding_model_file
        self.max_length = settings.embedding_max_length
        self.threads = settings.embedding_threads
        self.query_prefix = settings.embedding_query_prefix
        self.dimensions = 0
        self.available = False
        self._attempted = False
        self._session = None
        self._tokenizer = None
        self._thread_lock = threading.Lock()
        self._async_lock = asyncio.Lock()

    async def start(self) -> bool:
        if self._attempted:
            return self.available
        self._attempted = True
        try:
            await asyncio.to_thread(self._load)
            self.available = True
            logger.info("Embedding model ready: model=%s dimensions=%s", self.model_id, self.dimensions)
        except Exception as exc:
            self.available = False
            logger.warning(
                "Embedding model unavailable; using hash fallback: error=%s detail=%s",
                type(exc).__name__, exc,
            )
        return self.available

    def _load(self) -> None:
        if not self.path.is_dir():
            raise EmbeddingUnavailable("模型目录不存在")
        model_path = self.path / self.file_name
        if not model_path.is_file():
            alternatives = [self.path / "model_quantized.onnx", self.path / "model.onnx"]
            model_path = next((item for item in alternatives if item.is_file()), model_path)
        if not model_path.is_file() or not (self.path / "tokenizer.json").is_file():
            raise EmbeddingUnavailable("模型或Tokenizer文件不完整")
        import onnxruntime as ort
        from transformers import AutoTokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        # 先用一条短文本确认输入输出结构，并记录真实维度。
        self.dimensions = len(json.loads(self._embed_sync(["模型预热"])[0]))

    def _embed_sync(self, texts: list[str]) -> list[str]:
        if self._session is None or self._tokenizer is None:
            raise EmbeddingUnavailable("模型尚未加载")
        import numpy as np

        with self._thread_lock:
            encoded = self._tokenizer(
                texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="np"
            )
            input_names = {item.name for item in self._session.get_inputs()}
            if "token_type_ids" in input_names and "token_type_ids" not in encoded:
                encoded["token_type_ids"] = np.zeros_like(encoded["input_ids"])
            inputs = {
                name: np.asarray(encoded[name], dtype=np.int64)
                for name in input_names if name in encoded
            }
            if "input_ids" not in inputs or "attention_mask" not in inputs:
                raise EmbeddingUnavailable("ONNX模型输入不兼容")
            outputs = self._session.run(None, inputs)
            names = [item.name for item in self._session.get_outputs()]
            index = names.index("last_hidden_state") if "last_hidden_state" in names else 0
            hidden = np.asarray(outputs[index])
            if hidden.ndim != 3:
                raise EmbeddingUnavailable("ONNX模型输出不兼容")
            if hidden.shape[0] != len(texts):
                raise EmbeddingUnavailable("ONNX模型输出批次大小不匹配")
            vectors = hidden[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
            return [dense_vector(vector) for vector in vectors]

    async def embed_documents(self, texts: list[str]) -> list[str]:
        if not await self.start():
            raise EmbeddingUnavailable("BGE模型不可用")
        async with self._async_lock:
            return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, text: str) -> str:
        values = await self.embed_documents([self.query_prefix + text])
        return values[0]


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "bge_onnx":
        return BgeOnnxEmbeddingProvider(settings)
    return HashEmbeddingProvider()
