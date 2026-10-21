"""Dense embedding (BGE) + FAISS indexes (Flat / IVF / HNSW)."""
from __future__ import annotations

import hashlib
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

if sys.platform == "darwin":  # macOS: torch 与 faiss 各自带 libomp，避免 OMP 冲突导致的段错误
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    """bge-large-zh-v1.5 (GPU 推荐) / bge-small-zh-v1.5 (CPU 演示). Embeddings are L2-normalized."""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5", device: Optional[str] = None,
                 batch_size: int = 32, cache_dir: Optional[str] = None, query_instruction: str = BGE_QUERY_INSTRUCTION):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.bs = batch_size
        self.cache_dir = cache_dir
        self.qi = query_instruction if "bge" in model_name.lower() and "m3" not in model_name.lower() else ""

    def _cache_path(self, texts: List[str]) -> Optional[str]:
        if not self.cache_dir:
            return None
        h = hashlib.md5(("\x00".join(texts) + self.model_name).encode("utf-8")).hexdigest()[:16]
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"emb_{h}.npy")

    def encode_docs(self, texts: List[str]) -> np.ndarray:
        p = self._cache_path(texts)
        if p and os.path.exists(p):
            return np.load(p)
        e = self.model.encode(texts, batch_size=self.bs, normalize_embeddings=True, show_progress_bar=len(texts) > 500,
                              convert_to_numpy=True).astype("float32")
        if p:
            np.save(p, e)
        return e

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return self.model.encode([self.qi + q for q in queries], batch_size=self.bs, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")


class VectorIndex:
    def __init__(self, dim: int, kind: str = "flat", nlist: int = 64, nprobe: int = 8, hnsw_m: int = 32,
                 ef_search: int = 64, ef_construction: int = 200):
        import faiss

        if sys.platform == "darwin" or os.environ.get("FAISS_THREADS"):
            faiss.omp_set_num_threads(int(os.environ.get("FAISS_THREADS", 1)))
        self.faiss, self.kind = faiss, kind
        if kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "ivf":
            quant = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.nprobe = nprobe
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(kind)

    def build(self, vecs: np.ndarray):
        if self.kind == "ivf":
            self.index.train(vecs)
            self.index.nprobe = self.nprobe
        self.index.add(vecs)
        return self

    def search(self, q: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if q.ndim == 1:
            q = q[None, :]
        return self.index.search(q, k)

    def memory_mb(self) -> float:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as f:
            path = f.name
        self.faiss.write_index(self.index, path)
        size = os.path.getsize(path) / 1024 ** 2
        os.remove(path)
        return size

    def save(self, path: str):
        self.faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str):
        import faiss

        obj = cls.__new__(cls)
        obj.faiss, obj.kind, obj.index = faiss, "loaded", faiss.read_index(path)
        return obj
