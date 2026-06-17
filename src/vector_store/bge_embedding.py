"""
BGE 中文 Embedding — 基于 BAAI/bge-small-zh-v1.5
兼容 ChromaDB EmbeddingFunction 协议

安装依赖：
    pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

首次使用会自动下载模型（约 95MB），支持 HF 镜像加速：
    set HF_ENDPOINT=https://hf-mirror.com
"""
import os
from typing import List, Optional
import numpy as np


class BGEEmbedding:
    """
    BGE-small-zh-v1.5 中文 Embedding 函数
    - 向量维度：512
    - 模型大小：约 95MB
    - 专为中文法律/通用文本优化
    - 查询时自动添加 BGE 指令前缀
    """

    # BGE 模型推荐的查询前缀
    QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace 模型名或本地路径
            device: "cpu" 或 "cuda"
            normalize: 是否 L2 归一化（BGE 推荐开启）
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model = None

    def _load_model(self):
        """懒加载模型 — 优先 ModelScope，回退 HF 镜像"""
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        model_path = self.model_name

        # ── 尝试从 ModelScope 下载（国内更快） ──
        try:
            from modelscope import snapshot_download
            print(f"[BGE] 尝试从 ModelScope 下载 {self.model_name}...")
            model_path = snapshot_download(self.model_name)
            print(f"[BGE] ModelScope 下载完成: {model_path}")
        except Exception as e:
            print(f"[BGE] ModelScope 下载失败 ({e})，尝试 HF 镜像...")
            # 回退：使用 HF 镜像
            if "HF_ENDPOINT" not in os.environ:
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        print(f"[BGE] 加载模型 (device={self.device})...")
        self._model = SentenceTransformer(
            model_path,
            device=self.device,
        )
        self._model.max_seq_length = 512
        print(f"[BGE] 模型加载完成，向量维度: {self._model.get_embedding_dimension()}")
        return self._model

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """编码文本为向量"""
        model = self._load_model()

        if is_query:
            # 查询时添加 BGE 指令前缀
            texts = [self.QUERY_INSTRUCTION + t for t in texts]

        embeddings = model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=32,
        )
        return embeddings

    # ── ChromaDB EmbeddingFunction 协议 ──

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB 调用入口 — 文档侧 embedding（不加查询前缀）

        Args:
            input: 文档文本列表

        Returns:
            向量列表
        """
        embeddings = self._encode(input, is_query=False)
        return embeddings.tolist()

    # ── LangChain 兼容接口 ──

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """LangChain 文档 embedding"""
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        """LangChain 查询 embedding（带 BGE 指令前缀）"""
        embeddings = self._encode([text], is_query=True)
        return embeddings[0].tolist()

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        """批量查询 embedding（带 BGE 指令前缀）"""
        embeddings = self._encode(texts, is_query=True)
        return embeddings.tolist()
