"""
轻量 Embedding — 基于 sklearn HashingVectorizer
零模型下载，纯 CPU，立即可用
"""
from typing import List
import numpy as np


class LightweightEmbedding:
    """
    基于 sklearn HashingVectorizer 的轻量 Embedding 函数
    兼容 ChromaDB 的 EmbeddingFunction 协议

    优点：不需要下载任何模型文件，立即可用
    缺点：语义理解弱于 BGE/ONNX 模型，但关键词匹配精度尚可
    适用：快速验证 Agent 架构，后续可替换为更强的 Embedding
    """

    def __init__(self, n_features: int = 384):
        """
        Args:
            n_features: 输出向量维度，默认 384（与 MiniLM 一致）
        """
        self.n_features = n_features
        self._vectorizer = None

    def _get_vectorizer(self):
        """懒加载 HashingVectorizer"""
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import HashingVectorizer
            self._vectorizer = HashingVectorizer(
                n_features=self.n_features,
                alternate_sign=False,
                norm='l2',
                analyzer='char_wb',  # 字符级 n-gram，适合中文
                ngram_range=(2, 4),
            )
        return self._vectorizer

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB EmbeddingFunction 协议

        Args:
            input: 文本列表

        Returns:
            向量列表，每个向量是 float 列表
        """
        vectorizer = self._get_vectorizer()
        matrix = vectorizer.transform(input)
        # 转为 List[List[float]]
        return matrix.toarray().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """与 LangChain embedding 接口兼容"""
        return self(texts)

    def embed_query(self, text: str) -> List[float]:
        """与 LangChain embedding 接口兼容"""
        return self([text])[0]
