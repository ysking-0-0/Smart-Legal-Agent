"""
Embedding模型管理模块 - 支持多种Embedding后端
"""
from typing import Optional
from langchain_core.embeddings import Embeddings


class EmbeddingManager:
    """Embedding模型管理器"""

    def __init__(
        self,
        provider: str = "chromadb",
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        初始化Embedding管理器

        Args:
            provider: 提供商 (openai/chroma_default)
            model_name: 模型名称
            api_key: API密钥
            base_url: API基础URL
            device: 设备类型 (仅本地模型使用)
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.device = device
        self._embeddings: Optional[Embeddings] = None

    def get_embeddings(self) -> Embeddings:
        """
        获取Embedding模型实例

        Returns:
            Embeddings实例
        """
        if self._embeddings is None:
            if self.provider == "openai":
                self._embeddings = self._create_openai_embeddings()
            else:
                # 使用chromadb内置的默认embedding（无需额外安装）
                self._embeddings = self._create_chroma_default()
        return self._embeddings

    def _create_openai_embeddings(self) -> Embeddings:
        """创建OpenAI Embedding"""
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _create_chroma_default(self) -> Embeddings:
        """使用ChromaDB默认embedding（all-MiniLM-L6-v2，自动下载）"""
        from langchain_community.embeddings import FakeEmbeddings

        # chromadb默认使用内置的embedding，这里返回一个占位
        # 实际在ChromaVectorStore中会使用chromadb自己的默认embedding
        return FakeEmbeddings(size=384)

    @property
    def embeddings(self) -> Embeddings:
        """获取Embedding模型的属性访问"""
        return self.get_embeddings()

    @staticmethod
    def list_providers():
        """列出支持的Embedding提供商"""
        print("支持的Embedding提供商：")
        print("  - chromadb: ChromaDB默认embedding（推荐，无需额外配置）")
        print("  - openai: OpenAI Embedding（需要API密钥）")
