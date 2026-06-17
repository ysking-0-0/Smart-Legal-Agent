"""
ChromaDB向量存储封装
"""
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document


class ChromaVectorStore:
    """ChromaDB向量存储封装"""

    def __init__(
        self,
        persist_dir: str = "./vector_db",
        collection_name: str = "legal_documents",
        embedding_provider: str = "chromadb",
        embedding_model: str = "text-embedding-3-small",
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_device: str = "cpu"
    ):
        """
        初始化ChromaDB向量存储

        Args:
            persist_dir: 持久化目录
            collection_name: 集合名称
            embedding_provider: Embedding提供商 (chromadb/openai)
            embedding_model: Embedding模型名称
            embedding_api_key: API密钥
            embedding_base_url: API基础URL
            embedding_device: 设备类型
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider

        # 确保目录存在
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # 初始化ChromaDB客户端
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # 初始化Embedding函数
        self._embedding_function = self._get_embedding_function(
            provider=embedding_provider,
            model=embedding_model,
            api_key=embedding_api_key,
            base_url=embedding_base_url,
        )

        # 初始化向量存储
        self._vector_store: Optional[Chroma] = None

    def _get_embedding_function(self, provider, model, api_key, base_url):
        """获取Embedding函数"""
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        elif provider == "bge":
            from .bge_embedding import BGEEmbedding
            return BGEEmbedding(
                model_name=model or "BAAI/bge-small-zh-v1.5",
                device=getattr(self, 'embedding_device', 'cpu'),
            )
        elif provider == "lightweight":
            from .lightweight_embedding import LightweightEmbedding
            return LightweightEmbedding(n_features=384)
        else:
            # 使用ChromaDB默认的embedding (all-MiniLM-L6-v2)
            # chromadb会自动下载和使用默认的embedding模型
            return None  # 让chroma使用默认embedding

    def get_vector_store(self) -> Chroma:
        """
        获取Chroma向量存储实例

        Returns:
            Chroma实例
        """
        if self._vector_store is None:
            kwargs = {
                "client": self._client,
                "collection_name": self.collection_name,
            }
            if self._embedding_function is not None:
                kwargs["embedding_function"] = self._embedding_function

            self._vector_store = Chroma(**kwargs)
        return self._vector_store

    @property
    def vector_store(self) -> Chroma:
        """获取向量存储的属性访问"""
        return self.get_vector_store()

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        添加文档到向量存储

        Args:
            documents: 文档列表
            batch_size: 批量大小

        Returns:
            文档ID列表
        """
        if not documents:
            return []

        vector_store = self.get_vector_store()
        all_ids = []

        # 分批添加
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = vector_store.add_documents(batch)
            all_ids.extend(ids)
            print(f"[OK] 已添加 {len(all_ids)}/{len(documents)} 个文档块")

        return all_ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件

        Returns:
            相似文档列表
        """
        vector_store = self.get_vector_store()
        return vector_store.similarity_search(
            query,
            k=k,
            filter=filter_dict
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        相似度搜索（带分数）

        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: 过滤条件

        Returns:
            (文档, 分数) 列表
        """
        vector_store = self.get_vector_store()
        return vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict
        )

    def delete_collection(self):
        """删除集合"""
        try:
            self._client.delete_collection(self.collection_name)
            print(f"[OK] 已删除集合: {self.collection_name}")
        except Exception:
            pass  # 集合不存在时忽略
        self._vector_store = None

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        获取集合统计信息

        Returns:
            统计信息字典
        """
        try:
            collection = self._client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "persist_dir": str(self.persist_dir),
            }
        except Exception:
            return {
                "name": self.collection_name,
                "count": 0,
                "persist_dir": str(self.persist_dir),
            }
