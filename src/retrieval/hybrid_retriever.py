"""
混合检索器 — 密集向量检索 (BGE) + 稀疏关键词检索 (BM25) + RRF 融合

架构：
  查询 → ┌─ Dense:  ChromaDB + BGE embedding  → scores_dense
         └─ Sparse: BM25Okapi 关键词匹配       → scores_sparse
              ↓
         RRF 融合 (Reciprocal Rank Fusion, k=60)
              ↓
         Top-K 结果
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..vector_store.chroma_store import ChromaVectorStore


def _chinese_tokenizer(text: str) -> List[str]:
    """
    中文法律文本分词 — 字符级 + 数字/英文保留

    法律文本特点：
    - 关键词短："第X条"、"拘役"、"罚款"、"危险驾驶罪"
    - 法条编号："第一百三十三条之一"、"第91条"
    - 字符级分词对这些模式匹配最友好
    """
    tokens = []
    # 匹配：中文字符、连续数字、连续英文字母
    for match in re.finditer(r'[\u4e00-\u9fff]|[0-9]+|[a-zA-Z]+', text):
        token = match.group()
        tokens.append(token.lower())
    return tokens


class HybridRetriever(BaseRetriever):
    """
    混合检索器 — Dense + Sparse + RRF

    使用示例：
        retriever = HybridRetriever(vector_store=vs, top_k=5)
        docs = retriever.invoke("酒后驾驶怎么处罚")
    """

    vector_store: ChromaVectorStore
    top_k: int = 5
    dense_weight: float = 0.6   # 密集检索权重（RRF 融合中的相对权重）
    sparse_weight: float = 0.4  # 稀疏检索权重
    rrf_k: int = 60             # RRF 平滑常数
    use_web_search: bool = True
    web_search_max_results: int = 3

    # BM25 索引（懒加载）
    _bm25_index: Any = None
    _bm25_docs: List[Document] = None
    _bm25_texts: List[str] = None

    class Config:
        arbitrary_types_allowed = True

    # ── 主检索入口 ──

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        混合检索

        1. Dense: ChromaDB 向量检索 (BGE)
        2. Sparse: BM25 关键词检索
        3. RRF 融合
        4. 不足时联网补搜
        """
        # 1. Dense 检索
        dense_results = self._dense_search(query)

        # 2. Sparse 检索
        sparse_results = self._sparse_search(query)

        # 3. RRF 融合
        fused = self._rrf_fusion(dense_results, sparse_results, self.top_k)

        # 4. 不足时联网补搜
        if self.use_web_search and len(fused) < self.top_k:
            web_results = self._web_search(query)
            fused.extend(web_results[: self.top_k - len(fused)])

        return fused[: self.top_k]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """异步检索（使用同步实现）"""
        return self._get_relevant_documents(query)

    # ── Dense 检索 ──

    def _dense_search(self, query: str) -> List[Tuple[Document, float]]:
        """
        ChromaDB 向量检索

        Returns:
            [(doc, similarity_score), ...] — 分数越高越相关
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query, k=self.top_k * 3  # 多召回一些给融合留空间
            )
            # ChromaDB 返回的是 distance（越小越好），转为相似度
            scored = []
            for doc, distance in results:
                similarity = 1.0 / (1.0 + distance)  # distance → similarity
                doc.metadata["dense_score"] = float(similarity)
                doc.metadata["retrieval_method"] = "vector"
                scored.append((doc, similarity))
            return scored
        except Exception as e:
            print(f"[Dense] 检索失败: {e}")
            return []

    # ── Sparse 检索 (BM25) ──

    def _sparse_search(self, query: str) -> List[Tuple[Document, float]]:
        """
        BM25 关键词检索

        Returns:
            [(doc, bm25_score), ...] — 分数越高越相关
        """
        try:
            index, docs = self._get_bm25_index()
            if index is None or not docs:
                return []

            query_tokens = _chinese_tokenizer(query)
            scores = index.get_scores(query_tokens)

            # 取 top_k * 3
            top_n = min(self.top_k * 3, len(scores))
            if top_n == 0:
                return []

            # 获取 top-N 索引
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:top_n]

            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # 只返回有分数的
                    doc = docs[idx]
                    doc.metadata["bm25_score"] = float(scores[idx])
                    doc.metadata["retrieval_method"] = "bm25"
                    results.append((doc, float(scores[idx])))

            return results
        except Exception as e:
            print(f"[Sparse] BM25 检索失败: {e}")
            return []

    def _get_bm25_index(self):
        """
        懒加载 BM25 索引 — 从 ChromaDB 加载所有文档并构建

        Returns:
            (BM25Okapi 实例, documents 列表)
        """
        if self._bm25_index is not None:
            return self._bm25_index, self._bm25_docs

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("[BM25] rank_bm25 未安装，跳过稀疏检索。安装: pip install rank_bm25")
            return None, []

        print("[BM25] 构建索引...")

        # 从 ChromaDB 获取所有文档
        try:
            all_docs = self.vector_store.similarity_search(
                "",  # 空查询获取全部
                k=10000  # 足够大的 k
            )
        except Exception:
            # ChromaDB 不支持空查询，换一种方式
            collection = self.vector_store._client.get_collection(
                self.vector_store.collection_name
            )
            all_data = collection.get()
            all_docs = [
                Document(page_content=text, metadata=meta or {})
                for text, meta in zip(
                    all_data.get("documents", []),
                    all_data.get("metadatas", []),
                )
            ]

        if not all_docs:
            print("[BM25] 知识库为空，跳过")
            return None, []

        # 分词
        tokenized = [_chinese_tokenizer(doc.page_content) for doc in all_docs]

        # 构建 BM25
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_docs = all_docs
        self._bm25_texts = [doc.page_content for doc in all_docs]

        print(f"[BM25] 索引构建完成，共 {len(all_docs)} 个文档")
        return self._bm25_index, self._bm25_docs

    # ── RRF 融合 ──

    def _rrf_fusion(
        self,
        dense: List[Tuple[Document, float]],
        sparse: List[Tuple[Document, float]],
        top_k: int,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion — 鲁棒匹配版

        问题：Dense 和 Sparse 可能返回不同的 Document 对象，即使内容相同。
        解决：用归一化后的内容前 200 字符做匹配键，而非依赖于对象 identity。
        """
        if not dense and not sparse:
            return []
        if not dense:
            return [doc for doc, _ in sparse[:top_k]]
        if not sparse:
            return [doc for doc, _ in dense[:top_k]]

        import hashlib

        def _norm_key(text: str) -> str:
            """归一化文本用于匹配：去空白 + 取前 200 字符 MD5"""
            cleaned = re.sub(r'\s+', '', text)[:200]
            return hashlib.md5(cleaned.encode("utf-8")).hexdigest()

        # ── 第一阶段：收集 Dense 排名 ──
        dense_ranks: Dict[str, int] = {}    # key -> dense_rank
        key_to_doc: Dict[str, Document] = {}  # key -> best Document

        for rank, (doc, _) in enumerate(dense, 1):
            key = _norm_key(doc.page_content)
            if key not in dense_ranks:
                dense_ranks[key] = rank
                key_to_doc[key] = doc
                doc.metadata["dense_rank"] = rank

        # ── 第二阶段：收集 Sparse 排名 ──
        sparse_ranks: Dict[str, int] = {}   # key -> sparse_rank

        for rank, (doc, _) in enumerate(sparse, 1):
            key = _norm_key(doc.page_content)
            if key not in sparse_ranks:
                sparse_ranks[key] = rank
                if key not in key_to_doc:
                    key_to_doc[key] = doc
                key_to_doc[key].metadata["sparse_rank"] = rank

        # ── 第三阶段：RRF 融合 ──
        all_keys = set(dense_ranks.keys()) | set(sparse_ranks.keys())
        rrf_scores: Dict[str, float] = {}

        for key in all_keys:
            score = 0.0
            if key in dense_ranks:
                score += self.dense_weight / (self.rrf_k + dense_ranks[key])
            if key in sparse_ranks:
                score += self.sparse_weight / (self.rrf_k + sparse_ranks[key])
            rrf_scores[key] = score

        # ── 排序 ──
        sorted_keys = sorted(all_keys, key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for key in sorted_keys[:top_k]:
            doc = key_to_doc[key]
            doc.metadata["rrf_score"] = float(rrf_scores[key])
            doc.metadata["retrieval_method"] = "hybrid"
            results.append(doc)

        return results

    # ── 联网补搜 ──

    def _web_search(self, query: str) -> List[Document]:
        """DuckDuckGo 联网搜索"""
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                legal_query = f"{query} 法律 法规"
                search_results = list(ddgs.text(legal_query, max_results=self.web_search_max_results))
                for result in search_results:
                    doc = Document(
                        page_content=f"{result.get('title', '')}\n{result.get('body', '')}",
                        metadata={
                            "source": result.get('href', ''),
                            "title": result.get('title', ''),
                            "score": 0.3,
                            "retrieval_method": "web_search",
                            "file_type": "webpage",
                        }
                    )
                    results.append(doc)
            return results
        except ImportError:
            return []
        except Exception as e:
            print(f"[Web] 搜索失败: {e}")
            return []

    # ── 调试接口 ──

    def get_retrieval_sources(self, query: str) -> Dict[str, Any]:
        """获取检索来源详情（调试用）"""
        dense = self._dense_search(query)
        sparse = self._sparse_search(query)

        return {
            "dense": [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "unknown"),
                    "score": score,
                }
                for doc, score in dense[:5]
            ],
            "sparse": [
                {
                    "content": doc.page_content[:200],
                    "source": doc.metadata.get("source", "unknown"),
                    "score": score,
                }
                for doc, score in sparse[:5]
            ],
        }
