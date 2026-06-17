"""
法律RAG链 - 整合检索和生成
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

from ..vector_store.chroma_store import ChromaVectorStore
from ..retrieval.hybrid_retriever import HybridRetriever
from ..llm.legal_llm import LegalLLM


@dataclass
class RAGResponse:
    """RAG响应数据类"""
    answer: str
    sources: List[Dict[str, Any]]
    web_search_used: bool
    query: str
    is_legal_query: bool  # 是否是法律查询


class LegalRAGChain:
    """法律RAG链"""

    # 法律相关关键词
    LEGAL_KEYWORDS = [
        "法律", "法规", "法条", "条款", "规定", "条例", "办法",
        "民法", "刑法", "行政法", "商法", "劳动法", "合同法",
        "交通", "事故", "赔偿", "责任", "权利", "义务",
        "起诉", "诉讼", "判决", "仲裁", "调解",
        "犯罪", "刑罚", "处罚", "罚款", "拘留",
        "合同", "协议", "契约", "签字", "盖章",
        "离婚", "继承", "遗产", "婚姻", "家庭",
        "劳动", "用工", "解雇", "辞退", "工资",
        "租房", "买房", "房产", "产权",
        "闯红灯", "违章", "罚款", "扣分",
    ]

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        llm: LegalLLM,
        top_k: int = 5,
        use_web_search: bool = True,
    ):
        """
        初始化法律RAG链

        Args:
            vector_store: 向量存储实例
            llm: LLM实例
            top_k: 检索结果数量
            use_web_search: 是否启用联网搜索
        """
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.use_web_search = use_web_search

        # 创建检索器
        self.retriever = HybridRetriever(
            vector_store=vector_store,
            top_k=top_k,
            use_web_search=use_web_search
        )

    def _is_legal_query(self, question: str) -> bool:
        """
        判断是否是法律相关查询

        Args:
            question: 用户问题

        Returns:
            是否是法律查询
        """
        question_lower = question.lower()
        for keyword in self.LEGAL_KEYWORDS:
            if keyword in question_lower:
                return True
        return False

    def query(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> RAGResponse:
        """
        查询法律问题

        Args:
            question: 用户问题
            chat_history: 聊天历史

        Returns:
            RAG响应
        """
        # 判断是否是法律查询
        is_legal = self._is_legal_query(question)

        if is_legal:
            # 法律查询：检索知识库
            relevant_docs = self.retriever.invoke(question)
            context = self._build_context(relevant_docs)
            answer = self.llm.generate_answer(
                query=question,
                context=context,
                chat_history=chat_history,
                is_legal_query=True
            )
            sources = [
                {
                    "content": doc.page_content[:500],
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", ""),
                    "score": doc.metadata.get("score", 0),
                    "retrieval_method": doc.metadata.get("retrieval_method", "unknown"),
                }
                for doc in relevant_docs
            ]
            web_search_used = any(
                doc.metadata.get("retrieval_method") == "web_search"
                for doc in relevant_docs
            )
        else:
            # 普通查询：直接用大模型回答
            answer = self.llm.generate_answer(
                query=question,
                context="",
                chat_history=chat_history,
                is_legal_query=False
            )
            sources = []
            web_search_used = False

        return RAGResponse(
            answer=answer,
            sources=sources,
            web_search_used=web_search_used,
            query=question,
            is_legal_query=is_legal,
        )

    def _build_context(self, documents: List) -> str:
        """
        构建上下文

        Args:
            documents: 文档列表

        Returns:
            上下文字符串
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            context_parts.append(f"【来源{i}】{source}\n{doc.page_content}\n")
        return "\n".join(context_parts)

    def get_retrieval_sources(self, question: str) -> Dict[str, Any]:
        """
        获取检索来源详情（用于调试）

        Args:
            question: 用户问题

        Returns:
            检索来源详情
        """
        return self.retriever.get_retrieval_sources(question)
