"""
FastAPI服务器 - 暴露法律RAG知识库API
"""
from typing import List, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..config import config
from ..vector_store.chroma_store import ChromaVectorStore
from ..llm.legal_llm import LegalLLM
from ..chain.legal_rag_chain import LegalRAGChain
from ..agent.legal_agent import LegalAgent


# 请求/响应模型
class ChatRequest(BaseModel):
    """聊天请求"""
    query: str
    use_web_search: bool = True
    chat_history: Optional[List[dict]] = None


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str
    sources: List[dict]
    web_search_used: bool
    query: str


class DocumentUploadRequest(BaseModel):
    """文档上传请求"""
    content: str
    title: str
    doc_type: str = "custom"


class DocumentUploadResponse(BaseModel):
    """文档上传响应"""
    success: bool
    message: str
    doc_count: int


class AgentChatRequest(BaseModel):
    """Agent 聊天请求"""
    query: str
    chat_history: Optional[List[dict]] = None


class AgentChatResponse(BaseModel):
    """Agent 聊天响应"""
    answer: str
    tool_calls_count: int
    steps_count: int
    query: str


class StatsResponse(BaseModel):
    """统计信息响应"""
    collection_name: str
    document_count: int
    persist_dir: str


# 全局变量
rag_chain: Optional[LegalRAGChain] = None
legal_agent: Optional[LegalAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_chain, legal_agent

    # 初始化组件
    print("[...] 正在初始化法律知识库...")

    # 初始化向量存储
    vector_store = ChromaVectorStore(
        persist_dir=config.vector_db_dir,
        collection_name=config.collection_name,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_api_key=config.openai_api_key,
        embedding_base_url=config.openai_base_url,
        embedding_device=config.embedding_device,
    )

    # 初始化LLM (RAG 和 Agent 共用一个实例)
    llm = LegalLLM(
        provider=config.llm_provider,
        model=config.openai_model if config.llm_provider == "openai" else config.ollama_model,
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
        ollama_base_url=config.ollama_base_url,
        request_timeout=60,
    )

    # 初始化RAG链
    rag_chain = LegalRAGChain(
        vector_store=vector_store,
        llm=llm,
        top_k=config.retrieval_top_k,
        use_web_search=config.web_search_enabled,
    )

    # 初始化 Agent (复用同一个 LLM 和 vector_store)
    legal_agent = LegalAgent(
        vector_store=vector_store,
        llm=llm,
        top_k=config.retrieval_top_k,
        max_steps=config.agent_max_steps,
        verbose=False,  # API 模式不打印中间步骤
    )

    print("[OK] 法律知识库 + Agent 初始化完成")

    yield

    # 清理资源
    print("[STOP] 正在关闭服务...")


def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title=config.get("api.title", "法律RAG知识库API"),
        description=config.get("api.description", "法律咨询与法规查询API服务"),
        version=config.get("api.version", "1.0.0"),
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        """根路径"""
        return {
            "name": "法律RAG知识库API",
            "version": "1.0.0",
            "status": "running",
        }

    @app.get("/health")
    async def health_check():
        """健康检查"""
        return {"status": "healthy"}

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        聊天接口 - 法律咨询

        Args:
            request: 聊天请求

        Returns:
            聊天响应
        """
        if rag_chain is None:
            raise HTTPException(status_code=503, detail="服务未初始化")

        try:
            response = rag_chain.query(
                question=request.query,
                chat_history=request.chat_history,
            )

            return ChatResponse(
                answer=response.answer,
                sources=response.sources,
                web_search_used=response.web_search_used,
                query=response.query,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/stats", response_model=StatsResponse)
    async def get_stats():
        """获取知识库统计信息"""
        if rag_chain is None:
            raise HTTPException(status_code=503, detail="服务未初始化")

        stats = rag_chain.vector_store.get_collection_stats()

        return StatsResponse(
            collection_name=stats["name"],
            document_count=stats["count"],
            persist_dir=stats["persist_dir"],
        )

    @app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
    async def upload_document(request: DocumentUploadRequest):
        """
        上传文档到知识库

        Args:
            request: 文档上传请求

        Returns:
            上传结果
        """
        if rag_chain is None:
            raise HTTPException(status_code=503, detail="服务未初始化")

        try:
            from langchain_core.documents import Document

            # 创建文档对象
            doc = Document(
                page_content=request.content,
                metadata={
                    "title": request.title,
                    "doc_type": request.doc_type,
                    "source": "api_upload",
                }
            )

            # 添加到向量存储
            rag_chain.vector_store.add_documents([doc])

            return DocumentUploadResponse(
                success=True,
                message="文档上传成功",
                doc_count=1,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/agent/chat", response_model=AgentChatResponse)
    async def agent_chat(request: AgentChatRequest):
        """
        Agent 聊天接口 — 多步推理法律助手

        与 /api/v1/chat 的区别：Agent 会自主分步检索法条和案例，
        而非一次性检索后生成回答。

        Args:
            request: Agent 聊天请求

        Returns:
            Agent 聊天响应
        """
        if legal_agent is None:
            raise HTTPException(status_code=503, detail="Agent 服务未初始化")

        try:
            response = legal_agent.run(
                query=request.query,
                chat_history=request.chat_history,
            )

            return AgentChatResponse(
                answer=response.answer,
                tool_calls_count=response.tool_calls_count,
                steps_count=len(response.steps),
                query=response.query,
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
