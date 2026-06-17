"""
Agent 工具层 — 将检索能力封装为 Agent 可调用的独立工具

v2: 增加执行耗时日志 + 每步 try-except 保护
"""
import time
from typing import List, Dict, Any
from langchain_core.documents import Document


class AgentTools:
    """Agent 工具集：search_laws / search_cases / search_web"""

    def __init__(self, vector_store, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    @staticmethod
    def tool_definitions() -> List[Dict[str, Any]]:
        return [
            {
                "name": "search_laws",
                "description": (
                    "搜索法律法条知识库，查找相关的法律规定、法条原文、司法解释。"
                    "适用于：查找具体法律条款、法规内容、违法行为的法律定性。"
                    "参数 query 应该是法律关键词或问题描述，如 '醉酒驾驶处罚'、'劳动合同解除条件'。"
                ),
                "parameters": {"query": "string — 搜索查询字符串"},
            },
            {
                "name": "search_cases",
                "description": (
                    "搜索案例库，查找与当前问题相似的司法案例、判例、裁判文书。"
                    "适用于：查找类似案件的判决结果、量刑参考、法院观点。"
                    "参数 query 应该是案情关键词，如 '危险驾驶罪 醉驾 量刑'、'劳动纠纷 违法解除 赔偿'。"
                ),
                "parameters": {"query": "string — 搜索查询字符串"},
            },
            {
                "name": "search_web",
                "description": (
                    "联网搜索补充信息，当知识库中没有相关内容或需要查询最新法律法规时使用。"
                    "适用于：查询最新颁布的法律、社会热点法律事件、知识库中未收录的内容。"
                ),
                "parameters": {"query": "string — 搜索查询字符串"},
            },
        ]

    # ── 工具执行（带超时保护） ──

    def execute(self, tool_name: str, query: str, timeout_sec: int = 30) -> str:
        """
        执行工具调用，带超时保护。

        用 ThreadPoolExecutor 包裹执行，防止 BGE 加载或 ChromaDB 查询
        在网络异常时无限阻塞。
        """
        import concurrent.futures

        def _run():
            if tool_name == "search_laws":
                return self._search_laws(query)
            elif tool_name == "search_cases":
                return self._search_cases(query)
            elif tool_name == "search_web":
                return self._search_web(query)
            else:
                return f"错误：未知工具 '{tool_name}'，可用工具：search_laws, search_cases, search_web"

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run)
                return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            print(f"[TIMEOUT] 工具 '{tool_name}' 执行超过 {timeout_sec}s，已取消")
            return f"[错误] 知识库检索超时（>{timeout_sec}秒），请稍后重试或尝试更简洁的查询词。"
        except Exception as e:
            print(f"[ERROR] 工具 '{tool_name}' 执行异常: {e}")
            return f"[错误] 工具执行失败: {str(e)}"

    # ── search_laws ──

    def _search_laws(self, query: str) -> str:
        try:
            t0 = time.time()

            # 1. 向量检索（BGE 编码 + ChromaDB 查询）
            t1 = time.time()
            results = self.vector_store.similarity_search_with_score(
                query, k=self.top_k * 2
            )
            t2 = time.time()
            print(f"[search_laws] 向量检索完成: {len(results)} 条, "
                  f"BGE编码={t1-t0:.1f}s, DB查询={t2-t1:.1f}s")

            # 2. 后过滤
            filtered = []
            for doc, score in results:
                doc_type = doc.metadata.get("doc_type", "")
                source = doc.metadata.get("source", "")
                if doc_type in ("law", "custom") or (
                    not doc_type and "cases" not in source.lower()
                ):
                    filtered.append((doc, score))
                    if len(filtered) >= self.top_k:
                        break

            if not filtered:
                print(f"[search_laws] 无匹配结果 (query='{query[:50]}')")
                return "未找到相关法律条文。"

            formatted = self._format_results(filtered, "法律条文")
            print(f"[search_laws] 过滤后 {len(filtered)} 条, 总耗时 {time.time()-t0:.1f}s")
            return formatted

        except Exception as e:
            print(f"[search_laws] 异常: {e}")
            return f"搜索法律条文时出错：{str(e)}"

    # ── search_cases ──

    def _search_cases(self, query: str) -> str:
        try:
            t0 = time.time()

            results = self.vector_store.similarity_search_with_score(
                query, k=self.top_k * 2
            )

            t1 = time.time()
            print(f"[search_cases] 向量检索: {len(results)} 条, {t1-t0:.1f}s")

            filtered = []
            for doc, score in results:
                doc_type = doc.metadata.get("doc_type", "")
                source = doc.metadata.get("source", "")
                if doc_type == "case" or (
                    not doc_type and "cases" in source.lower()
                ):
                    filtered.append((doc, score))
                    if len(filtered) >= self.top_k:
                        break

            if not filtered:
                return "未找到相关案例。案例库可能尚未收录此类案件。"

            return self._format_results(filtered, "案例")

        except Exception as e:
            print(f"[search_cases] 异常: {e}")
            return f"搜索案例时出错：{str(e)}"

    # ── search_web ──

    def _search_web(self, query: str) -> str:
        try:
            t0 = time.time()
            from duckduckgo_search import DDGS

            legal_query = f"{query} 法律 法规"
            results = []
            with DDGS() as ddgs:
                search_results = list(ddgs.text(legal_query, max_results=3))
                for i, r in enumerate(search_results, 1):
                    results.append(
                        f"【搜索结果{i}】{r.get('title', '')}\n"
                        f"来源：{r.get('href', '')}\n"
                        f"内容：{r.get('body', '')}\n"
                    )

            print(f"[search_web] {len(results)} 条, {time.time()-t0:.1f}s")

            if not results:
                return "联网搜索未找到相关内容。"
            return "\n".join(results)

        except ImportError:
            return "联网搜索功能未安装（需要 duckduckgo_search 包）。"
        except Exception as e:
            print(f"[search_web] 异常: {e}")
            return f"联网搜索失败：{str(e)}"

    # ── 格式化 ──

    def _format_results(self, scored_docs: List[tuple], doc_type: str) -> str:
        parts = []
        for i, (doc, score) in enumerate(scored_docs, 1):
            source = doc.metadata.get("source", "未知来源")
            title = doc.metadata.get("title", "")
            content = doc.page_content[:600]
            if len(doc.page_content) > 600:
                content += "..."
            parts.append(
                f"【{doc_type}{i}】相似度 {score:.3f}\n"
                f"来源：{source}\n"
                + (f"标题：{title}\n" if title else "")
                + f"内容：\n{content}\n"
            )
        return "\n".join(parts)
