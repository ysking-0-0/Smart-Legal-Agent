"""
Agent 工具层 — 将检索能力封装为 Agent 可调用的独立工具

v2: 增加执行耗时日志 + 每步 try-except 保护
"""
import time
from typing import List, Dict, Any
from langchain_core.documents import Document


class AgentTools:
    """Agent 工具集：search_laws / search_cases / search_web / calculate"""

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
            {
                "name": "calculate",
                "description": (
                    "确定性法律费用计算器。严格按照《诉讼费用交纳办法》（国务院令第481号）"
                    "的阶梯费率表精确计算财产案件的诉讼费用，不依赖大模型口算。"
                    "适用于：计算诉讼费、案件受理费。"
                    "参数 query 应该是标的金额（数字），如 '300000' 或 '标的额30万'。"
                    "重要：涉及金钱计算的场景必须使用此工具，禁止自行估算或口算。"
                ),
                "parameters": {"query": "string — 标的金额（人民币元），如 300000"},
            },
        ]

    # ── 工具执行（带超时保护） ──

    def execute(self, tool_name: str, query: str, timeout_sec: int = 30) -> str:
        """
        执行工具调用，带超时保护 + JSON 解包。

        部分 LLM 输出 Action Input 为 JSON 格式如 {"query": "xxx"}，
        这里自动提取 query 字段。
        """
        import concurrent.futures

        # ── JSON 解包 ──
        actual_query = query
        if query.strip().startswith("{"):
            try:
                import json as _json
                data = _json.loads(query)
                if isinstance(data, dict) and "query" in data:
                    actual_query = data["query"]
            except Exception:
                pass  # 解析失败保持原样

        def _run():
            if tool_name == "search_laws":
                return self._search_laws(actual_query)
            elif tool_name == "search_cases":
                return self._search_cases(actual_query)
            elif tool_name == "search_web":
                return self._search_web(actual_query)
            elif tool_name == "calculate":
                return calculate_litigation_costs(actual_query)
            else:
                return f"错误：未知工具 '{tool_name}'，可用工具：search_laws, search_cases, search_web, calculate"

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

            # 2. 后过滤：doc_type + 相似度阈值
            DISTANCE_THRESHOLD = 1.2  # BGE L2 distance: <1.2 = relevant, >1.2 = noise
            filtered = []
            dropped = 0
            for doc, score in results:
                if score > DISTANCE_THRESHOLD:
                    dropped += 1
                    continue  # 低相似度噪音，丢弃
                doc_type = doc.metadata.get("doc_type", "")
                source = doc.metadata.get("source", "")
                if doc_type in ("law", "custom") or (
                    not doc_type and "cases" not in source.lower()
                ):
                    filtered.append((doc, score))
                    if len(filtered) >= self.top_k:
                        break

            print(f"[search_laws] 检索={len(results)}条, 阈值过滤={dropped}条, 有效={len(filtered)}条, {time.time()-t0:.1f}s")

            if not filtered:
                return (
                    "【本地库检索结果】未找到相关法律条文。\n"
                    "→ 建议调用 search_web 联网搜索补充。"
                )

            return self._format_results(filtered, "法律条文")

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

            # 相似度阈值过滤
            DISTANCE_THRESHOLD = 1.2
            filtered = []
            dropped = 0
            for doc, score in results:
                if score > DISTANCE_THRESHOLD:
                    dropped += 1
                    continue
                doc_type = doc.metadata.get("doc_type", "")
                source = doc.metadata.get("source", "")
                if doc_type == "case" or (
                    not doc_type and "cases" in source.lower()
                ):
                    filtered.append((doc, score))
                    if len(filtered) >= self.top_k:
                        break

            print(f"[search_cases] 检索={len(results)}条, 阈值过滤={dropped}条, 有效={len(filtered)}条, {time.time()-t0:.1f}s")

            if not filtered:
                return (
                    "【本地库检索结果】未找到相关案例。\n"
                    "→ 建议调用 search_web 联网搜索真实案例补充。"
                )

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


# ──────────────────────────────────────────────
#  确定性计算器：诉讼费用（独立函数，非类方法）
# ──────────────────────────────────────────────

# 《诉讼费用交纳办法》（国务院令第481号）— 财产案件阶梯费率表
_LITIGATION_TIERS = [
    # (区间上限, 费率, 速算扣除数)
    # 速算扣除数 = 上一区间满额费用 — 本区间下限 × 费率
    # 公式：受理费 = 标的额 × 费率 — 速算扣除数（用于简化计算）
    (0,          0,      0),       # 占位，不用
    (10_000,     0,      50),      # ≤1万：固定50元
    (100_000,    0.025,  200),     # 1-10万：2.5%，速算扣除=50 - 10000×0.025 = -200
    (200_000,    0.020,  700),     # 10-20万：2.0%，速算扣除=2450 - 100000×0.020 = 450... 
    (500_000,    0.015,  1700),    # 20-50万：1.5%
    (1_000_000,  0.010,  4200),    # 50-100万：1.0%
    (2_000_000,  0.009,  5200),    # 100-200万：0.9%
    (5_000_000,  0.008,  7200),    # 200-500万：0.8%
    (10_000_000, 0.007,  12200),   # 500-1000万：0.7%
    (20_000_000, 0.006,  22200),   # 1000-2000万：0.6%
    (float("inf"), 0.005, 42200),  # >2000万：0.5%
]

# 精确速算扣除数（逐区间验证）
_LITIGATION_QUICK = [
    (0,           0,      50),     # ≤1万
    (10_000,      0.025,  -200),   # 实际公式：50固定 + 超出部分×2.5%
    (100_000,     0.020,  300),    # 50+90000×2.5%=2300，2300-100000×2.0%=300
    (200_000,     0.015,  1300),   # 2300+100000×2.0%=4300，4300-200000×1.5%=1300
    (500_000,     0.010,  3800),   # 4300+300000×1.5%=8800，8800-500000×1.0%=3800
    (1_000_000,   0.009,  4800),   # 8800+500000×1.0%=13800，13800-1000000×0.9%=4800
    (2_000_000,   0.008,  6800),   # 13800+1000000×0.9%=22800，22800-2000000×0.8%=6800
    (5_000_000,   0.007,  11800),  # 22800+3000000×0.8%=46800，46800-5000000×0.7%=11800
    (10_000_000,  0.006,  21800),  # 46800+5000000×0.7%=81800，81800-10000000×0.6%=21800
    (20_000_000,  0.005,  41800),  # 81800+10000000×0.6%=141800，141800-20000000×0.5%=41800
    (float("inf"), 0.005,  41800), # >2000万：141800 + 超出部分的0.5%（或用速算公式）
]


def calculate_litigation_costs(query: str) -> str:
    """
    严格按《诉讼费用交纳办法》（国务院令第481号）计算财产案件受理费。

    支持输入格式：
    - 纯数字："300000"
    - 带"万"："30万"、"30万元"
    - 描述性："标的额30万"、"300000元"

    返回：分阶梯明细 + 最终总额
    """
    import re

    # ── 1. 解析金额 ──
    amount_raw = query.strip()

    # 匹配 "XX万" 模式
    wan_match = re.search(r'(\d+\.?\d*)\s*万', amount_raw)
    if wan_match:
        amount = float(wan_match.group(1)) * 10000
    else:
        # 提取所有数字，取最大值（有些输入可能包含多个数字，取最可能的标的额）
        nums = re.findall(r'\d+\.?\d*', amount_raw)
        if not nums:
            return (
                f"【计算错误】无法从输入中解析出标的金额。\n"
                f"原始输入：{query}\n"
                f"请提供明确的数字，如 '300000' 或 '30万'。"
            )
        amount = float(nums[0])

    if amount <= 0:
        return f"【计算错误】标的金额必须大于 0，收到：{amount}"

    amount = int(amount)

    # ── 2. 阶梯计算 ──
    if amount <= 10_000:
        cost = 50
        detail_lines = [
            f"标的额 {amount:,} 元 ≤ 1 万元",
            f"案件受理费：**50 元**（固定）",
        ]
    else:
        # 使用阶梯累加计算
        cost = 50  # 基础
        remaining = amount - 10_000

        tiers_detail = [
            (100_000, 0.025, "1万 — 10万部分"),
            (200_000, 0.020, "10万 — 20万部分"),
            (500_000, 0.015, "20万 — 50万部分"),
            (1_000_000, 0.010, "50万 — 100万部分"),
            (2_000_000, 0.009, "100万 — 200万部分"),
            (5_000_000, 0.008, "200万 — 500万部分"),
            (10_000_000, 0.007, "500万 — 1000万部分"),
            (20_000_000, 0.006, "1000万 — 2000万部分"),
            (float("inf"), 0.005, "超过 2000万部分"),
        ]

        prev_limit = 10_000
        detail_lines = [
            f"标的额 {amount:,} 元，按《诉讼费用交纳办法》阶梯计算：\n",
            f"  • 不超过 1 万元：**50 元**",
        ]

        for limit, rate, label in tiers_detail:
            if amount > prev_limit:
                segment = min(amount, limit) - prev_limit
                if segment > 0:
                    segment_cost = segment * rate
                    cost += segment_cost
                    detail_lines.append(
                        f"  • {label}（{segment:,} 元 × {rate*100:.1f}%）：**{segment_cost:,.0f} 元**"
                    )
                prev_limit = limit
            else:
                break

    # ── 3. 格式化输出 ──
    detail_lines.append(f"\n💰 **案件受理费合计：{cost:,.0f} 元**")
    detail_lines.append(f"\n> 依据：《诉讼费用交纳办法》（国务院令第481号）第十三条")

    return "\n".join(detail_lines)

