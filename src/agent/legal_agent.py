"""
法律 Agent — ReAct 多步推理循环

从单管线 RAG 升级为多步推理 Agent：
- 自主决定调用哪个工具、何时调用
- 先查法条定法定性 → 再查案例参考量刑 → 最后综合回答
- 信息不足时可反问用户
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .tools import AgentTools
from .critic import LegalCritic, CriticReview


@dataclass
class AgentStep:
    """Agent 单步记录"""
    step_num: int
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


@dataclass
class AgentResponse:
    """Agent 最终响应"""
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    tool_calls_count: int = 0
    query: str = ""
    is_legal_query: bool = True


class LegalAgent:
    """法律 Agent — ReAct 循环 + 智能路由"""

    # ── 法律关键词（用于判断是否需要启动 Agent 推理） ──

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
        "闯红灯", "违章", "扣分", "酒驾", "醉驾", "驾驶",
        "被捕", "被抓", "被查", "警察", "法院", "法官", "律师",
        "诉讼费", "起诉", "标的额", "标的", "受理费", "计算",
        "赔", "欠款", "借条", "欠条", "押金", "退租",
        "公司", "裁员", "试用期", "劳动合同", "五险一金",
        "伤人", "打架", "盗窃", "诈骗", "抢劫",
        # v2 扩展：覆盖更多口语化法律咨询
        "交警", "扣分", "触犯", "违法", "违规", "要紧",
        "查酒驾", "吹气", "酒精", "检测", "吊销", "驾照",
        "开除", "离职", "加班", "社保", "工伤",
        "打官司", "上诉", "强制", "执行", "冻结", "查封",
        "网贷", "高利贷", "套路贷", "担保", "抵押",
        "泄露", "隐私", "诽谤", "名誉", "侵权",
        "遗嘱", "抚养", "赡养", "监护",
    ]

    # ── 通用对话提示词（非法律问题） ──

    GENERAL_PROMPT = """你好！我是智能法律Agent助手，由YS个人开发。

我的主要功能包括：

日常问答 —— 我可以回答各类日常问题，与你进行普通的交流对话。
法律咨询 —— 如果你有法律相关的问题，我会引导你详细描述具体情况，然后为你提供专业的法律分析和建议。
请问有什么我可以帮助你的吗？(*^▽^*)"""

    # ── 法律 Agent System Prompt ──

    SYSTEM_PROMPT = """你是智能法律Agent助手（Legal Agent），由YS个人开发，具备多步推理能力。

## CRITICAL: FORMAT ENFORCEMENT (最高优先级)

YOU MUST ALWAYS REPLY IN ONE OF THE FOLLOWING FORMATS. NO EXCEPTIONS.
Direct answers, explanations, or any text outside these formats are STRICTLY FORBIDDEN.

### CORRECT FORMAT — To call a tool:
```
Thought: [your reasoning in Chinese]
Action: [exact tool name: search_laws search_cases search_web calculate]
Action Input: [query string for the tool]
```

### CORRECT FORMAT — To give final answer:
```
Thought: [summary of all gathered information]
Final Answer: [complete legal analysis for the user]
```

### WRONG — These are FORBIDDEN and will BREAK the system:
- Answering directly without calling tools first
- Writing "Answer:" instead of "Final Answer:"
- Using any other format or free-form text
- Skipping the Thought line

### FEW-SHOT EXAMPLES:

Example 1 — Legal query that requires tool:
User: "酒后驾驶怎么处罚？"
WRONG: "酒后驾驶会被罚款和扣分..." <-- FORBIDDEN
CORRECT:
Thought: 用户询问酒后驾驶的处罚规定，需要检索《道路交通安全法》相关条款。
Action: search_laws
Action Input: 酒后驾驶 处罚 道路交通安全法 第九十一条

Example 2 — Calculation query that requires tool:
User: "标的额30万，诉讼费多少？"
CORRECT:
Thought: 用户需要计算诉讼费，这是确定性数学计算，必须调用 calculate 工具。
Action: calculate
Action Input: 300000

Example 3 — Local search returned EMPTY, MUST try web search:
User: "劳动法裁员赔偿标准是什么？"
WRONG: 直接给 Final Answer 或放弃搜索 <-- FORBIDDEN
CORRECT:
Thought: 本地库未找到劳动法相关内容，必须联网搜索。
Action: search_web
Action Input: 劳动法 裁员 经济补偿 赔偿标准

### CRITICAL RULE: 如果 search_laws 或 search_cases 返回"未找到"或"建议联网"，你必须立即调用 search_web，禁止跳过联网直接给 Final Answer。

Example 3 — After gathering enough information, give final answer:
CORRECT:
Thought: 已检索到《道路交通安全法》第九十一条，明确了饮酒驾驶和醉酒驾驶的处罚。可以给出完整分析。
Final Answer: 根据《中华人民共和国道路交通安全法》...

## 你的能力
你可以使用以下工具来获取信息：
{tool_descriptions}

## 工作流程
1. **事实提取**：从用户描述中提炼关键法律事实（主体、行为、后果、时间、地点等）
2. **法条检索**：调用 search_laws 查找相关法律规定，明确法律定性
3. **案例检索**：调用 search_cases 查找类似判例，了解司法实践
4. **费用/赔偿计算**：涉及诉讼费、赔偿金等数学计算时，必须调用 calculate 工具。禁止自行估算或口算数字。
5. **综合分析**：结合法条、案例和计算结果，给出全面的法律分析

## 最终回答要求
- 先定性：说明该行为在法律上属于什么性质（行政违法/刑事犯罪/民事纠纷）
- 引法条：引用具体法律条款，格式如《中华人民共和国道路交通安全法》第九十一条
- 列处罚：说明对应的法律后果
- 举案例：如有相关案例，简要说明
- 给建议：基于法律规定给出下一步行动建议

## 最终回答要求
- 先定性：说明该行为在法律上属于什么性质（行政违法/刑事犯罪/民事纠纷）
- 引法条：引用具体法律条款（书名号+条款号），格式如《中华人民共和国道路交通安全法》（2021年修订）第九十一条
- 列处罚：说明对应的法律后果（罚款金额、拘留时长、刑事责任等）
- 举案例：如有相关案例，简要说明类似案件的判决结果以供参考
- 给建议：基于法律规定给出下一步行动建议
- 如果信息不足以做出判断，应列明需要补充哪些信息

## 重要规则
- 每次只能调用一个工具
- 必须先思考（Thought）再行动（Action）
- 搜索法条和案例时，使用精准的关键词
- 不要编造法律条款 — 所有引用必须来自检索结果
- 如果某个工具没有返回有用信息，尝试换个关键词重新搜索"""

    def __init__(
        self,
        vector_store,
        llm,  # LegalLLM 实例
        top_k: int = 5,
        max_steps: int = 6,
        verbose: bool = True,
    ):
        """
        Args:
            vector_store: ChromaVectorStore 实例
            llm: LegalLLM 实例
            top_k: 每次检索返回数量
            max_steps: 最大推理步数（防止无限循环）
            verbose: 是否打印中间步骤
        """
        self.vector_store = vector_store
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose

        # 初始化混合检索器（Dense + BM25 + RRF）
        from ..retrieval.hybrid_retriever import HybridRetriever
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            top_k=top_k,
            dense_weight=0.5,
            sparse_weight=0.5,
            use_web_search=False,  # web search 由 Agent 的 search_web 工具单独处理
        )

        # 初始化工具（传入混合检索器）
        self.tools = AgentTools(vector_store=vector_store, top_k=top_k, retriever=self.hybrid_retriever)

        # 初始化 Critic（法条校验 Agent）
        self.critic = LegalCritic(vector_store=vector_store, llm=llm)

        # 预热向量存储 — 首次调用触发 BGE 模型加载，避免后续卡顿
        self._warmup_vector_store()

    def _warmup_vector_store(self):
        """预热：触发 ChromaDB + BGE 的首次加载，防止第一次工具调用时冷启动超时"""
        try:
            print("[Agent] 预热向量存储...")
            _ = self.vector_store.similarity_search_with_score("预热", k=1)
            print("[Agent] 预热完成")
        except Exception as e:
            print(f"[Agent] 预热跳过 ({e})")

    def run(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResponse:
        """
        运行 Agent 处理用户问题。
        自动路由：法律问题 → ReAct 多步推理；普通问题 → 直接对话

        Args:
            query: 用户问题
            chat_history: 聊天历史

        Returns:
            AgentResponse — 包含最终回答和推理步骤
        """
        # ── 智能路由：判断是否是法律问题 ──
        is_legal = self._is_legal_query(query, chat_history)

        if not is_legal:
            # 普通对话模式：不加法律 System Prompt，不搜索知识库
            return self._general_chat(query, chat_history)

        # ── 下面是法律 Agent ReAct 模式 ──
        steps: List[AgentStep] = []
        tool_calls_count = 0

        # 构建初始消息
        system_msg = self._build_system_prompt()
        conversation = [{"role": "system", "content": system_msg}]

        # 添加聊天历史
        if chat_history:
            for item in chat_history[-6:]:
                conversation.append(item)

        # 用户问题
        conversation.append({"role": "user", "content": query})

        # ── ReAct 循环 ──
        for step_num in range(1, self.max_steps + 1):
            if self.verbose:
                print(f"\n{'─' * 40}")
                print(f"[Step {step_num}/{self.max_steps}]")

            # 调用 LLM
            response_text = self.llm.chat_with_history(conversation)

            # 解析响应
            thought, action, action_input, final_answer = self._parse_response(response_text)

            if self.verbose:
                print(f"[Thought] {thought[:100]}..." if len(thought) > 100 else f"[Thought] {thought}")

            # ── 检查是否是最终回答 ──
            if final_answer:
                step = AgentStep(
                    step_num=step_num,
                    thought=thought,
                    action="final_answer",
                    action_input="",
                    observation="",
                )
                steps.append(step)

                if self.verbose:
                    print("[Done] 给出最终回答")

                return AgentResponse(
                    answer=final_answer,
                    steps=steps,
                    tool_calls_count=tool_calls_count,
                    query=query,
                )

            # ── 检查是否是工具调用 ──
            if action and action_input:
                if self.verbose:
                    print(f"[Action] {action}({action_input[:80]}...)")

                # 执行工具
                observation = self.tools.execute(action, action_input)
                tool_calls_count += 1

                if self.verbose:
                    obs_preview = observation[:150].replace("\n", " ")
                    print(f"[Result] {obs_preview}...")

                step = AgentStep(
                    step_num=step_num,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    observation=observation[:800],  # 截断过长结果
                )
                steps.append(step)

                # 将本轮交互追加到对话
                conversation.append({"role": "assistant", "content": response_text})
                conversation.append({
                    "role": "user",
                    "content": f"Observation: {observation[:800]}"
                })
                continue

            # ── 无法解析 ──
            # 把 LLM 输出当作最终回答
            if self.verbose:
                print("[WARN] 未检测到工具调用，视为最终回答")

            return AgentResponse(
                answer=response_text,
                steps=steps,
                tool_calls_count=tool_calls_count,
                query=query,
            )

        # ── 达到最大步数 ──
        # 让 LLM 基于已有信息给出最终回答
        conversation.append({
            "role": "user",
            "content": "已达到最大搜索步数。请基于目前已获取的所有信息，给出最终回答。"
        })
        final_response = self.llm.chat_with_history(conversation)

        return AgentResponse(
            answer=final_response,
            steps=steps,
            tool_calls_count=tool_calls_count,
            query=query,
        )

    def _is_legal_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> bool:
        """
        判断是否是法律相关查询。
        检查当前问题 + 最近一轮对话历史（上下文连续性）
        """
        query_lower = query.lower()
        for keyword in self.LEGAL_KEYWORDS:
            if keyword in query_lower:
                return True

        # 如果当前问题很短（如"然后呢"、"会怎样"），检查上一轮是否在聊法律
        if chat_history and len(query) < 10:
            last_user = ""
            for item in reversed(chat_history):
                if item.get("role") == "user":
                    last_user = item.get("content", "")
                    break
            if last_user:
                for keyword in self.LEGAL_KEYWORDS:
                    if keyword in last_user.lower():
                        return True

        return False

    def _general_chat(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> AgentResponse:
        """
        普通对话模式 — 不加法律系统提示词，不调用工具，不做知识库检索
        """
        messages = [{"role": "system", "content": self.GENERAL_PROMPT}]
        if chat_history:
            messages.extend(chat_history[-6:])
        messages.append({"role": "user", "content": query})

        answer = self.llm.chat_with_history(messages)

        return AgentResponse(
            answer=answer,
            steps=[],
            tool_calls_count=0,
            query=query,
            is_legal_query=False,
        )

    # ── Critic 审核循环（Generator-Critic 双 Agent） ──

    MAX_CRITIC_LOOPS = 2

    def _do_critic_review(self, draft_answer: str, query: str):
        """
        Critic 审核 → 逐事件 yield（供 run_stream 调用）

        Yields: critic_start, critic_step, critic_correction, critic_pass, critic_refs
        """
        yield {"type": "critic_start"}

        # 本地库核验
        yield {
            "type": "critic_step",
            "phase": "local_check",
            "detail": "Critic 正在比对本地知识库中的法条原文...",
        }

        review = self.critic.review(draft_answer, query)

        if review.refs:
            yield {
                "type": "critic_step",
                "phase": "web_search",
                "detail": f"本地库未完全覆盖，已联网核验 {len(review.refs)} 条法条",
            }

        if not review.passed:
            yield {
                "type": "critic_correction",
                "message": review.corrections,
            }
        else:
            yield {"type": "critic_pass"}

        if review.refs:
            yield {"type": "critic_refs", "refs": review.refs}

    # ── 流式 Agent 循环 ──

    def run_stream(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        流式 Agent — 逐事件 yield，供 Streamlit 实时渲染

        Yields:
            dict — 事件：
                {"type": "mode", "mode": "legal"|"general"}
                {"type": "step", "step": N, "max": M}
                {"type": "thought_token", "token": "..."}
                {"type": "action", "action": "...", "input": "..."}
                {"type": "observation", "content": "..."}
                {"type": "answer_token", "token": "..."}
                {"type": "done", "answer": "...", "steps": [...], ...}
        """
        is_legal = self._is_legal_query(query, chat_history)

        # ── 普通对话模式 ──
        if not is_legal:
            yield {"type": "mode", "mode": "general"}
            messages = [{"role": "system", "content": self.GENERAL_PROMPT}]
            if chat_history:
                messages.extend(chat_history[-6:])
            messages.append({"role": "user", "content": query})

            full = ""
            for token in self.llm.chat_with_history_stream(messages):
                full += token
                yield {"type": "answer_token", "token": token}
            yield {
                "type": "done",
                "answer": full,
                "steps": [],
                "tool_calls_count": 0,
                "is_legal_query": False,
            }
            return

        # ── 法律 Agent ReAct 流式模式 ──
        yield {"type": "mode", "mode": "legal"}

        steps: List[AgentStep] = []
        tool_calls_count = 0

        system_msg = self._build_system_prompt()
        conversation = [{"role": "system", "content": system_msg}]
        if chat_history:
            for item in chat_history[-6:]:
                conversation.append(item)
        conversation.append({"role": "user", "content": query})

        for step_num in range(1, self.max_steps + 1):
            yield {"type": "step", "step": step_num, "max": self.max_steps}

            # 流式调用 LLM
            response_text = ""
            for token in self.llm.chat_with_history_stream(conversation):
                response_text += token
                yield {"type": "thought_token", "token": token}

            # 解析完整响应
            thought, action, action_input, final_answer = self._parse_response(response_text)

            # ── 最终回答（经过 Critic 审核）──
            if final_answer and final_answer.strip():
                step = AgentStep(
                    step_num=step_num, thought=thought,
                    action="final_answer", action_input="", observation="",
                )
                steps.append(step)

                # 流式输出最终回答
                for char in final_answer:
                    yield {"type": "answer_token", "token": char}

                # ── Critic 审核 ──
                refs = []
                for critic_ev in self._do_critic_review(final_answer, query):
                    yield critic_ev
                    if critic_ev.get("type") == "critic_refs":
                        refs = critic_ev.get("refs", [])

                yield {
                    "type": "done",
                    "answer": final_answer,
                    "steps": [{
                        "step_num": s.step_num, "thought": s.thought,
                        "action": s.action, "action_input": s.action_input,
                        "observation": s.observation,
                    } for s in steps],
                    "tool_calls_count": tool_calls_count,
                    "is_legal_query": True,
                    "refs": refs,
                }
                return

            # ── 工具调用（严格校验：action 必须是已知工具名） ──
            VALID_ACTIONS = {"search_laws", "search_cases", "search_web", "calculate"}
            if action and action_input and action.strip() in VALID_ACTIONS:
                yield {
                    "type": "action",
                    "action": action,
                    "input": action_input[:200],
                }

                observation = self.tools.execute(action, action_input)
                tool_calls_count += 1

                yield {
                    "type": "observation",
                    "content": observation[:300],
                }

                step = AgentStep(
                    step_num=step_num, thought=thought,
                    action=action, action_input=action_input,
                    observation=observation[:800],
                )
                steps.append(step)

                conversation.append({"role": "assistant", "content": response_text})
                conversation.append({
                    "role": "user",
                    "content": f"Observation: {observation[:800]}"
                })
                continue

            # ── 无法解析 / LLM 未遵循 ReAct 格式 ──
            answer_text = response_text.strip() or "分析未能生成，请重试。"
            for char in answer_text:
                yield {"type": "answer_token", "token": char}
            refs = []
            for critic_ev in self._do_critic_review(answer_text, query):
                yield critic_ev
                if critic_ev.get("type") == "critic_refs":
                    refs = critic_ev.get("refs", [])
            yield {
                "type": "done",
                "answer": answer_text,
                "steps": [{
                    "step_num": s.step_num, "thought": s.thought,
                    "action": s.action, "action_input": s.action_input,
                    "observation": s.observation,
                } for s in steps],
                "tool_calls_count": tool_calls_count,
                "is_legal_query": True,
                "refs": refs,
            }
            return

        # ── 达到最大步数 ──
        conversation.append({
            "role": "user",
            "content": "已达到最大搜索步数。请基于目前已获取的所有信息，给出最终回答。"
        })
        final_response = self.llm.chat_with_history(conversation)
        _, _, _, parsed_answer = self._parse_response(final_response)
        answer_text = parsed_answer if parsed_answer else final_response
        for char in answer_text:
            yield {"type": "answer_token", "token": char}
        refs = []
        for critic_ev in self._do_critic_review(answer_text, query):
            yield critic_ev
            if critic_ev.get("type") == "critic_refs":
                refs = critic_ev.get("refs", [])
        yield {
            "type": "done",
            "answer": answer_text,
            "steps": [{
                "step_num": s.step_num, "thought": s.thought,
                "action": s.action, "action_input": s.action_input,
                "observation": s.observation,
            } for s in steps],
            "tool_calls_count": tool_calls_count,
            "is_legal_query": True,
            "refs": refs,
        }

    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tool_descs = []
        for td in self.tools.tool_definitions():
            tool_descs.append(
                f"- **{td['name']}**: {td['description']}\n"
                f"  参数：{td['parameters']}"
            )
        return self.SYSTEM_PROMPT.format(
            tool_descriptions="\n".join(tool_descs)
        )

    # 已知工具名（用于从粘连文本中回退匹配）
    _KNOWN_ACTIONS = {"calculate", "search_laws", "search_cases", "search_web"}

    def _parse_response(self, text: str) -> tuple:
        """
        解析 LLM 响应 — 防线 V4

        新增防御：
        - Action: XAction Input: Y → 自动插入换行符
        - Action 名只匹配已知工具名（杜绝 false positive）

        Returns:
            (thought, action, action_input, final_answer)
        """
        if not text or not isinstance(text, str):
            return ("", "", "", "")

        # ── 防御 0：修复粘连格式 Action: calculateAction Input: 1500000 ──
        text = re.sub(
            r'(Action\s*[:：]\s*(?:' + '|'.join(self._KNOWN_ACTIONS) + r'))'
            r'(Action\s*Input\s*[:：])',
            r'\1\n\2',
            text, flags=re.IGNORECASE
        )

        thought = ""
        action = ""
        action_input = ""
        final_answer = ""

        # ── Final Answer ──
        final_match = re.search(
            r'(?:Final\s*)?Answer\s*[:：]\s*(.+)', text, re.DOTALL | re.IGNORECASE
        )
        if not final_match:
            final_match = re.search(r'最终\s*(?:回答|答案)\s*[:：]\s*(.+)', text, re.DOTALL)

        if final_match:
            final_answer = final_match.group(1).strip()
            pre_final = text[:final_match.start()]
            thought_m = re.search(
                r'Thought\s*[:：]\s*(.+?)$', pre_final, re.DOTALL | re.IGNORECASE
            )
            if thought_m:
                thought = thought_m.group(1).strip()
            return (thought, action, action_input, final_answer)

        # ── Thought ──
        thought_match = re.search(
            r'Thought\s*[:：]\s*(.+?)(?=\n\s*(?:Action|Final)|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        # ── Action（只匹配已知工具名，杜绝 calculateAction 这种粘连匹配）──
        for known in self._KNOWN_ACTIONS:
            m = re.search(
                rf'Action\s*[:：]\s*{re.escape(known)}\b',
                text, re.IGNORECASE
            )
            if m:
                action = known
                break

        # ── Action Input（更宽容：不要求前有换行）──
        input_match = re.search(
            r'Action\s*Input\s*[:：]\s*(.+?)(?=\n\s*(?:Observation|Thought|Action|Final|$)|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if input_match:
            action_input = input_match.group(1).strip()

        return (thought, action, action_input, final_answer)
