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
        "赔", "欠款", "借条", "欠条", "押金", "退租",
        "公司", "裁员", "试用期", "劳动合同", "五险一金",
        "伤人", "打架", "盗窃", "诈骗", "抢劫",
    ]

    # ── 通用对话提示词（非法律问题） ──

    GENERAL_PROMPT = """你是一个友好的AI助手。你可以回答各种日常问题、进行普通对话。
如果用户问的是法律相关问题，你会建议他详细描述情况以便进行法律分析。

请保持友好、自然的态度。"""

    # ── 法律 Agent System Prompt ──

    SYSTEM_PROMPT = """你是一个专业的中国法律助手（Legal Agent），具备多步推理能力。

## 你的能力
你可以使用以下工具来获取信息：
{tool_descriptions}

## 工作流程
面对用户的法律问题，你应该按以下步骤思考：

1. **事实提取**：从用户描述中提炼关键法律事实（主体、行为、后果、时间、地点等）
2. **法条检索**：调用 search_laws 查找相关法律规定，明确法律定性
3. **案例检索**：调用 search_cases 查找类似判例，了解司法实践
4. **综合分析**：结合法条和案例，给出全面的法律分析

如果知识库信息不足，可以调用 search_web 联网搜索补充。

## 回答格式（严格遵守）

每次响应必须遵循以下格式之一：

### 需要调用工具时：
```
Thought: [你的推理过程 — 当前已知什么、还需要知道什么、为什么选择这个工具]
Action: [工具名称 — search_laws, search_cases, search_web]
Action Input: [传给工具的查询字符串]
```

### 准备最终回答时：
```
Thought: [最终推理 — 总结所有获取的信息，得出结论]
Final Answer: [给用户的完整回答]
```

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

        # 初始化工具
        self.tools = AgentTools(vector_store=vector_store, top_k=top_k)

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

            # ── 最终回答 ──
            if final_answer and final_answer.strip():
                step = AgentStep(
                    step_num=step_num, thought=thought,
                    action="final_answer", action_input="", observation="",
                )
                steps.append(step)

                # 流式输出最终回答
                for char in final_answer:
                    yield {"type": "answer_token", "token": char}

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
                }
                return

            # ── 工具调用（严格校验：action 必须是已知工具名） ──
            VALID_ACTIONS = {"search_laws", "search_cases", "search_web"}
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
            # 防御：把整个 response_text 当作最终回答
            answer_text = response_text.strip() or "分析未能生成，请重试。"
            for char in answer_text:
                yield {"type": "answer_token", "token": char}
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
            }
            return

        # ── 达到最大步数 ──
        conversation.append({
            "role": "user",
            "content": "已达到最大搜索步数。请基于目前已获取的所有信息，给出最终回答。"
        })
        final_response = self.llm.chat_with_history(conversation)
        # 解析出 Final Answer 部分（去掉可能的 Thought 前缀）
        _, _, _, parsed_answer = self._parse_response(final_response)
        answer_text = parsed_answer if parsed_answer else final_response
        for char in answer_text:
            yield {"type": "answer_token", "token": char}
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

    def _parse_response(self, text: str) -> tuple:
        """
        解析 LLM 响应 — 防御性版本

        兼容 LLM 输出的微小格式偏差：
        - "Thought:" / "thought:" 大小写
        - 中英文混排
        - Final Answer 后有多余内容

        Returns:
            (thought, action, action_input, final_answer)
        """
        if not text or not isinstance(text, str):
            return ("", "", "", "")

        thought = ""
        action = ""
        action_input = ""
        final_answer = ""

        # ── Final Answer（先检查，因为可能有 Thought + Final Answer 同时出现）──
        final_match = re.search(
            r'Final\s*Answer\s*[:：]\s*(.+)', text, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            final_answer = final_match.group(1).strip()
            # 提取 Final Answer 之前的 Thought
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

        # ── Action ──
        action_match = re.search(
            r'Action\s*[:：]\s*(\S+)', text, re.IGNORECASE
        )
        if action_match:
            action = action_match.group(1).strip().rstrip(".,;")

        # ── Action Input ──
        input_match = re.search(
            r'Action\s*Input\s*[:：]\s*(.+?)(?=\n\s*(?:Observation|Thought|Action|Final)|\Z)',
            text, re.DOTALL | re.IGNORECASE
        )
        if input_match:
            action_input = input_match.group(1).strip()

        return (thought, action, action_input, final_answer)
