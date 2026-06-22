"""
Streamlit Web UI — 法律 Agent 流式多步推理助手

核心特性：流式输出 — Agent 思考过程和最终回答实时逐字打印
"""
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.vector_store.chroma_store import ChromaVectorStore
from src.llm.legal_llm import LegalLLM
from src.agent.legal_agent import LegalAgent


# ── 页面配置 ──

st.set_page_config(
    page_title="法律 Agent — 流式推理",
    page_icon="⚖️",
    layout="wide",
)


# ── 会话状态 ──

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None


def init_agent():
    if st.session_state.agent is None:
        with st.spinner("正在初始化法律 Agent..."):
            vector_store = ChromaVectorStore(
                persist_dir=config.vector_db_dir,
                collection_name=config.collection_name,
                embedding_provider=config.embedding_provider,
                embedding_model=config.embedding_model,
                embedding_api_key=config.openai_api_key,
                embedding_base_url=config.openai_base_url,
                embedding_device=config.embedding_device,
            )
            llm = LegalLLM(
                provider=config.llm_provider,
                model=config.openai_model if config.llm_provider == "openai" else config.ollama_model,
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                ollama_base_url=config.ollama_base_url,
                request_timeout=60,
            )
            st.session_state.agent = LegalAgent(
                vector_store=vector_store,
                llm=llm,
                top_k=config.retrieval_top_k,
                max_steps=config.agent_max_steps,
                verbose=False,
            )
            st.session_state.vector_store = vector_store
    return st.session_state.agent


# ── 侧边栏 ──

def display_sidebar():
    with st.sidebar:
        st.title("⚖️ 法律 Agent")
        st.caption("多步推理 · 法条检索 · 案例比对 · 流式输出")
        st.markdown("---")

        if st.session_state.vector_store:
            stats = st.session_state.vector_store.get_collection_stats()
            st.metric("知识库文档块", stats["count"])
        st.markdown("---")

        st.subheader("工作流程")
        st.markdown("""
        1. 🧠 **提取事实**
        2. 📖 **检索法条**
        3. ⚖️ **查找案例**
        4. 📝 **综合分析**
        """)
        st.markdown("---")

        st.subheader("示例问题")
        examples = [
            "我昨晚喝了三瓶啤酒开车回家，被交警查了，吹气95，会怎么处理？",
            "公司没有提前通知就辞退了我，还说试用期不需要赔偿，这合法吗？",
            "房东在我租期没到的时候换了门锁，还说押金不退，我该怎么办？",
        ]
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.example_query = ex

        st.markdown("---")
        if st.button("🗑️ 清空对话历史", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── 流式 Agent 响应处理器 ──

def _render_thought_block(placeholder, text: str):
    """渲染思考块 — 半透明淡色等宽字体样式（占位符模式：替换不追加）"""
    # 截断：去掉 "Final Answer:" 及之后的内容，防止污染推理区
    import re
    clean = re.split(r'\n\s*(?:Final\s*)?Answer\s*[:：]', text, maxsplit=1)[0]
    clean = clean.rstrip()
    safe = clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    placeholder.markdown(
        f"""<div style="
            background: rgba(240, 244, 248, 0.06);
            color: #94a3b8;
            font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
            font-size: 13px;
            border-left: 3px solid rgba(148, 163, 184, 0.25);
            border-radius: 0 6px 6px 0;
            padding: 10px 14px;
            margin: 4px 0;
            white-space: pre-wrap;
            line-height: 1.65;
            max-height: 360px;
            overflow-y: auto;
        ">{safe}</div>""",
        unsafe_allow_html=True,
    )


def _render_observation_block(placeholder, text: str):
    """渲染工具返回结果 — 绿色调，区别于思考块"""
    # 截断过长的结果
    display = text[:500] + ("..." if len(text) > 500 else "")
    safe = display.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    placeholder.markdown(
        f"""<div style="
            background: rgba(16, 185, 129, 0.06);
            color: #6ee7b7;
            font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
            font-size: 12px;
            border-left: 3px solid rgba(16, 185, 129, 0.3);
            border-radius: 0 6px 6px 0;
            padding: 8px 14px;
            margin: 4px 0 4px 8px;
            white-space: pre-wrap;
            line-height: 1.55;
            max-height: 200px;
            overflow-y: auto;
        ">{safe}</div>""",
        unsafe_allow_html=True,
    )


def _render_step_summary(i: int, step: dict, labels: dict):
    """渲染步骤摘要 — 小巧的标签式展示"""
    action = step.get("action", "")
    label = labels.get(action, action)
    inp = (step.get("action_input", "") or "")[:80]
    st.markdown(
        f"""<div style="
            display: flex;
            align-items: baseline;
            gap: 8px;
            padding: 4px 0;
            font-size: 13px;
        ">
            <span style="
                color: #64748b;
                font-weight: 600;
                min-width: 44px;
            ">Step {i}</span>
            <span style="
                background: rgba(99, 102, 241, 0.12);
                color: #a5b4fc;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 12px;
                font-family: monospace;
            ">{label}</span>
            <span style="
                color: #94a3b8;
                font-family: monospace;
                font-size: 12px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            ">{inp}</span>
        </div>""",
        unsafe_allow_html=True,
    )


def render_streaming_response(agent: LegalAgent, query: str, chat_history: list):
    """
    消费 agent.run_stream() 流式事件，实时更新 UI。

    v4 — 体验优化：
    - 推理过程用 st.status 包裹，清晰标注"智能思考与检索过程"
    - 一旦开始输出 answer_token 或 done，自动折叠推理区，让用户聚焦结论
    - 最终状态栏简洁摘要
    """

    TOOL_LABELS = {
        "search_laws": "检索法条",
        "search_cases": "检索案例",
        "search_web": "联网搜索",
    }

    # ── UI 区域 ──
    status_area = st.empty()
    thinking_container = st.status(
        "🧠 智能思考与检索过程（点击展开）",
        expanded=True,
    )
    thinking_placeholder = thinking_container.empty()  # 占位符：替换而非追加
    thinking_placeholder.markdown(
        '<span style="color:#64748b;font-size:13px;">等待大模型开始推理...</span>',
        unsafe_allow_html=True,
    )
    answer_area = st.empty()

    # ── 状态 ──
    thinking_lines: list[str] = []
    final_answer = ""
    phase = "thinking"
    steps_data: list[dict] = []
    tool_calls_count = 0
    is_legal = True
    web_searched = False  # 本轮是否触发过联网搜索
    step_info = ""

    # ── 消费事件流 ──
    for event in agent.run_stream(query, chat_history):
        etype = event.get("type", "")

        # --- mode ---
        if etype == "mode":
            # 防御：main() 已判定为法律问题，强制 is_legal=True
            is_legal = True
            status_area.caption("⚖️ 法律分析模式 — 启动多步推理...")

        # --- step ---
        elif etype == "step":
            step_info = f"Step {event['step']}/{event['max']}"
            status_area.caption(f"🔍 {step_info} — 分析中...")
            thinking_lines = []

        # --- thought_token（实时更新推理区） ---
        elif etype == "thought_token":
            if phase == "thinking":
                thinking_lines.append(event["token"])
                _render_thought_block(thinking_placeholder, "".join(thinking_lines))

        # --- action ---
        elif etype == "action":
            name = event["action"]
            if name == "search_web":
                web_searched = True  # 标记本轮使用了联网搜索
            status_area.caption(
                f"🔧 {step_info} — {TOOL_LABELS.get(name, name)}：{event.get('input', '')[:60]}"
            )
            steps_data.append({
                "thought": "".join(thinking_lines),  # 完整保存思考文本
                "action": name,
                "action_input": event.get("input", ""),
            })

        # --- observation ---
        elif etype == "observation":
            tool_calls_count += 1
            status_area.caption(f"📋 {step_info} — 检索完成")
            # 保存工具返回结果到上一步
            if steps_data:
                steps_data[-1]["observation"] = event.get("content", "")
            thinking_lines = []

        # --- answer_token（切换阶段 + 自动折叠推理区） ---
        elif etype == "answer_token":
            if phase == "thinking":
                phase = "answering"
                # ⬇ 自动折叠推理区，让用户聚焦结论
                thinking_container.update(
                    label="🧠 推理过程（已自动收起）",
                    state="complete",
                    expanded=False,
                )
                status_area.caption(
                    f"📝 综合分析中...（已检索 {tool_calls_count} 次）"
                )
            final_answer += event["token"]
            answer_area.markdown(f"### 分析结论\n\n{final_answer}▌")

        # --- critic_start ---
        elif etype == "critic_start":
            status_area.caption(f"🛡️ Critic 审核中 — 核验法条引用准确性...")

        # --- critic_step ---
        elif etype == "critic_step":
            phase_label = {
                "local_check": "🔍 Critic 本地核验 — 比对知识库原文",
                "web_search": "🌐 Critic 联网升级 — 本地未命中，联网核对",
            }.get(event.get("phase", ""), event.get("detail", ""))
            status_area.caption(phase_label)

        # --- critic_correction ---
        elif etype == "critic_correction":
            status_area.caption("⚠️ Critic 发现法条引用异常")

        # --- critic_pass ---
        elif etype == "critic_pass":
            status_area.caption("✅ Critic 审核通过 — 法条引用准确")

        # --- critic_refs ---
        elif etype == "critic_refs":
            pass  # refs 在 done 中统一处理

        # --- done（最终渲染 + 推理摘要写入折叠区） ---
        elif etype == "done":
            done_answer = event.get("answer", "") or final_answer

            # 防御：answer_token 从未到达时，一次性渲染
            if not final_answer and done_answer:
                final_answer = done_answer
                import time as _time
                display = ""
                for ch in done_answer:
                    display += ch
                    answer_area.markdown(f"### 分析结论\n\n{display}▌")
                    _time.sleep(0.002)

            # 最终回答（无光标闪烁）
            refs = event.get("refs", [])
            display_answer = final_answer

            # 🌐 联网检索成果回填
            if web_searched or refs:
                badge = "\n\n---\n🌐 **本回答部分法条已通过互联网实时检索核验。**\n"
                if refs:
                    for i, r in enumerate(refs, 1):
                        badge += f"\n[{i}] [{r.get('title','链接')}]({r.get('url','#')})"
                display_answer = final_answer + badge

            # 捕获 Final Answer 中的参考链接 [1] title | url 格式
            import re as _re
            inline_refs = _re.findall(r'\[(\d+)\]\s*(.+?)\s*\|\s*(https?://\S+)', final_answer)
            if inline_refs:
                ref_block = "\n\n---\n🌐 **本回答参考来源：**\n"
                for num, title, url in inline_refs:
                    ref_block += f"\n[{num}] [{title.strip()}]({url.strip()})"
                display_answer += ref_block

            answer_area.markdown(f"### 分析结论\n\n{display_answer}")

            # 状态摘要 — 强制法律模式标签
            mode_label = "⚖️ 法律分析"
            if web_searched:
                mode_label += " + 🌐 联网"
            if tool_calls_count > 0:
                mode_label += f" | {tool_calls_count} 次检索"
            if steps_data:
                mode_label += f" | {len(steps_data)} 步推理"
            if refs:
                mode_label += f" | Critic 核验 {len(refs)} 条"
            status_area.caption(mode_label)

            # 推理摘要写入折叠区（保留完整 thought + action）
            if steps_data:
                thinking_container.update(
                    label="🧠 推理过程（已自动收起，点击展开）",
                    state="complete",
                    expanded=False,
                )
                with thinking_container:
                    for i, s in enumerate(steps_data, 1):
                        action_label = TOOL_LABELS.get(s.get('action',''), s.get('action',''))
                        st.markdown(f"**Step {i}**：{action_label}  →  `{s.get('action_input','')[:80]}`")
                        if s.get("thought", "").strip():
                            _render_thought_block(st, s["thought"])
                        if s.get("observation", "").strip():
                            _render_observation_block(st, s["observation"])
                        st.markdown("---")
            else:
                thinking_container.update(
                    label="🧠 推理过程", state="complete", expanded=False
                )

            return {
                "role": "assistant",
                "content": final_answer,
                "steps": steps_data,
                "steps_count": len(steps_data),
                "tool_calls_count": tool_calls_count,
                "is_legal_query": is_legal,
                "web_searched": web_searched,
            }

    # 极端兜底
    if final_answer:
        display = final_answer
        if web_searched:
            display += "\n\n---\n🌐 **本回答部分法条已通过互联网实时检索核验。**"
        answer_area.markdown(f"### 分析结论\n\n{display}")
    else:
        answer_area.markdown("### 分析结论\n\n分析未能完成，请重试。")

    return {
        "role": "assistant",
        "content": final_answer or "分析未能完成",
        "steps": steps_data,
        "steps_count": len(steps_data),
        "tool_calls_count": tool_calls_count,
        "is_legal_query": is_legal,
        "web_searched": web_searched,
    }


# ── 聊天历史显示 ──

def render_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("steps"):
                with st.expander("查看推理过程", expanded=False):
                    for i, s in enumerate(msg["steps"], 1):
                        st.markdown(f"**Step {i}**")
                        if s.get("thought"):
                            st.caption(f"思考：{s['thought'][:200]}")
                        if s.get("action"):
                            st.caption(f"动作：{s['action']}({s.get('action_input', '')[:100]})")
                        st.markdown("---")
            # 模式标签
            if msg.get("is_legal_query") is False:
                st.caption("💬 普通对话模式")
            else:
                # 法律模式 — 始终显示 ⚖️，不显示 💬
                tc = msg.get("tool_calls_count", 0)
                sc = msg.get("steps_count", 0)
                label = "⚖️ 法律分析模式"
                if tc: label += f" | {tc} 次检索"
                if sc: label += f" | {sc} 步推理"
                st.caption(label)


# ── 主函数 ──

def main():
    init_session_state()
    display_sidebar()

    st.title("⚖️ 法律 Agent — 流式多步推理")
    st.caption("描述你遇到的法律问题，Agent 会实时展示思考过程并流式输出分析结论")

    render_history()

    # 用户输入
    if "example_query" in st.session_state:
        user_input = st.session_state.pop("example_query")
    else:
        user_input = st.chat_input(
            "描述你遇到的法律问题，比如'我酒后开车被查了，吹气95，会怎么处理？'"
        )

    if user_input:
        # 显示用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── 路由守卫：非法律问题不初始化 Agent，直接轻量对话 ──
        LEGAL_KW = [
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
            "交警", "扣分", "触犯", "违法", "违规", "要紧",
            "查酒驾", "吹气", "酒精", "检测", "吊销", "驾照",
            "开除", "离职", "加班", "社保", "工伤",
            "打官司", "上诉", "强制", "执行", "冻结", "查封",
            "网贷", "高利贷", "套路贷", "担保", "抵押",
            "泄露", "隐私", "诽谤", "名誉", "侵权",
            "遗嘱", "抚养", "赡养", "监护",
        ]
        is_quick_chat = not any(kw in user_input for kw in LEGAL_KW)

        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        if is_quick_chat:
            # 普通对话：跳过 Agent 初始化，直接用 LLM 回复
            with st.chat_message("assistant"):
                from src.llm.legal_llm import LegalLLM
                llm = LegalLLM(
                    provider=config.llm_provider,
                    model=config.openai_model,
                    api_key=config.openai_api_key,
                    base_url=config.openai_base_url,
                    request_timeout=30,
                )
                messages = [{"role": "system", "content": "你好！我是智能法律Agent助手，由YS个人开发。\n\n我的主要功能包括：\n\n日常问答 —— 我可以回答各类日常问题，与你进行普通的交流对话。\n法律咨询 —— 如果你有法律相关的问题，我会引导你详细描述具体情况，然后为你提供专业的法律分析和建议。\n请问有什么我可以帮助你的吗？(*^▽^*)"}]
                messages.extend(chat_history[-4:])
                messages.append({"role": "user", "content": user_input})
                full = ""
                placeholder = st.empty()
                for token in llm.chat_with_history_stream(messages):
                    full += token
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
                st.caption("💬 普通对话模式")
                st.session_state.messages.append({
                    "role": "assistant", "content": full,
                    "is_legal_query": False,
                })
        else:
            # 法律问题：初始化 Agent 并流式推理
            agent = init_agent()
            with st.chat_message("assistant"):
                try:
                    result = render_streaming_response(agent, user_input, chat_history)
                    st.session_state.messages.append(result)
                except Exception as e:
                    st.error(f"查询失败: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"查询失败: {str(e)}",
                    })


if __name__ == "__main__":
    main()
