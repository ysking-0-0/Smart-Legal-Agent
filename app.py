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
    answer_area = st.empty()

    # ── 状态 ──
    thinking_lines: list[str] = []
    final_answer = ""
    phase = "thinking"
    steps_data: list[dict] = []
    tool_calls_count = 0
    is_legal = True
    step_info = ""

    # ── 消费事件流 ──
    for event in agent.run_stream(query, chat_history):
        etype = event.get("type", "")

        # --- mode ---
        if etype == "mode":
            is_legal = (event["mode"] == "legal")
            if not is_legal:
                phase = "answering"
                thinking_container.update(
                    label="💬 普通对话", state="complete", expanded=False
                )
                status_area.caption("智能路由：普通对话模式")

        # --- step ---
        elif etype == "step":
            step_info = f"Step {event['step']}/{event['max']}"
            status_area.caption(f"🔍 {step_info} — 分析中...")
            thinking_lines = []

        # --- thought_token（实时更新推理区） ---
        elif etype == "thought_token":
            if phase == "thinking":
                thinking_lines.append(event["token"])
                thinking_container.write("".join(thinking_lines))

        # --- action ---
        elif etype == "action":
            name = event["action"]
            status_area.caption(
                f"🔧 {step_info} — {TOOL_LABELS.get(name, name)}：{event.get('input', '')[:60]}"
            )
            steps_data.append({
                "thought": "".join(thinking_lines)[:300],
                "action": name,
                "action_input": event.get("input", ""),
            })

        # --- observation ---
        elif etype == "observation":
            tool_calls_count += 1
            status_area.caption(f"📋 {step_info} — 检索完成")
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
            answer_area.markdown(f"### 分析结论\n\n{final_answer}")

            # 简洁状态摘要
            status_area.caption(
                f"{'⚖️ 法律分析' if is_legal else '💬 普通对话'}"
                f" | 检索 {tool_calls_count} 次"
                + (f" | 推理 {len(steps_data)} 步" if steps_data else "")
            )

            # 推理摘要写入折叠区
            if steps_data:
                thinking_container.update(
                    label="🧠 推理过程（已自动收起，点击展开）",
                    state="complete",
                    expanded=False,
                )
                with thinking_container:
                    for i, s in enumerate(steps_data, 1):
                        action_label = TOOL_LABELS.get(s.get("action", ""), s.get("action", ""))
                        st.caption(f"**Step {i}**：{action_label} → {s.get('action_input', '')[:80]}")
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
            }

    # 极端兜底
    if final_answer:
        answer_area.markdown(f"### 分析结论\n\n{final_answer}")
    else:
        answer_area.markdown("### 分析结论\n\n分析未能完成，请重试。")

    return {
        "role": "assistant",
        "content": final_answer or "分析未能完成",
        "steps": steps_data,
        "steps_count": len(steps_data),
        "tool_calls_count": tool_calls_count,
        "is_legal_query": is_legal,
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
            if msg.get("is_legal_query") is False:
                st.caption("💬 普通对话模式")
            elif msg.get("tool_calls_count"):
                st.caption(
                    f"⚖️ 法律分析模式 | "
                    f"调用了 {msg['tool_calls_count']} 次工具，"
                    f"共 {msg.get('steps_count', 0)} 步"
                )


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

        # Agent 响应
        agent = init_agent()
        with st.chat_message("assistant"):
            try:
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

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
