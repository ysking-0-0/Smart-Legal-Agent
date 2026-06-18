"""
法律LLM封装 - 支持 MiniMax / DeepSeek / OpenAI / Ollama
"""
from typing import Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class LegalLLM:
    """法律LLM封装 — 支持 MiniMax / DeepSeek / OpenAI 等 OpenAI 兼容 API"""

    # 法律问题专用提示词
    LEGAL_PROMPT = """你是一个专业的法律助手，专门回答中国法律相关问题。

请遵守以下规则：
1. 基于提供的法律条文和案例进行回答
2. 引用具体的法律条款作为依据
3. 如果知识库中没有相关内容，请说明并建议咨询专业律师
4. 保持客观中立，不提供法律建议
5. 回答要准确、简洁、易懂

【法律引用规范】
引用法律文件时，必须标注以下要素（按顺序）：
1. 法律全称（使用书名号《》）
2. 施行/修订年份版本
3. 具体章节、条款项
4. 法律分类（部门法）

标准句式：
《中华人民共和国XXX法》（XXXX年施行/修订），第X编/章 第X条/款，属于XX法类别。

举例示范：
- 《中华人民共和国民法典》（2021年施行），第二编物权 第一章一般规定，属于民商法类别。
- 《中华人民共和国道路交通安全法》（2021年修订），第四章道路通行规定 第二十六条，属于行政法类别。
- 《中华人民共和国劳动合同法》（2012年修订），第二章劳动合同的订立 第十条，属于社会法类别。
"""

    # 普通问题提示词
    GENERAL_PROMPT = """你是一个友好的AI助手，可以回答各种问题。

请遵守以下规则：
1. 回答要准确、简洁、易懂
2. 如果不确定答案，请诚实说明
3. 保持友好和专业的态度
4. 使用中文回答
"""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        request_timeout: int = 60,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.base_url = base_url
        self.ollama_base_url = ollama_base_url
        self.request_timeout = request_timeout
        self._llm: Optional[BaseChatModel] = None

    def get_llm(self) -> BaseChatModel:
        if self._llm is None:
            if self.provider == "openai":
                self._llm = self._create_openai_llm()
            elif self.provider == "ollama":
                self._llm = self._create_ollama_llm()
            else:
                raise ValueError(f"不支持的LLM提供商: {self.provider}")
        return self._llm

    def _create_openai_llm(self) -> BaseChatModel:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.model or "gpt-3.5-turbo",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
            base_url=self.base_url,
            request_timeout=self.request_timeout,
            max_retries=3,
        )

    def _create_ollama_llm(self) -> BaseChatModel:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=self.model or "qwen2:7b",
            temperature=self.temperature,
            num_predict=self.max_tokens,
            base_url=self.ollama_base_url or "http://localhost:11434",
        )

    def generate_answer(self, query, context, chat_history=None, is_legal_query=True) -> str:
        llm = self.get_llm()
        system_prompt = self.LEGAL_PROMPT if is_legal_query else self.GENERAL_PROMPT
        messages = [SystemMessage(content=system_prompt)]
        if chat_history:
            for item in chat_history[-5:]:
                if item.get("role") == "user":
                    messages.append(HumanMessage(content=item["content"]))
                elif item.get("role") == "assistant":
                    messages.append(AIMessage(content=item["content"]))
        if is_legal_query and context:
            prompt = f"基于以下法律信息回答用户问题。\n\n【法律信息】\n{context}\n\n【用户问题】\n{query}\n\n请提供准确、专业的回答，并引用相关法律条款。"
        else:
            prompt = query
        messages.append(HumanMessage(content=prompt))
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "ssl" in error_msg.lower():
                return f"抱歉，网络连接超时，请稍后重试。错误详情：{error_msg}"
            else:
                return f"抱歉，生成回答时出现错误：{error_msg}"

    def chat_with_history(self, messages: list) -> str:
        """通用对话接口"""
        llm = self.get_llm()
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        try:
            response = llm.invoke(lc_messages)
            return response.content
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "ssl" in error_msg.lower():
                return f"抱歉，网络连接超时，请稍后重试。错误详情：{error_msg}"
            else:
                return f"抱歉，生成回答时出现错误：{error_msg}"

    def chat_with_history_stream(self, messages: list):
        """
        流式对话接口 — 逐 Token yield。
        内置 MiniMax-M1 think 标签剥离：跨 chunk 缓冲，自动过滤。
        """
        import re as _re
        llm = self.get_llm()
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        buf = ""
        in_think = False

        try:
            for chunk in llm.stream(lc_messages):
                if not chunk.content:
                    continue
                buf += chunk.content

                while True:
                    if not in_think:
                        m = _re.search(r'<\s*think\s*>', buf, _re.IGNORECASE)
                        if not m:
                            yield buf; buf = ""; break
                        if m.start() > 0:
                            yield buf[:m.start()]
                        buf = buf[m.end():]
                        in_think = True
                    else:
                        m = _re.search(r'<\s*/\s*think\s*>', buf, _re.IGNORECASE)
                        if not m:
                            break
                        buf = buf[m.end():]
                        in_think = False

            if not in_think and buf.strip():
                yield buf

        except Exception as e:
            yield f"\n[stream error: {str(e)}]"

    @property
    def llm(self) -> BaseChatModel:
        return self.get_llm()
