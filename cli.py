"""
命令行交互界面 - 法律RAG知识库
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.vector_store.chroma_store import ChromaVectorStore
from src.llm.legal_llm import LegalLLM
from src.chain.legal_rag_chain import LegalRAGChain


def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("[LegalRAG] 法律RAG知识库 - 命令行交互")
    print("=" * 60)
    print("输入您的法律问题，我将为您检索相关法律法规并提供解答。")
    print("输入 'quit' 或 'exit' 退出程序。")
    print("输入 'stats' 查看知识库统计信息。")
    print("=" * 60 + "\n")


def print_response(response):
    """打印响应"""
    print("\n" + "-" * 60)
    print("--- 回答：")
    print("-" * 60)
    print(response.answer)
    print("\n" + "-" * 60)
    print("--- 参考来源：")
    print("-" * 60)
    for i, source in enumerate(response.sources, 1):
        method = "[vector]" if source.get("retrieval_method") == "vector" else "[web]"
        score = f"(相似度: {source.get('score', 'N/A'):.2f})" if source.get("retrieval_method") == "vector" else ""
        print(f"\n{i}. {method} {score}")
        print(f"   来源: {source.get('source', '未知')}")
        if source.get("title"):
            print(f"   标题: {source['title']}")
        print(f"   内容: {source.get('content', '')[:100]}...")
    print("-" * 60)


def main():
    """主函数 — RAG 模式"""
    print_welcome()

    try:
        print("[...] 正在初始化向量存储...")
        vector_store = ChromaVectorStore(
            persist_dir=config.vector_db_dir,
            collection_name=config.collection_name,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_api_key=config.openai_api_key,
            embedding_base_url=config.openai_base_url,
            embedding_device=config.embedding_device,
        )

        print("[...] 正在初始化LLM...")
        llm = LegalLLM(
            provider=config.llm_provider,
            model=config.openai_model if config.llm_provider == "openai" else config.ollama_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            ollama_base_url=config.ollama_base_url,
            request_timeout=60,
        )

        print("[...] 正在初始化RAG链...")
        rag_chain = LegalRAGChain(
            vector_store=vector_store,
            llm=llm,
            top_k=config.retrieval_top_k,
            use_web_search=config.web_search_enabled,
        )

        print("\n[OK] 初始化完成！\n")

        chat_history = []

        while True:
            try:
                user_input = input("\n>> 您的问题: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n再见！")
                    break

                if user_input.lower() == "stats":
                    stats = vector_store.get_collection_stats()
                    print(f"\n--- 知识库统计：")
                    print(f"   集合名称: {stats['name']}")
                    print(f"   文档数量: {stats['count']}")
                    print(f"   存储位置: {stats['persist_dir']}")
                    continue

                if not user_input:
                    print("[WARN] 请输入您的问题")
                    continue

                print("\n[...] 正在检索相关法律信息...")
                response = rag_chain.query(
                    question=user_input,
                    chat_history=chat_history,
                )

                print_response(response)

                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response.answer})

                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n[ERROR] 查询失败: {str(e)}")

    except Exception as e:
        print(f"\n[ERROR] 初始化失败: {str(e)}")
        sys.exit(1)


def agent_main():
    """Agent 模式入口 — 多步推理法律助手"""
    print_agent_welcome()

    try:
        print("[...] 正在初始化向量存储...")
        vector_store = ChromaVectorStore(
            persist_dir=config.vector_db_dir,
            collection_name=config.collection_name,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_api_key=config.openai_api_key,
            embedding_base_url=config.openai_base_url,
            embedding_device=config.embedding_device,
        )

        print("[...] 正在初始化LLM...")
        llm = LegalLLM(
            provider=config.llm_provider,
            model=config.openai_model if config.llm_provider == "openai" else config.ollama_model,
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
            ollama_base_url=config.ollama_base_url,
            request_timeout=60,
        )

        print("[...] 正在初始化法律 Agent...")
        from src.agent.legal_agent import LegalAgent
        agent = LegalAgent(
            vector_store=vector_store,
            llm=llm,
            top_k=config.retrieval_top_k,
            max_steps=config.agent_max_steps,
            verbose=config.agent_verbose,
        )

        print("\n[OK] Agent 初始化完成！\n")

        chat_history = []

        while True:
            try:
                user_input = input("\n>> 您的问题: ").strip()

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\n再见！")
                    break

                if user_input.lower() == "stats":
                    stats = vector_store.get_collection_stats()
                    print(f"\n--- 知识库统计：")
                    print(f"   集合名称: {stats['name']}")
                    print(f"   文档数量: {stats['count']}")
                    print(f"   存储位置: {stats['persist_dir']}")
                    continue

                if not user_input:
                    print("[WARN] 请输入您的问题")
                    continue

                response = agent.run(
                    query=user_input,
                    chat_history=chat_history,
                )

                print("\n" + "=" * 60)
                print("--- Agent 最终回答：")
                print("=" * 60)
                print(response.answer)
                print(f"\n[INFO] 本次推理调用了 {response.tool_calls_count} 次工具，"
                      f"共 {len(response.steps)} 步")

                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response.answer})
                if len(chat_history) > 10:
                    chat_history = chat_history[-10:]

            except KeyboardInterrupt:
                print("\n\n再见！")
                break
            except Exception as e:
                print(f"\n[ERROR] 查询失败: {str(e)}")

    except Exception as e:
        print(f"\n[ERROR] 初始化失败: {str(e)}")
        sys.exit(1)


def print_agent_welcome():
    """Agent 模式欢迎信息"""
    print("\n" + "=" * 60)
    print("[LegalAgent] 法律 Agent - 多步推理助手")
    print("=" * 60)
    print("我是一个具备多步推理能力的法律助手。")
    print("您只需描述事实，我会自动：")
    print("  1. 提取关键法律事实")
    print("  2. 检索相关法律法规")
    print("  3. 查找相似案例")
    print("  4. 综合分析给出法律意见")
    print("")
    print("输入 'quit' 或 'exit' 退出程序。")
    print("输入 'stats' 查看知识库统计信息。")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
