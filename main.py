"""
法律RAG知识库 - 主入口
"""
import sys
import argparse
from pathlib import Path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="法律RAG知识库")
    parser.add_argument(
        "command",
        choices=["cli", "agent", "web", "api", "ingest"],
        help="运行模式: cli(RAG命令行), agent(Agent多步推理), web(Web界面), api(API服务), ingest(导入文档)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API服务监听地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API服务监听端口"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="数据目录路径（仅ingest模式有效）"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="清空现有数据后重新导入（仅ingest模式有效）"
    )

    args = parser.parse_args()

    if args.command == "cli":
        # 命令行模式 (RAG)
        from cli import main as cli_main
        cli_main()

    elif args.command == "agent":
        # Agent 多步推理模式
        from cli import agent_main
        agent_main()

    elif args.command == "web":
        # Web界面模式
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
        ])

    elif args.command == "api":
        # API服务模式
        import uvicorn
        from src.api.server import create_app

        app = create_app()
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
        )

    elif args.command == "ingest":
        # 导入文档模式
        from scripts.ingest_document import ingest_documents

        data_dir = args.data_dir or str(Path(__file__).parent / "data")
        ingest_documents(data_dir, args.clear)


if __name__ == "__main__":
    main()
