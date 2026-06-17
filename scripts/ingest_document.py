"""
文档导入脚本 - 将法律文档导入知识库
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.document_processor.loader import DocumentLoader
from src.document_processor.splitter import LegalDocumentSplitter
from src.vector_store.chroma_store import ChromaVectorStore


def ingest_documents(data_dir: str, clear_existing: bool = False):
    """
    导入文档到知识库

    Args:
        data_dir: 数据目录
        clear_existing: 是否清空现有数据
    """
    print("=" * 50)
    print("[LegalRAG] 法律文档导入工具")
    print("=" * 50)

    # 1. 初始化文档加载器
    print("\n1. 初始化文档加载器...")
    loader = DocumentLoader(data_dir)

    # 2. 初始化文档分块器
    print("2. 初始化文档分块器...")
    splitter = LegalDocumentSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        legal_split=config.get("document.chunk.legal_split", True),
    )

    # 3. 初始化向量存储
    print("3. 初始化向量存储...")
    vector_store = ChromaVectorStore(
        persist_dir=config.vector_db_dir,
        collection_name=config.collection_name,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_api_key=config.openai_api_key,
        embedding_base_url=config.openai_base_url,
        embedding_device=config.embedding_device,
    )

    # 4. 清空现有数据（可选）
    if clear_existing:
        print("4. 清空现有数据...")
        vector_store.delete_collection()
        # 重新初始化
        vector_store = ChromaVectorStore(
            persist_dir=config.vector_db_dir,
            collection_name=config.collection_name,
            embedding_provider=config.embedding_provider,
            embedding_model=config.embedding_model,
            embedding_api_key=config.openai_api_key,
            embedding_base_url=config.openai_base_url,
            embedding_device=config.embedding_device,
        )

    # 5. 加载文档
    print(f"\n5. 从 {data_dir} 加载文档...")
    documents = loader.load_directory(data_dir)
    print(f"   [OK] 加载了 {len(documents)} 个文档")

    if not documents:
        print("\n[WARN] 未找到任何文档，请将文档放入 data 目录")
        return

    # 6. 文档分块
    print("\n6. 文档智能分块...")
    chunks = splitter.split_documents(documents)
    print(f"   [OK] 生成了 {len(chunks)} 个文档块")

    # 7. 导入向量存储
    print("\n7. 导入向量存储...")
    vector_store.add_documents(chunks)

    # 8. 显示统计信息
    stats = vector_store.get_collection_stats()
    print("\n" + "=" * 50)
    print("[DONE] 导入完成！")
    print(f"   集合名称: {stats['name']}")
    print(f"   文档数量: {stats['count']}")
    print(f"   存储位置: {stats['persist_dir']}")
    print("=" * 50)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="法律文档导入工具")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.data_dir),
        help="数据目录路径"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="清空现有数据后重新导入"
    )

    args = parser.parse_args()

    try:
        ingest_documents(args.data_dir, args.clear)
    except Exception as e:
        print(f"\n[ERROR] 导入失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
