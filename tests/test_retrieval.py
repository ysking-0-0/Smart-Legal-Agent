"""
检索准确率测试脚本

测试流程：
  1. 准备 N 个标准法律问题 + 期望命中的法条/案例
  2. 对每个问题分别跑 Dense-only / Sparse-only / Hybrid
  3. 计算 Hit@K、MRR 指标
  4. 输出对比报告

用法：
  python tests/test_retrieval.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from src.vector_store.chroma_store import ChromaVectorStore
from src.retrieval.hybrid_retriever import HybridRetriever


# ── 测试问题集 ──
# 每个问题：query + 期望在 Top-5 中出现的文档关键词
TEST_QUERIES = [
    {
        "query": "酒后驾驶怎么处罚？",
        "expect_keywords": ["第九十一条", "暂扣", "驾驶证", "饮酒后驾驶"],
    },
    {
        "query": "醉酒驾驶构成什么罪？",
        "expect_keywords": ["危险驾驶罪", "第一百三十三条", "拘役"],
    },
    {
        "query": "闯红灯违反了什么法律？",
        "expect_keywords": ["交通信号灯", "红灯", "第二十六条", "禁止通行"],
    },
    {
        "query": "交通事故责任怎么划分？",
        "expect_keywords": ["交通事故", "过错", "责任", "第七十条"],
    },
    {
        "query": "劳动合同解除的条件",
        "expect_keywords": ["劳动合同", "解除", "第三十六条", "协商"],
    },
    {
        "query": "经济补偿金怎么计算？",
        "expect_keywords": ["经济补偿", "第四十七条", "工作年限", "一个月工资"],
    },
    {
        "query": "个人信息被泄露怎么办？",
        "expect_keywords": ["个人信息", "保护", "网络安全"],
    },
    {
        "query": "租房押金不退合法吗？",
        "expect_keywords": ["押金", "租房", "返还"],
    },
]


def compute_metrics(retriever, queries, retriever_name: str):
    """
    计算检索指标

    Returns:
        {
            "hit_at_1": float,  # Top-1 命中率
            "hit_at_3": float,
            "hit_at_5": float,
            "mrr": float,       # Mean Reciprocal Rank
            "details": [...],   # 每个 query 的详情
        }
    """
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    reciprocal_ranks = []
    details = []

    for item in queries:
        query = item["query"]
        expect_kw = item["expect_keywords"]

        docs = retriever.invoke(query)

        # 检查每个 rank 是否命中
        found_rank = None
        for rank, doc in enumerate(docs[:5], 1):
            content = doc.page_content
            # 至少匹配 1 个期望关键词即算命中
            hits = sum(1 for kw in expect_kw if kw in content)
            if hits >= 1:
                found_rank = rank
                break

        if found_rank:
            if found_rank == 1:
                hit_at_1 += 1
            if found_rank <= 3:
                hit_at_3 += 1
            if found_rank <= 5:
                hit_at_5 += 1
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)

        details.append({
            "query": query,
            "found_rank": found_rank,
            "top_docs": [
                {
                    "rank": i,
                    "source": d.metadata.get("source", "?")[-50:],
                    "method": d.metadata.get("retrieval_method", "?"),
                    "preview": d.page_content[:80].replace("\n", " "),
                }
                for i, d in enumerate(docs[:3], 1)
            ],
        })

    n = len(queries)
    return {
        "name": retriever_name,
        "hit_at_1": hit_at_1 / n,
        "hit_at_3": hit_at_3 / n,
        "hit_at_5": hit_at_5 / n,
        "mrr": sum(reciprocal_ranks) / n,
        "details": details,
    }


def print_report(all_metrics: list):
    """打印对比报告"""
    print("\n" + "=" * 70)
    print("  检索准确率对比报告")
    print("=" * 70)

    # 汇总表
    print(f"\n{'指标':<15}", end="")
    for m in all_metrics:
        print(f"{m['name']:<15}", end="")
    print()
    print("-" * (15 + 15 * len(all_metrics)))

    for metric_name in ["hit_at_1", "hit_at_3", "hit_at_5", "mrr"]:
        label = {
            "hit_at_1": "Hit@1",
            "hit_at_3": "Hit@3",
            "hit_at_5": "Hit@5",
            "mrr": "MRR",
        }[metric_name]
        print(f"{label:<15}", end="")
        for m in all_metrics:
            print(f"{m[metric_name]:.2%}          ", end="")
        print()

    print()

    # 逐题详情
    print("─" * 70)
    print("逐题详情（以 Hybrid 结果为准）")
    print("─" * 70)
    hybrid_metrics = all_metrics[-1] if all_metrics else None
    if hybrid_metrics:
        for d in hybrid_metrics["details"]:
            status = f"Hit@Rank-{d['found_rank']}" if d["found_rank"] else "MISS"
            print(f"\n  [{status}] {d['query']}")
            for doc in d["top_docs"]:
                print(f"    Rank{doc['rank']} [{doc['method']}] {doc['source']}")
                print(f"         {doc['preview']}...")


def main():
    print("[...] 初始化向量存储 (embedding={})...".format(config.embedding_provider))

    vector_store = ChromaVectorStore(
        persist_dir=config.vector_db_dir,
        collection_name=config.collection_name,
        embedding_provider=config.embedding_provider,
        embedding_model=config.embedding_model,
        embedding_api_key=config.openai_api_key,
        embedding_base_url=config.openai_base_url,
        embedding_device=config.embedding_device,
    )

    stats = vector_store.get_collection_stats()
    print(f"[OK] 知识库就绪，文档块数: {stats['count']}")

    if stats["count"] == 0:
        print("[ERROR] 知识库为空，请先运行: python main.py ingest")
        return

    # ── 测试三种模式 ──
    all_metrics = []

    # 模式1: Dense-only (权重 dense=1.0, sparse=0.0)
    print("\n>>> 测试 Dense-only (纯向量检索)...")
    retriever_dense = HybridRetriever(
        vector_store=vector_store,
        top_k=5,
        dense_weight=1.0,
        sparse_weight=0.0,
        use_web_search=False,
    )
    metrics_dense = compute_metrics(retriever_dense, TEST_QUERIES, "Dense(BGE)")
    all_metrics.append(metrics_dense)
    print(f"    Hit@5={metrics_dense['hit_at_5']:.0%}  MRR={metrics_dense['mrr']:.3f}")

    # 模式2: Sparse-only (权重 dense=0.0, sparse=1.0)
    print(">>> 测试 Sparse-only (纯 BM25)...")
    retriever_sparse = HybridRetriever(
        vector_store=vector_store,
        top_k=5,
        dense_weight=0.0,
        sparse_weight=1.0,
        use_web_search=False,
    )
    metrics_sparse = compute_metrics(retriever_sparse, TEST_QUERIES, "Sparse(BM25)")
    all_metrics.append(metrics_sparse)
    print(f"    Hit@5={metrics_sparse['hit_at_5']:.0%}  MRR={metrics_sparse['mrr']:.3f}")

    # 模式3: Hybrid (dense=0.5, sparse=0.5 — 从 config 读取)
    print(">>> 测试 Hybrid (BGE + BM25 + RRF)...")
    retriever_hybrid = HybridRetriever(
        vector_store=vector_store,
        top_k=5,
        dense_weight=config.get("retrieval.hybrid_weights.dense", 0.5),
        sparse_weight=config.get("retrieval.hybrid_weights.sparse", 0.5),
        use_web_search=False,
    )
    metrics_hybrid = compute_metrics(retriever_hybrid, TEST_QUERIES, "Hybrid(RRF)")
    all_metrics.append(metrics_hybrid)
    print(f"    Hit@5={metrics_hybrid['hit_at_5']:.0%}  MRR={metrics_hybrid['mrr']:.3f}")

    # ── 打印报告 ──
    print_report(all_metrics)


if __name__ == "__main__":
    main()
