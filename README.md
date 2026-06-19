# ⚖️ Smart-Legal-Agent

> 基于 MiniMax 大模型、ReAct 推理引擎与 BGE 语义检索的中国法律智能 Agent。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 解决的问题

通用大模型在法律咨询中容易**编造法条**（幻觉）。本系统不依赖模型"记住"法律——而是让模型作为调度者，主动检索私有知识库中的法律法规和判例，给出可追溯到原文的分析。

---

## 真实架构

```
用户输入
  │
  ├─ 路由守卫（app.py 前置关键词）
  │    ├─ 寒暄 → 直接 LLM 回复（零 Agent 开销）
  │    └─ 法律问题 ↓
  │
  ├─ ReAct Agent 多步推理（legal_agent.py）
  │    Thought → Action → Observation → ...
  │    │
  │    ├─ search_laws  ──→ ChromaDB + BGE dense 检索 + 相似度阈值
  │    ├─ search_cases ──→ 同上（doc_type 过滤）
  │    ├─ search_web   ──→ DuckDuckGo 联网搜索
  │    └─ calculate    ──→ 《诉讼费用交纳办法》阶梯费率（确定性计算）
  │
  ├─ Critic 审核（critic.py）
  │    提取法条引用 → 本地核对（含阈值）→ 联网核验 → 脏数据域名/内容过滤
  │
  └─ 流式 UI（app.py）
       thought_token 实时渲染 + observation 绿色块 + critic 事件 + refs 引用
```

### 用到的技术

| 层 | 技术 |
|----|------|
| LLM | MiniMax-M3（`abab6.5s-chat`, OpenAI 兼容 API） |
| Embedding | BAAI/bge-small-zh-v1.5（512维，CPU 推理，ModelScope 下载） |
| 向量库 | ChromaDB（本地持久化） |
| 稀疏检索 | BM25（`rank_bm25`，字符级中文分词） |
| 混合融合 | RRF（Reciprocal Rank Fusion，`tests/` 中有完整测试） |
| 联网 | DuckDuckGo（`duckduckgo_search`） |
| 前端 | Streamlit（流式事件消费） |

---

## 真实性能

在 **99 个文档块** 的法律知识库上测试（8 题标准检索）：

| 模式 | Hit@5 |
|------|-------|
| 纯 BGE（Agent 生产路径） | 87.5% |
| 纯 BM25 | 100% |
| Hybrid RRF（`tests/test_retrieval.py`） | 100% |

测试脚本：`python tests/test_retrieval.py`

---

## 快速开始

```bash
git clone https://github.com/ysking-0-0/Smart-Legal-Agent.git
cd Smart-Legal-Agent
pip install -r requirements.txt

# 配置 MiniMax API Key
# 编辑 .env：OPENAI_API_KEY=sk-cp-xxx  OPENAI_BASE_URL=https://api.minimaxi.com/v1

# 导入知识库
python main.py ingest

# 启动
python -m streamlit run app.py --server.port 8501
```

---

## 真实测试示例

### 示例 1：诉讼费计算

```
输入：标的额150万，诉讼费多少？

Agent 推理过程：
  Step 1: Thought → Action: calculate → Action Input: 1500000
          Observation: 标的额 1,500,000 元，按《诉讼费用交纳办法》阶梯计算...
          → 案件受理费合计：18,300 元

  Critic: 审核通过（无联网触发）

输出：
  根据《诉讼费用交纳办法》，标的额 150 万元的财产案件受理费为 18,300 元。
  分档明细：≤1万→50元 | 1-10万→2,250元 | 10-20万→2,000元 | 20-50万→4,500元 | 50-100万→5,000元 | 100-150万→4,500元
```

### 示例 2：酒驾法律分析

```
输入：酒驾吹气95，怎么处理？

Agent 推理过程：
  Step 1: Thought → Action: search_laws → 检索《道路交通安全法》第91条
  Step 2: Thought → Action: search_cases → 检索危险驾驶罪案例
  Step 3: Final Answer（含法条引用 + 案例参考）

  Critic: 本地核对通过（道路交通安全法在库）

输出：
  醉酒驾驶（血液酒精含量≥80mg/100ml）构成危险驾驶罪。
  依据《中华人民共和国道路交通安全法》第91条：吊销驾驶证，依法追究刑事责任，5年内不得重考。
  依据《中华人民共和国刑法》第133条之一：危险驾驶罪，处拘役并处罚金。
  参考案例：（2023）京0105刑初1258号 张某危险驾驶案，血液酒精含量187mg/100ml，判处拘役2个月并处罚金5000元。
```

---

## 知识库添加

```bash
# 法条放 data/laws/，案例放 data/cases/
cp 新法条.md data/laws/
python main.py ingest
```

分块器自动按"第X条/章/节"边界做条款级语义切片，单条法条超过 500 字也保持完整不切断。

---

## 后续方向

- [ ] 将 Hybrid RRF 融合从测试接入 Agent 生产检索链路
- [ ] 引入 bge-reranker 精排提升 Hit@1
- [ ] 扩展计算器：迟延履行利息、工伤赔偿公式
- [ ] Docker 一键部署
