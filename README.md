# ⚖️ Smart-Legal-Agent

> 基于 MiniMax 大模型、ReAct 推理引擎、BGE+BM25+RRF 混合检索的中国法律智能 Agent。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 解决的问题

通用大模型在法律咨询中**编造法条**（幻觉）、**口算诉讼费**（符号计算弱）、**训练数据滞后**。本系统不依赖模型记忆法律——模型作为调度者，主动检索私有知识库并调用确定性计算工具，所有法条引用可追溯到原文。

---

## 架构

```
用户输入
  │
  ├─ 路由守卫（app.py 前置关键词）→ 寒暄秒回 / 法律问题进入 Agent
  │
  ├─ ReAct Agent（legal_agent.py）
  │    Thought → Action → Observation → ...
  │    │
  │    ├─ search_laws  ──→ Hybrid RRF（BGE dense + BM25 sparse）→ ChromaDB
  │    ├─ search_cases ──→ 同上
  │    ├─ search_web   ──→ DuckDuckGo 联网（自建，非模型内置插件）
  │    └─ calculate    ──→ 《诉讼费用交纳办法》阶梯费率（确定性计算）
  │
  ├─ Critic 审核（critic.py）
  │    提取法条引用 → 本地核对（含相似度阈值）→ 联网核验
  │    → 脏数据域名/内容多层过滤 → 仅采信权威法律源
  │
  └─ 流式 UI（app.py）
       thought_token 实时渲染 / observation 绿色块 / critic 事件 / refs 引用徽章
```

### 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| LLM | MiniMax-M3 | OpenAI 兼容 API, `abab6.5s-chat` |
| Embedding | BAAI/bge-small-zh-v1.5 | 512维, CPU 推理, ModelScope 下载 |
| 密集检索 | ChromaDB + BGE | L2 距离, 阈值 1.2 |
| 稀疏检索 | BM25 (rank_bm25) | 字符级中文分词 |
| 融合重排 | RRF (k=60) | 生产路径已实装, dense=0.5 sparse=0.5 |
| 联网 | DuckDuckGo (duckduckgo_search) | Python 自建, 非模型内置 |
| 前端 | Streamlit | 流式事件消费 |

---

## 💡 核心设计哲学与技术协同（为什么采用这些组合？）

在垂直法律行业落地 RAG 存在天然的"容错率极低"死穴。本系统拒绝盲目堆砌技术，每一个组件的引入都为了克制特定的工程痛点，形成了强大的协同效应：

### 1. 为什么用【BGE 密集向量 + BM25 稀疏文本】组合？

- **BGE Dense（语义理解）**：负责处理用户的"模糊口语化表达"。例如用户输入"被撞进女厕所看光了"，BGE 能通过语义向量空间关联到"隐私权纠纷"。
- **BM25 Sparse（精准对齐）**：法律条文极其严谨，字面失之毫厘谬以千里。当输入"第九十一条"或特定罪名时，BM25 能够通过精确的字面词频（TF-IDF 变体）瞬间咬死目标切片。
- **协同效果**：通过 **RRF（倒数排名融合）** 算法将两路召回强行对齐，做到了**"既懂用户口语的深层含义，又能精准锁死严谨的法条字面"**，将系统本地 Hit@5 检索准确率直接从 87.5% 飙升至 100%。

### 2. 为什么用【大模型推理 + 纯 Python 计算工具】组合？

- **痛点**：大模型在处理《诉讼费用交纳办法》这种高度复杂的"多段阶梯累进税率"时，由于其 Token 预测的本质，口算准确率极低，且温度稍高就会胡编数字。
- **解法**：模型作为"总调度大脑"（基于 ReAct 范式），一旦识别到财产标的，立即阻断文本生成，将数字剥离并推给后端纯 Python 编写的确定性计算函数，**用确定性的传统代码作为安全垫，确保核心数学计算 100% 准确**。

### 3. 为什么用【Generator + Critic 双 Agent 反思环】组合？

- **痛点**：传统 RAG 系统中，模型拿到检索内容后直接进行生成，缺乏"内容合规审查"，经常发生法条张冠李戴、乱套罪名的幻觉。
- **解法**：引入裁判机制。Generator 负责生成律师意见书初稿；Critic 作为独立节点，利用正则强制抠出初稿中的法条引用，与本地库（有书）或 DuckDuckGo 联网抓取的数据（无书）进行一字不差的**硬核交叉比对（Cross-Check）**。发现幻觉直接拦截并打回重写，构成了后端的柔性抗幻觉防御网。

---

## 快速开始

```bash
git clone https://github.com/ysking-0-0/Smart-Legal-Agent.git
cd Smart-Legal-Agent
pip install -r requirements.txt

# 配置 MiniMax API Key (.env)
OPENAI_API_KEY=sk-cp-xxx
OPENAI_BASE_URL=https://api.minimaxi.com/v1

# 导入知识库
python main.py ingest

# 启动
python -m streamlit run app.py --server.port 8501
```

---

## 测试

```bash
# 检索准确率（Dense / BM25 / Hybrid RRF 对比）
python tests/test_retrieval.py

# Agent CLI 模式
python main.py agent

# API 服务
python main.py api --port 8000
```

### 检索性能（99 chunks, 8 题）

| 模式 | Hit@5 | 说明 |
|------|-------|------|
| 纯 BGE Dense | 87.5% | 旧生产路径 |
| 纯 BM25 | 100% | 小库关键词优势 |
| **Hybrid RRF** | **100%** | **当前生产路径** |

---

## 知识库

```
data/
├── laws/        → doc_type=law
├── cases/       → doc_type=case
└── custom/      → doc_type=custom
```

```bash
cp 新法条.md data/laws/
python main.py ingest
```

分块器按"第X条/章/节"边界做条款级语义切片，单条超 500 字保持完整。

---

## 示例

### 诉讼费计算

```
输入：标的额150万，诉讼费多少？

Agent: Action: calculate → Action Input: 1500000
       Observation: 合计 18,300 元（分档明细）

输出：18,300 元，依据《诉讼费用交纳办法》第十三条
```

### 酒驾法律分析

```
输入：酒驾吹气95，怎么处理？

Agent: Step1 search_laws → 道路交通安全法第91条
       Step2 search_cases → 张某危险驾驶案(2023)京0105刑初1258号
       Step3 Final Answer + Critic 本地核对通过

输出：醉酒驾驶构成危险驾驶罪，吊销驾照+刑事处罚
      参考案例：张某案, 187mg/100ml → 拘役2月+罚金5000元
```

---

## 后续

- [ ] bge-reranker 精排（提升 Hit@1）
- [ ] 扩展计算器：迟延履行利息、工伤赔偿
- [ ] Docker 一键部署
