# ⚖️ Smart-Legal-Agent

> 基于大语言模型、自研 ReAct 推理引擎与 BGE 语义检索的智能法律 Agent —— 让 AI 真正理解中国法律。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4+-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.5+-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.58+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 1. 项目简介

通用大模型在法律垂直领域面临三大核心痛点：**高幻觉率**（编造不存在的法条）、**数据滞后**（训练数据截止导致新法缺失）、**符号计算弱**（诉讼费、赔偿金口算错误）。

**Smart-Legal-Agent** 是一个面向中国法律场景的智能 RAG Agent，它不依赖大模型"记住"所有法条，而是通过 **ReAct 多步推理引擎** 主动检索私有知识库中的法律法规与判例，逐条引用、逐案比对，最终给出可追溯的法律分析。

### 核心能力

| 能力 | 说明 |
|------|------|
| 🔍 **事实 → 法条推理** | 用户描述案情，Agent 自动提取法律要素，检索适配法条 |
| ⚖️ **案例比对** | 检索相似判例，以案说法，提供量刑参考 |
| 🛡️ **低幻觉** | 所有法条引用来自私有知识库，可追溯到原文出处 |
| 💬 **智能路由** | 自动区分寒暄/法律咨询，普通对话不浪费检索资源 |
| 🌊 **流式输出** | 思考过程 + 分析结论实时逐字呈现 |

---

## 🏗️ 2. 核心架构设计

```
用户输入
    │
    ▼
┌──────────────────────────────────────────┐
│  ① 意图路由 (Intent Router)               │
│     寒暄? → 直接回复                       │
│     法律咨询? → 启动 ReAct 推理引擎         │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  ② 双路混合检索 (Hybrid Retrieval)         │
│                                             │
│  Dense (BGE-small-zh-v1.5)                  │
│       │                                     │
│       ├── RRF 融合 ──→ Top-5 法条           │
│       │                                     │
│  Sparse (BM25)                              │
│                                             │
│  知识库: 自适应"条款级"语义切片              │
│  (按"第X条/章/节"边界切割, 保持完整性)       │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  ③ ReAct 工具链调度                        │
│                                             │
│  Thought → Action → Observation → Loop      │
│     │          │           │                │
│     │    search_laws  search_cases          │
│     │    search_web   计算器(规划中)         │
└──────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────┐
│  ④ 分析结论输出                            │
│     法条定性 + 处罚说明 + 案例参考 + 建议    │
│     出处透明可追溯 → 流式逐字渲染           │
└──────────────────────────────────────────┘
```

### 四大技术亮点

- **意图路由与任务规划**：前置中央处理器精准识别寒暄、法律咨询、计算和起草意图，普通对话零开销。

- **自适应语义切片与双路混合检索**：正则匹配"第X条/章/节"边界进行条款级自适应切片；BGE 密集向量检索 + BM25 稀疏关键词检索，通过 **RRF（Reciprocal Rank Fusion）** 融合，兼具语义理解和精确匹配能力。

- **工具链集成**：基于 ReAct 推理范式的自主工具调度，Agent 自己决定何时检索法条、何时查找案例、何时联网补充。

- **全链路流式事件响应**：异步流式消费事件机制，思考过程实时可见。初始化阶段预加载 BGE 模型避免冷启动卡顿。

---

## 📊 3. 性能表现与测试数据

在 **99 chunks** 的法律知识库（覆盖道路交通安全法、刑法、劳动合同法、常见案例）上进行 8 题标准检索测试：

```
指标          Dense(BGE)    Sparse(BM25)   Hybrid(RRF)
──────────────────────────────────────────────────────
Hit@1         87.5%         87.5%          75.0%
Hit@3         87.5%         100.0%         87.5%
Hit@5         87.5%         100.0%         100.0%  ✅
MRR           87.5%         93.8%          84.4%
```

### 关键发现

- **Hybrid Hit@5 = 100%**：混合检索完美弥补了纯向量检索在特定罪名、专有名词上的漏检（MISS）。例如"个人信息被泄露"在 Dense 模式漏检，Hybrid 成功召回。

- **BM25 在小规模知识库优势显著**：99 chunks 场景下，关键字匹配天然优于语义向量。随着知识库规模增长，BGE 的语义泛化优势将逐步体现。

- **RRF 融合的取舍**：Hit@5 提升的代价是 MRR 和 Hit@1 略降 —— 融合引入了多样性，首位命中率需要 Reranker 精排来补强（见 Roadmap）。

---

## 📥 4. 知识库数据灌录指南

### 目录结构

```
data/
├── laws/        ← 法律法规文件（自动标记为 doc_type=law）
├── cases/       ← 案例判决文件（自动标记为 doc_type=case）
└── custom/      ← 自定义问答/分析文档（自动标记为 doc_type=custom）
```

### 标准操作流程

**Step 1 — 准备文件**

推荐使用 Markdown 格式，按"条款"结构编写：

```markdown
# 中华人民共和国个人信息保护法

## 第一章 总则

第一条 为了保护个人信息权益，规范个人信息处理活动...

第二条 自然人的个人信息受法律保护，任何组织、个人不得侵害...

## 第二章 个人信息处理规则

第十三条 符合下列情形之一的，个人信息处理者方可处理个人信息：
（一）取得个人的同意；
（二）为订立、履行个人作为一方当事人的合同所必需...
```

分块器会自动按"第X条/章/节"边界做语义完整切片，单条法条即使超过 500 字也保持完整不切断。

**Step 2 — 放入目录**

```bash
# 法条
cp 个人信息保护法.md data/laws/

# 案例
cp 某侵犯公民个人信息案.md data/cases/
```

**Step 3 — 运行导入**

```bash
# 增量导入（保留已有数据）
python main.py ingest

# 清空重建（切换 embedding 模型或修改分块配置时）
python main.py ingest --clear
```

导入过程：文档加载 → 条款级自适应切片 → BGE 向量化 → ChromaDB 持久化写入。99 chunks 约 5-10 秒完成。

---

## 🚀 5. 如何快速开始

### 环境要求

- Python 3.10+
- 能访问 `api.deepseek.com`（或任何 OpenAI 兼容 API）
- 首次启动需下载 BGE 模型（~95 MB，走 ModelScope 国内通道）

### 安装

```bash
git clone https://github.com/ysking-0-0/Smart-Legal-Agent.git
cd Smart-Legal-Agent
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 配置

在项目根目录创建 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-deepseek-api-key
OPENAI_BASE_URL=https://api.deepseek.com/v1
```

### 导入知识库

```bash
python main.py ingest
```

### 启动

```bash
# Web 界面（推荐）
python -m streamlit run app.py --server.port 8501

# CLI Agent 模式
python main.py agent

# API 服务
python main.py api --port 8000
```

打开 `http://localhost:8501`，试试问：

> "我昨晚喝了三瓶啤酒开车回家，被交警查了，吹气95，会怎么处理？"

Agent 会实时展示推理过程（查法条 → 查案例 → 综合分析），最后流式输出法律结论。

---

## 🗺️ 6. 后续改进方向

- [ ] **精密计算器路由**：引入严格遵循《诉讼费用交纳办法》阶梯费率的确定性 Python 计算工具链，杜绝大模型口算诉讼费/赔偿金。

- [ ] **双 Agent 反思校验模块（Generator-Critic）**：生成回答后，Critic Agent 强制提取输出中的法条引用实体，与 ChromaDB 原文做一字不差的精确匹配校验。发现幻觉自动打回 Generator 修正，最多 3 轮闭环。

- [ ] **Reranker 精排**：接入 `bge-reranker-v2-m3` 对双路检索的候选法条做二次精细化打分，将 Hit@1 从当前 75%（Hybrid）提升至 90%+。

- [ ] **文书起草路由**：扩展意图识别，支持"帮我起草一份劳动合同解除协议"类长文本生成，接入法律模板引擎。

- [ ] **知识库管理界面**：在 Streamlit 侧边栏增加文档上传、删除、统计的可视化管理。

- [ ] **Docker 一键部署**：提供 Dockerfile 和 docker-compose.yml，实现生产环境开箱即用。

---

## 📄 License

MIT License

---

<p align="center">
  <sub>Built with LangChain · ChromaDB · BGE · DeepSeek · Streamlit</sub>
</p>
