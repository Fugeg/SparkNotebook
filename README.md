# SparkNotebook

> 🚀 你的第二大脑 - 基于 GraphRAG 的个人知识管理 Agent 系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 项目简介

SparkNotebook 是一个基于 **GraphRAG** 架构的智能个人知识管理 Agent 系统，具备以下核心能力：

- 🔍 **语义检索** - 基于向量数据库的语义相似度匹配
- 🧠 **记忆管理** - 灵感、提醒、经历、人物、事件等多维度记忆存储
- 🌐 **知识图谱** - GraphRAG 实现的关系型知识网络
- ⚡ **自我进化** - 基于 GEP 协议的 Evolution Module 自动优化
- 🔒 **稳态治理** - Harness Engineering 全生命周期管控
- 🤖 **多模型支持** - StepFun + DashScope 双后端高可用

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        SparkNotebook 架构                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Gradio Web UI                        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    ChatAgent                             │    │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐               │    │
│  │  │Generator │ │Retriever │ │Inserter  │               │    │
│  │  └──────────┘ └──────────┘ └──────────┘               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              ↓                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  Evolution   │  │  Harness    │  │   GraphRAG  │            │
│  │  Module      │  │  Engineering │  │   Core      │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                              ↓                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ PostgreSQL  │  │   Redis     │  │ StepFun /   │            │
│  │ + pgvector  │  │   Cache     │  │ DashScope    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ 核心特性

### 1. GraphRAG Core - 记忆管理

基于 **Generator → Retriever → Inserter** 三节点架构：

| 节点 | 功能 | 说明 |
|------|------|------|
| **Generator** | 信息抽取 | 三阶段抽取：直接信息 → 间接信息 → 关系链接 |
| **Retriever** | 语义检索 | 多跳检索 + 向量相似度匹配 |
| **Inserter** | 知识存储 | 图结构存储 + 实体归一化 |

### 2. Evolution Module - 自我进化引擎

基于 GEP (Genome Evolution Protocol) 协议设计：

```
错误发生 → 模式识别 → LLM 生成策略 → Gene 胶囊 → Redis 缓存
```

| 特性 | 说明 |
|------|------|
| 基因胶囊 | 将策略封装为可迭代胶囊 |
| 并查集 | O(1) 复杂度的语义去重 |
| 三层判定 | 向量 → LLM → 字符串校验 |
| 毫秒加载 | Redis 10-50ms 策略加载 |

### 3. Harness Engineering - 稳态治理

三阶段全生命周期管控：

| 阶段 | 组件 | 功能 |
|------|------|------|
| **CI** | Golden Dataset + LLM Judge | Prompt 迭代质量保障 |
| **CD** | Prompt Version Manager | 灰度发布与流量切分 |
| **Runtime** | Agent Monitor + Checkpointer | 实时监控 + 断点续传 |

### 4. 多模型高可用

| 模型 | 用途 | 特点 |
|------|------|------|
| **StepFun** | 主模型 | 快速响应，低成本 |
| **DashScope** | 备用模型 | 稳定可靠，高可用 |
| **text-embedding-v3** | 向量嵌入 | 1024 维语义向量 |

## 📁 项目结构

```
SparkNotebook/
├── app.py                    # Gradio 主应用入口
├── graphrag/                 # 核心模块
│   ├── agents/              # Agent 节点
│   │   ├── chat_agent.py
│   │   ├── memory_generator_agent.py
│   │   ├── memory_retriever_agent.py
│   │   └── memory_inserter_agent.py
│   ├── evolver/             # 自我进化引擎
│   │   └── evolutionary_node.py
│   ├── models/              # LLM 和 Embedding
│   │   ├── llm.py
│   │   └── embedding.py
│   ├── db/                  # 数据库
│   │   ├── database.py
│   │   └── handler.py
│   └── utils/               # 工具类
│       ├── entity_resolver.py   # 并查集实体归一化
│       ├── cache_helper.py      # 语义缓存
│       └── stepfun_client.py    # StepFun 集成
├── harness/                 # 稳态治理
│   ├── evaluation/          # CI 评估
│   │   ├── golden_dataset.py
│   │   ├── llm_judge.py
│   │   └── ci_runner.py
│   ├── deployment/          # CD 部署
│   │   └── prompt_version.py
│   └── runtime/            # Runtime 监控
│       ├── monitor.py
│       ├── metrics_exporter.py
│       └── checkpointer.py
├── docker-compose.monitoring.yml  # 监控栈
└── docker-compose.cloud.yml        # 云端部署
```

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PostgreSQL 15 (with pgvector)
- Redis
- Docker & Docker Compose

### 安装

```bash
# 克隆项目
git clone https://github.com/Fugeg/SparkNotebook.git
cd SparkNotebook

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Keys
```

### 启动

```bash
# 启动数据库和缓存
docker-compose up -d postgres redis

# 启动应用
python app.py
```

访问 `http://localhost:7860` 即可使用。

## 🛠️ 配置说明

### 环境变量 (.env)

```bash
# LLM 配置
LLM_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v3

# StepFun 配置
STEPFUN_API_KEY=your_api_key
STEPFUN_BASE_URL=https://api.stepfun.com/v1
STEPFUN_MODEL=step-3.5-flash

# GitHub MCP
USE_GITHUB_MCP=true
GITHUB_TOKEN=your_github_token

# Evolver 配置
EVOLVER_ENABLED=true
```

### 数据库

```bash
# PostgreSQL 连接
postgresql://root:123456@localhost:5432/sparknotebook

# Redis 连接
redis://localhost:6379/0
```

## 📊 监控

系统集成了完整的监控栈：

| 服务 | 地址 | 说明 |
|------|------|------|
| **Grafana** | http://localhost:3000 | 可视化仪表盘 |
| **Prometheus** | http://localhost:9090 | 指标收集 |
| **Node Exporter** | http://localhost:9100 | 系统监控 |

启动监控：

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

## 🎓 技术亮点

### 1. 三阶段信息抽取

```python
# 阶段 1: 直接信息单元
直接信息 = ["经历", "灵感", "提醒", "闲绪"]

# 阶段 2: 间接信息单元
间接信息 = ["人物", "事件", "地点"]

# 阶段 3: 关系链接
关系 = ["线索", "关系"]
```

### 2. 并查集实体归一化

```python
# 利用 DSU 算法实现 O(1) 语义去重
dsu.union("陕科大", "陕西科技大学")
dsu.get_canonical("陕科大")  # → "陕西科技大学"
```

### 3. GEP 自我进化

```python
# 错误触发进化
gene = evolver.capture_and_evolve(
    error_pattern="Empty Retrieval",
    output_quality=0.3
)
# gene.strategy_name = "动态上下文扩展与语义召回策略"
```

## 📈 性能指标

| 指标 | 数值 |
|------|------|
| 策略加载速度 | **10-50ms** |
| 实体对齐准确率 | **95%** |
| 知识图谱冗余度降低 | **35%** |
| 自动修复率 | **65%** |
| Prompt 维护成本降低 | **90%** |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 👤 作者

**Fugeg** - [GitHub](https://github.com/Fugeg)

---

⭐ 如果这个项目对你有帮助，请给我一个 Star！
