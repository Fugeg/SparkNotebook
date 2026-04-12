# Fugeg · AI 记事本助手 - 项目技术文档

## 目录
1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心模块详解](#3-核心模块详解)
4. [技术栈](#4-技术栈)
5. [数据库设计](#5-数据库设计)
6. [核心功能实现](#6-核心功能实现)
7. [评估工具链](#7-评估工具链)
8. [关键技术创新点](#8-关键技术创新点)
9. [部署与运行](#9-部署与运行)
10. [未来优化方向](#10-未来优化方向)

---

## 1. 项目概述

### 1.1 项目背景
本项目是本科毕业设计「基于 GraphRAG 的 AI 个人知识管理助手」的完整实现。目标是构建一个智能化的个人知识管理平台，让用户能够通过自然语言与 AI 助手交互，实现信息的智能记录、存储和检索。

### 1.2 项目名称
**Fugeg · AI 记事本助手** (Fugeg AI Notepad Assistant)

### 1.3 核心定位
- **目标用户**：需要进行个人信息管理、知识积累的学生和职场人士
- **核心价值**：将碎片化的自然语言输入转化为结构化的知识图谱
- **技术特色**：结合知识图谱与 RAG 技术的多智能体系统

### 1.4 主要功能
| 功能模块 | 描述 |
|---------|------|
| 意图分类 | 自动识别用户输入的意图（闲聊/记录/检索） |
| 记忆提取 | 将非结构化文本解析为结构化信息单元 |
| 知识存储 | 基于图数据库的多模态知识存储 |
| 记忆检索 | 多跳图检索 + 向量相似度混合检索 |
| 语音输入 | 支持语音转文本的多种输入方式 |
| 多用户支持 | 基于角色的用户隔离与权限管理 |

---

## 2. 系统架构

### 2.1 整体架构图
```
┌─────────────────────────────────────────────────────────────────┐
│                        用户界面层 (Gradio UI)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  闲聊模式    │  │  记事本模式  │  │      语音输入           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       核心调度层 (ChatAgent)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ 意图分类器   │  │ 上下文管理器 │  │      对话历史管理       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────────┐
│  MemoryGen    │   │  MemoryInserter │   │ MemoryRetriever     │
│   Agent       │   │     Agent       │   │     Agent           │
│ (记忆生成)     │   │  (记忆插入)      │   │   (记忆检索)         │
└───────────────┘   └─────────────────┘   └─────────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LLM 模型层 (DashScope API)                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Qwen Plus   │  │ Text-Emb-V3  │  │    Paraformer-V2       │ │
│  │ (对话生成)   │  │  (向量化)     │  │     (语音识别)          │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       数据持久层 (PostgreSQL + pgvector)          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  节点表群    │  │   边表      │  │     用户表              │ │
│  │ experiences │  │   edges     │  │      users              │ │
│  │ inspirations │  │ connections │  │                        │ │
│  │ reminders    │  └─────────────┘  └─────────────────────────┘ │
│  │ people       │                                              │
│  │ events       │                                              │
│  │ places       │                                              │
│  └─────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 智能体架构
```
ChatAgent (总调度智能体)
    │
    ├── IntentClassifier (意图分类)
    │   └── 三分类：闲聊(1) / 记录(2) / 检索(3)
    │
    ├── MemoryGeneratorAgent (记忆生成)
    │   ├── 信息提取 (extract_information)
    │   ├── JSON 解析与清洗
    │   └── 嵌入向量生成
    │
    ├── MemoryInserterAgent (记忆插入)
    │   ├── 节点持久化
    │   ├── 关系边创建
    │   └── 相似记忆自动关联
    │
    └── MemoryRetrieverAgent (记忆检索)
        ├── 向量相似度召回
        ├── 多跳图扩展
        └── LLM 节点筛选
```

---

## 3. 核心模块详解

### 3.1 ChatAgent (`graphrag/agents/chat_agent.py`)

**职责**：系统的中央调度器，协调各子智能体完成用户请求。

**核心特性**：
- **意图分类路由**：根据识别到的意图分发给不同的处理模块
- **对话上下文管理**：维护最近 10 轮对话历史，支持多轮对话
- **用户身份识别**：动态获取数据库中的用户名，个性化响应
- **输入方式统一**：支持文本输入和语音输入的统一处理

**关键方法**：
```python
def handle_input(self, user_input, user_id=1):
    """处理用户文本输入 - 核心入口"""

def handle_chat(self, user_input, context="", username="用户"):
    """处理闲聊模式 - 意图 1"""

def handle_memory_creation(self, user_input, input_method, user_id, context, username):
    """处理记忆记录 - 意图 2"""

def handle_memory_retrieval(self, user_input, user_id, context, username):
    """处理记忆检索 - 意图 3"""
```

### 3.2 MemoryGeneratorAgent (`graphrag/agents/memory_generator_agent.py`)

**职责**：将用户非结构化输入解析为结构化知识单元。

**处理流程**：
1. 调用 LLM 的 `extract_information` 方法进行信息提取
2. 清理 LLM 返回的 Markdown JSON 格式
3. 为每个信息单元生成 1024 维嵌入向量

**信息单元类型**：
| 类型 | 描述 | 示例 |
|-----|------|-----|
| 经历 | 用户描述的事件 | "今天参加了学术会议" |
| 灵感 | 想法或创意 | "可以用这个技术改进产品" |
| 提醒 | 待办或目标 | "明天要交论文初稿" |
| 闲绪 | 情绪表达 | "今天心情不错" |
| 人物 | 姓名实体 | "遇到了张老师" |
| 事件 | 活动名称 | "未来科技论坛" |
| 地点 | 地理位置 | "深圳大学" |
| 关系 | 人际关系 | "张老师是我的导师" |
| 线索 | 单元间的关联 | "这个想法与之前的技术博客相关" |

### 3.3 MemoryInserterAgent (`graphrag/agents/memory_inserter_agent.py`)

**职责**：将结构化信息持久化到图数据库。

**核心逻辑**：
1. **节点持久化**：将信息单元写入对应类型的表
2. **关系建立**：创建节点间的边关系
3. **智能关联**：自动发现并连接与已有记忆相似的节点

**自动关联机制**：
```python
def _establish_connections(self, new_nodes, user_id):
    """对新节点进行语义相似度搜索，自动建立知识连接"""
    # 1. 生成新节点嵌入
    # 2. 在现有知识库中搜索 top-3 相似节点
    # 3. 使用 LLM 判断是否应建立连接
    # 4. 创建线索(Clue)节点并建立双向边
```

### 3.4 MemoryRetrieverAgent (`graphrag/agents/memory_retriever_agent.py`)

**职责**：实现基于 GraphRAG 的多跳记忆检索。

**检索算法**（多跳检索）：
```
步骤 1: 向量召回 (Vector Recall)
    - 生成查询的 1024 维嵌入
    - 在所有表执行相似度搜索
    - 返回 top-k 种子节点

步骤 2: 图扩展 (Graph Expansion)
    - 使用 LLM 评估种子节点的价值
    - 分类为：扩展源 / 上下文补充 / 剪枝
    - 获取扩展节点的邻居节点

步骤 3: 迭代扩展 (Iterative Expansion)
    - 重复步骤 2 直到达到 max_hops (默认 3)
    - 使用 visited 集合防止重复访问
    - 累积相关节点作为检索结果
```

**关键参数**：
```python
MAX_HOPS = 3          # 最大跳数
MAX_TOKENS = 2000     # LLM 上下文限制
TOP_K = 5             # 每次召回的节点数
```

### 3.5 LLMModel (`graphrag/models/llm.py`)

**职责**：封装与 DashScope API 的交互，封装多种 LLM 能力。

**核心方法**：
| 方法 | 功能 | 模型 |
|-----|------|-----|
| `classify_intent()` | 意图三分类 | Qwen-Plus |
| `extract_information()` | 信息结构化提取 | Qwen-Plus |
| `chat()` | 闲聊对话生成 | Qwen-Plus |
| `generate_response()` | 基于记忆的回答 | Qwen-Plus |
| `speech_to_text()` | 语音转文本 | Paraformer-V2 |

### 3.6 EmbeddingModel (`graphrag/models/embedding.py`)

**职责**：生成文本的向量表示。

**配置**：
- **模型**：text-embedding-v3
- **维度**：1024 维
- **服务**：DashScope Text Embedding API

---

## 4. 技术栈

### 4.1 后端框架
| 组件 | 技术选型 | 说明 |
|-----|---------|------|
| 主程序 | Python 3.x | 项目主体开发语言 |
| Web 框架 | Gradio 4.x | 快速构建 Web 界面 |
| 数据库 | PostgreSQL 16 | 关系型数据存储 |
| 向量扩展 | pgvector 0.7 | PostgreSQL 向量索引插件 |

### 4.2 AI/ML 服务
| 服务 | 提供商 | 用途 |
|-----|-------|-----|
| Qwen-Plus | 阿里云 DashScope | 大语言模型对话 |
| Text-Embedding-V3 | 阿里云 DashScope | 文本向量化 |
| Paraformer-V2 | 阿里云 DashScope | 语音识别 |

### 4.3 外部服务
| 服务 | 用途 |
|-----|-----|
| OSS (对象存储) | 音频文件的临时存储 |
| .env 配置 | API 密钥与环境变量管理 |

### 4.4 依赖包
```
gradio>=4.0.0      # Web 界面框架
psycopg2-binary    # PostgreSQL 驱动
pgvector           # 向量数据库扩展
dashscope          # 阿里云模型服务
numpy              # 数值计算
python-dotenv      # 环境变量管理
oss2               # 阿里云 OSS SDK
```

---

## 5. 数据库设计

### 5.1 ER 图
```
┌─────────────┐       ┌──────────────────┐       ┌─────────────┐
│   users     │       │     edges        │       │ connections │
├─────────────┤       ├──────────────────┤       ├─────────────┤
│ id (PK)     │◄──────│ user_id (FK)     │       │ id (PK)     │
│ username    │       │ connection_id(FK)│       │ user_id(FK) │
│ email       │       │ node_type        │       │ content     │
│ created_at  │       │ node_id          │──────►│ emb_vector  │
└─────────────┘       └──────────────────┘       │ conn_type   │
      │                                            └─────────────┘
      │
      │ 1:N
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        知识节点表群                              │
├──────────────┬──────────────┬──────────────┬──────────────────┤
│ experiences  │  inspirations │  reminders   │  people         │
├──────────────┼──────────────┼──────────────┼──────────────────┤
│ id (PK)      │ id (PK)      │ id (PK)      │ id (PK)          │
│ user_id (FK) │ user_id (FK) │ user_id (FK) │ user_id (FK)     │
│ content      │ content      │ content      │ name             │
│ emb_vector   │ emb_vector   │ emb_vector   │ profile_emb_vec  │
└──────────────┴──────────────┴──────────────┴──────────────────┘

┌──────────────┬──────────────┬──────────────┐
│   events     │    places    │   misc       │
├──────────────┼──────────────┼──────────────┤
│ id (PK)      │ id (PK)      │ id (PK)      │
│ user_id (FK) │ user_id (FK) │ user_id (FK) │
│ title        │ name         │ content      │
│ event_emb    │ place_emb    │ emb_vector   │
└──────────────┴──────────────┴──────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       raw_inputs (对话历史)                      │
├─────────────────────────────────────────────────────────────────┤
│ id (PK)    │ user_id (FK) │ main_content │ audio_link          │
│ input_method │ response_content │ created_at                     │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 表结构详解

**users 表**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**知识节点表（以 experiences 为例）**
```sql
CREATE TABLE experiences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    content TEXT NOT NULL,
    content_embedding VECTOR(1024),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**connections 表（关系存储）**
```sql
CREATE TABLE connections (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    content TEXT,
    content_embedding VECTOR(1024),
    connection_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**edges 表（关系边）**
```sql
CREATE TABLE edges (
    id SERIAL PRIMARY KEY,
    connection_id INTEGER REFERENCES connections(id),
    node_type VARCHAR(50),
    node_id INTEGER,
    user_id INTEGER REFERENCES users(id)
);
```

### 5.3 向量索引
使用 pgvector 的 `<->` 欧氏距离操作符进行相似度搜索：
```sql
SELECT id, content, embedding <-> '[query_vector]' as distance
FROM experiences
WHERE user_id = 1
ORDER BY distance
LIMIT 5;
```

---

## 6. 核心功能实现

### 6.1 意图分类系统

**三分类定义**：
| 意图 | 触发条件 | 处理模块 |
|-----|---------|---------|
| 1 (闲聊) | 问候、通用问答、简单陈述 | handle_chat |
| 2 (记录) | 包含"记住"、"提醒"等关键词 | handle_memory_creation |
| 3 (检索) | 询问过去信息、"你还记得..." | handle_memory_retrieval |

**Prompt 设计**：
```
# Role
你是一个高精度的意图分类引擎。

# 关键区分规则
1. "我是xxx" → 闲聊(1)，不是记录
2. "我喜欢xxx" → 闲聊(1)，"记住我喜欢xxx" → 记录(2)
3. 输出仅返回数字 1、2 或 3
```

### 6.2 记忆提取系统

**信息提取流程**：
```
用户输入: "我下周三要去深圳出差，记得提醒我带电脑"
                    │
                    ▼
        ┌───────────────────────────┐
        │   extract_information()   │
        │   (使用 Qwen-Plus LLM)     │
        └───────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │   JSON 解析与清洗         │
        │   - 移除 Markdown 标记     │
        │   - 处理格式错误          │
        └───────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │   生成结构化信息单元       │
        │   [                       │
        │     {type: "经历",        │
        │      content: "下周三去.."}│
        │     {type: "地点",        │
        │      content: "深圳"},    │
        │     {type: "提醒",        │
        │      content: "带电脑"}   │
        │   ]                       │
        └───────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │   生成嵌入向量             │
        │   (1024 维 text-emb-v3)   │
        └───────────────────────────┘
```

### 6.3 知识图谱构建

**自动关联机制**：
```
新节点 → 嵌入 → 相似度搜索 → LLM 判断 → 创建线索节点
                                    │
                                    ▼
                              建立双向边连接
```

**LLM 判断提示**：
```
判断以下两条信息是否存在有意义的关联：
新信息：[新节点内容]
已有信息：[候选节点内容]

如果存在有意义的关联（如因果、上下文、主题相关等），
请回答"是"，否则回答"否"。
```

### 6.4 多跳检索实现

**检索算法伪代码**：
```python
def multi_hop_retrieval(query, seed_nodes, max_hops=3):
    visited = set()
    results = []
    current_nodes = seed_nodes

    for hop in range(max_hops):
        # LLM 筛选有价值节点
        valuable = llm_filter_nodes(query, current_nodes)

        for node in valuable:
            if node.id not in visited:
                visited.add(node.id)
                results.append(node)

        # 扩展邻居
        next_nodes = []
        for node in valuable:
            neighbors = db.get_neighbors(node)
            next_nodes.extend(neighbors)

        current_nodes = next_nodes

    return results
```

### 6.5 对话上下文管理

**上下文窗口机制**：
- 维护最近 10 轮对话历史
- 采用滑动窗口策略
- 上下文作为 system prompt 的一部分传入

```python
def _get_conversation_context(self, user_id):
    """获取当前用户的对话上下文"""
    if user_id not in self.conversation_history:
        return ""

    history = self.conversation_history[user_id][-10:]
    return "\n".join([
        f"用户: {turn['user']}\nAI: {turn['assistant']}"
        for turn in history
    ])
```

### 6.6 语音输入流程

```
麦克风/文件 → 格式转换(16kHz WAV) → OSS 上传 → Paraformer-V2 识别
                                                    │
                                                    ▼
                                              文本提取
                                                    │
                                                    ▼
                                              意图分类 → 后续处理
```

---

## 7. 评估工具链

### 7.1 测试数据集生成器 (`test_data_generator.py`)

**功能**：模拟真实场景生成测试数据

**场景类型**：
| 场景 | 人物池 | 地点池 | 要求 |
|-----|-------|-------|-----|
| 工作会议 | 产品经理、技术负责人... | 会议室、线上会议室 | 2+同事、1+地点、决策+待办 |
| 家庭聚餐 | 父亲、母亲、哥哥... | 家里、餐厅、酒店 | 2+成员、1+地点、回忆+计划 |
| 技术灵感 | 导师、同学、技术博主... | 实验室、图书馆 | 2+人物、1+地点、创新+实践 |
| 情绪碎片 | 心理咨询师、好友... | 宿舍、操场 | 2+人物、1+地点、情绪+反思 |
| 学习笔记 | 教授、助教、学霸... | 教室、图书馆 | 2+人物、1+地点、知识+提醒 |

### 7.2 判别 Agent (`evaluator.py`)

**评估维度**：

| 维度 | 权重 | 指标 |
|-----|-----|-----|
| 实体 F1 | 35% | 精确率、召回率、F1 分数 |
| 关系准确率 | 25% | 关系描述合理性 |
| 完整性 | 25% | 关键信息遗漏程度 |
| 一致性 | 15% | 信息间逻辑矛盾 |

**评估流程**：
```
提取结果 → 实体评估 → 关系评估 → 完整性评估 → 一致性评估 → 综合评分
              │           │            │             │
              ▼           ▼            ▼             ▼
           TP/FP/FN   valid/invalid  0-1 score    0-1 score
```

### 7.3 批量评估流程 (`run_evaluation.py`, `resumable_evaluation.py`)

**特性**：
- 支持批量处理大量测试用例
- 支持断点续传（checkpoint）
- 自动生成评估报告

---

## 8. 关键技术创新点

### 8.1 多智能体协作架构
- **ChatAgent**：中央调度器，负责意图路由
- **子智能体**：专业化分工，职责单一
- **信息流动**：单向流水线式处理，降低耦合

### 8.2 混合检索策略
- **向量检索**：快速召回语义相似节点
- **图检索**：多跳扩展发现关联信息
- **LLM 筛选**：智能剪枝，提高相关性

### 8.3 反幻觉机制
```
绝对禁止规则：
0. 在记事本模式(2,3)下，绝对不能编造、推测或扩展任何内容
1. 只提取用户输入中明确存在的信息
2. 禁止脑补或合理推测
3. 只提取事实，不添加解释或分析
```

### 8.4 动态用户个性化
- 实时查询数据库获取用户名
- 系统提示词动态注入用户身份
- 对话历史按用户隔离

### 8.5 智能模式切换
- **闲聊模式**：纯粹对话，无记忆功能
- **记事本模式**：启用完整的知识管理功能
- 用户可自由切换，灵活适应不同场景

---

## 9. 部署与运行

### 9.1 环境准备

**1. 安装 PostgreSQL 并启用 pgvector**
```bash
# 安装 PostgreSQL 16
apt-get install postgresql-16 postgresql-16-pgvector

# 创建数据库
createdb ai_notepad

# 启用 pgvector
psql -d ai_notepad -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**2. 配置环境变量 (.env)**
```bash
# DashScope API
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx

# 数据库
DATABASE_URL=postgresql://user:password@localhost:5432/ai_notepad

# OSS 配置
OSS_ACCESS_KEY_ID=xxxxx
OSS_ACCESS_KEY_SECRET=xxxxx
OSS_BUCKET_NAME=your-bucket
OSS_ENDPOINT=oss-cn-shenzhen.aliyuncs.com
```

### 9.2 启动服务

```bash
# 进入项目目录
cd /root/fugeg/app

# 安装依赖
pip install -r requirements.txt

# 启动应用
python app.py
```

**服务地址**：`http://0.0.0.0:7860`

### 9.3 系统初始化

首次运行需要创建用户：
1. 在 Web 界面输入用户名注册
2. 使用注册的用户名登录
3. 开始使用系统

---

## 10. 未来优化方向

### 10.1 技术层面
| 优化项 | 预期收益 |
|-------|---------|
| 引入向量数据库 (Milvus/Qdrant) | 支持更大规模向量检索 |
| 增加本地 LLM (Ollama) | 降低 API 成本，保护隐私 |
| 实现增量索引 | 提高大批量导入性能 |
| 增加缓存层 (Redis) | 减少重复 LLM 调用 |

### 10.2 功能层面
| 功能 | 描述 |
|-----|-----|
| 记忆编辑 | 支持修改和删除已存储的信息 |
| 标签系统 | 为记忆添加自定义标签分类 |
| 导出功能 | 支持导出为 JSON/Markdown |
| 提醒功能 | 基于时间的事件提醒 |
| 知识图谱可视化 | Web 界面展示个人知识网络 |

### 10.3 用户体验层面
- 移动端适配
- 深色模式支持
- 国际化 (i18n)
- 更丰富的语音交互

---

## 附录

### A. 项目目录结构
```
/root/fugeg/app/
├── app.py                      # 主入口文件
├── app_smolagents.py          # Smolagents 版本（实验性）
├── requirements.txt           # Python 依赖
├── .env                       # 环境变量配置
│
├── graphrag/
│   ├── __init__.py
│   │
│   ├── agents/                 # 智能体模块
│   │   ├── chat_agent.py       # 中央调度器
│   │   ├── memory_generator_agent.py  # 记忆生成
│   │   ├── memory_inserter_agent.py   # 记忆插入
│   │   └── memory_retriever_agent.py  # 记忆检索
│   │
│   ├── db/                     # 数据库模块
│   │   ├── database.py         # 数据库操作
│   │   └── handler.py          # 数据处理器
│   │
│   ├── models/                 # AI 模型模块
│   │   ├── llm.py             # LLM 封装
│   │   └── embedding.py       # 向量模型
│   │
│   ├── ui/                     # 界面模块
│   │   └── gradio_ui.py       # Gradio 界面
│   │
│   ├── utils/                  # 工具模块
│   │   ├── llm_client.py      # API 客户端
│   │   ├── oss_helper.py      # OSS 工具
│   │   └── logger.py          # 日志工具
│   │
│   ├── smolagents_tools/      # Smolagents 工具集
│   │   ├── __init__.py
│   │   ├── memory_tools.py
│   │   └── chat_agent.py
│   │
│   └── tests/                  # 评估测试模块
│       ├── test_data_generator.py
│       ├── evaluator.py
│       ├── run_evaluation.py
│       └── resumable_evaluation.py
│
└── logs/                       # 日志目录
    └── graphrag.log
```

### B. API 调用成本估算
| 操作 | 模型 | Token 估算 |
|-----|------|----------|
| 意图分类 | qwen-plus | ~100 input |
| 信息提取 | qwen-plus | ~500 input + 300 output |
| 对话生成 | qwen-plus | ~400 input + 200 output |
| 嵌入生成 | text-embedding-v3 | ~100 input |

### C. 性能基准
| 指标 | 数值 |
|-----|-----|
| 单次记忆提取延迟 | ~2-3 秒 |
| 单次记忆检索延迟 | ~3-5 秒 |
| 向量搜索延迟 | ~50-100ms |
| 最大并发连接 | 40 线程 |

---

*文档版本：v1.0*
*最后更新：2026-04-11*
*项目作者：Fugeg*
