# Fugeg · AI 记事本助手 - 对话上下文传递文档

## 项目概述

**项目名称**: Fugeg · AI 记事本助手  
**技术栈**: GraphRAG-based AI 个人知识管理系统  
**用途**: 毕业论文项目，用于面试展示

---

## 核心架构

### 多智能体架构 (Multi-Agent)
1. **ChatAgent** (`graphrag/agents/chat_agent.py`) - 核心编排器
2. **MemoryGeneratorAgent** - 从自然语言提取结构化信息
3. **MemoryInserterAgent** - 将记忆持久化到知识图谱
4. **MemoryRetrieverAgent** - 多跳图检索

### 技术组件
- **数据库**: PostgreSQL + pgvector (向量存储)
- **LLM**: DashScope/Qwen (通义千问)
- **语音**: Paraformer-v2 (语音识别)
- **缓存**: Redis 语义缓存 (向量相似度匹配)
- **前端**: Gradio
- **外部API**: GitHub API, StepFun API (阶跃星辰)

---

## 意图分类系统 (4种模式)

| 意图值 | 模式 | 功能描述 |
|--------|------|----------|
| 1 | 闲聊模式 | 日常对话，AI自动识别记忆查询并检索数据库 |
| 2 | 记录记忆 | 提取结构化信息存入知识图谱 |
| 3 | 搜索记忆 | 专门用于记忆检索查询 |
| 4 | GitHub灵感 | 结合GitHub开源项目+本地记忆生成技术报告 |

### 模式切换说明
- **闲聊模式**: AI会自动判断用户是否在询问记忆（使用LLM进行专业意图识别）
- **记事本模式**: 所有输入自动保存为结构化记忆，不需要意图识别
- **GitHub灵感模式**: 强制使用意图4，查询GitHub项目并生成技术报告

---

## 最近修复的关键问题

### 1. 记忆查询检测优化
**文件**: `graphrag/agents/chat_agent.py`  
**方法**: `_is_asking_for_memories()`  
**说明**: 使用LLM进行专业意图识别，替代之前的关键词匹配

```python
def _is_asking_for_memories(self, user_input: str) -> bool:
    system_prompt = """You are an intent classifier...
    Respond with ONLY "YES" or "NO":..."""
    result = self.llm.chat(...)
    return "YES" in result.upper()
```

### 2. user_id 传递修复
**文件**: `graphrag/agents/chat_agent.py`  
**问题**: `handle_chat()` 调用时缺少 `user_id` 参数  
**修复**: 第131行已添加 `user_id=user_id`

```python
response = self.handle_chat(user_input, context, username, user_id=user_id)
```

### 3. 模式描述优化
**文件**: `graphrag/ui/gradio_ui.py`  
**说明**: 更新了模式切换时的提示文字，让用户更清楚每个模式的作用

---

## 文件结构

```
/root/fugeg/app/
├── main.py                          # 项目入口
├── INTERVIEW_DOCUMENTATION.md       # 面试文档
├── CONVERSATION_CONTEXT.md          # 本文件
│
├── graphrag/
│   ├── agents/
│   │   ├── chat_agent.py           # 核心编排器 (最近修改)
│   │   ├── memory_generator_agent.py
│   │   ├── memory_inserter_agent.py
│   │   └── memory_retriever_agent.py
│   │
│   ├── models/
│   │   ├── llm.py                  # LLM接口 (意图分类、信息提取)
│   │   └── embedding.py            # 向量嵌入
│   │
│   ├── db/
│   │   └── database.py             # PostgreSQL操作
│   │
│   ├── ui/
│   │   └── gradio_ui.py            # 前端界面 (最近修改)
│   │
│   └── utils/
│       ├── cache_helper.py         # Redis语义缓存
│       ├── mcp_github_client.py    # GitHub API客户端
│       └── stepfun_client.py       # StepFun API客户端
│
├── evaluation/                      # 评估工具链
│   ├── test_data_generator.py
│   ├── evaluator_agent.py
│   └── run_evaluation.py
│
└── data/                           # 数据目录
```

---

## 关键配置

### Redis 缓存配置
```python
self.cache = SemanticCache(
    host='localhost',
    port=6379,
    db=0,
    expire_time=3600*24,  # 24小时过期
    similarity_threshold=0.95  # 相似度阈值
)
```

### StepFun API 配置
- **模型**: `step-3.5-flash` (注意不是 `step-3.5`)
- **用途**: 
  - 中文查询翻译为英文关键词 (用于GitHub搜索)
  - 生成技术灵感报告

---

## 当前状态

### 已完成功能
1. ✅ 多智能体架构
2. ✅ 意图分类 (4种意图)
3. ✅ 语音输入支持
4. ✅ Redis语义缓存
5. ✅ GitHub灵感模式
6. ✅ 模式切换 (闲聊/记事本/GitHub)
7. ✅ 用户管理和权限控制
8. ✅ 评估工具链

### 最近修复
1. ✅ `handle_chat()` 调用添加 `user_id` 参数
2. ✅ 模式描述文字优化
3. ✅ LLM-based 记忆查询检测

### 待测试
- 闲聊模式下询问记忆是否能正确查询数据库
- 模式切换是否正常工作

---

## 启动命令

```bash
cd /root/fugeg/app
python main.py
```

---

## 常见问题

### Q: 闲聊模式和记事本模式的区别？
**A**: 
- 闲聊模式：AI会自动识别用户是否在询问记忆，如果是则查询数据库
- 记事本模式：所有输入自动保存为记忆，不进行意图识别

### Q: 为什么之前"该查询数据库记住的东西没记住"？
**A**: 因为 `handle_chat()` 调用时缺少 `user_id` 参数，导致记忆检索时无法正确过滤用户数据。已在第131行修复。

### Q: GitHub灵感模式如何工作？
**A**: 
1. 使用StepFun将中文查询翻译为英文关键词
2. 搜索GitHub相关项目 (取前3个)
3. 获取第一个项目的README
4. 检索本地相关记忆
5. 使用StepFun生成技术灵感报告

---

## 下一步建议

1. **测试修复效果**: 启动项目后，在闲聊模式下询问"你还记得我的灵感有哪些吗"，验证是否能正确返回数据库中的记忆
2. **验证模式切换**: 测试从闲聊模式切换到记事本模式，再切换回闲聊模式是否正常
3. **运行评估工具链**: 使用 `evaluation/run_evaluation.py` 测试系统性能

---

## 重要代码片段

### 记忆查询检测 (chat_agent.py L157-192)
```python
def _is_asking_for_memories(self, user_input: str) -> bool:
    """使用 LLM 判断用户是否在询问记忆/灵感"""
    system_prompt = """You are an intent classifier...
    
Examples:
Input: "你还记得我的灵感有哪些吗" -> YES
Input: "我之前说过什么" -> YES
Input: "你好" -> NO
Input: "今天天气怎么样" -> NO"""
    
    result = self.llm.chat(...)
    return "YES" in result.upper()
```

### 闲聊处理 (chat_agent.py L194-247)
```python
def handle_chat(self, user_input, context="", username="用户", use_cache=True, user_id=1):
    """处理普通闲聊（支持记忆检索）"""
    
    # 检测是否在询问记忆
    if not context and self._is_asking_for_memories(user_input):
        retrieval_result = self.memory_retriever.retrieve_memory(user_input, user_id=user_id)
        if retrieval_result:
            # 基于记忆生成回答
            ...
    
    # 尝试从缓存获取
    if use_cache and not context and query_vector:
        cached_response = self.cache.get(user_input, query_vector)
        if cached_response:
            return f"[来自缓存] {cached_response}"
    
    # 调用LLM生成回复
    ...
```

---

**生成时间**: 2026-04-11  
**项目状态**: 开发中，核心功能已完成，正在修复边界问题
