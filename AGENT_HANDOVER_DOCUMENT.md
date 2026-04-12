# Fugeg · AI 记事本助手 - Agent 交接文档

**生成时间**: 2026-04-12  
**交接人**: AI Agent  
**项目状态**: 运行中（后台模式）

---

## 📋 项目概述

**项目名称**: Fugeg · AI 记事本助手  
**技术栈**: GraphRAG-based AI 个人知识管理系统  
**用途**: 毕业论文项目，用于面试展示  
**当前状态**: ✅ 核心功能完成，MCP GitHub Server 集成完成

---

## 🎯 本次完成的主要工作

### 1. MCP GitHub Server 集成

#### 新增文件
- `graphrag/utils/mcp_github_server_client.py` - MCP Server 底层客户端
- `graphrag/utils/mcp_github_client.py` - 双模式 GitHub 客户端（REST API + MCP Server）
- `test_mcp_github.py` - MCP 客户端测试脚本
- `test_github_inspiration.py` - GitHub 灵感模式测试脚本

#### 关键功能
- **双模式支持**: REST API 模式 + MCP Server 模式
- **自动切换**: MCP Server 失败时自动回退到 REST API
- **LLM 相似度评估**: 搜索后使用 LLM 评估相似度，只返回 >= 90% 的结果
- **环境控制**: 通过 `USE_GITHUB_MCP` 环境变量控制模式

#### 配置 (.env)
```bash
# GitHub 配置
GITHUB_TOKEN=your_github_token_here
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here

# GitHub MCP Server 配置
USE_GITHUB_MCP=true
GITHUB_MCP_SERVER_PATH=github-mcp-custom
```

### 2. 翻译功能修复

#### 问题
- StepFun API (`step-3.5-flash`) 返回空字符串
- 中文查询无法正确翻译为英文
- GitHub 搜索返回 0 个结果

#### 解决方案
- **Qwen-plus 备用翻译**: 当 StepFun 失败时，自动使用 Qwen-plus 进行翻译
- **翻译流程**:
  ```
  用户中文查询
      ↓
  StepFun LLM 翻译
      ↓
  如果返回空/无效
      ↓
  自动使用 Qwen-plus 备用翻译
      ↓
  返回英文关键词
  ```

#### 修改文件
- `graphrag/utils/stepfun_client.py` - 添加 `_translate_with_qwen()` 方法

### 3. 项目部署

#### 启动方式
```bash
# 后台启动（断开 SSH 后继续运行）
cd /root/fugeg/app && nohup python3 app.py > app.log 2>&1 &
```

#### 当前状态
- **进程 ID**: 1597528
- **日志文件**: `/root/fugeg/app/app.log`
- **访问地址**: http://0.0.0.0:7860

#### 常用命令
```bash
# 查看日志（实时）
tail -f /root/fugeg/app/app.log

# 查看进程
ps aux | grep "python3 app.py"

# 停止项目
kill 1597528

# 重启项目
cd /root/fugeg/app && nohup python3 app.py > app.log 2>&1 &
```

---

## 📁 文件结构更新

```
/root/fugeg/app/
├── main.py                          # 项目入口（已弃用）
├── app.py                           # 当前入口
├── app_smolagents.py                # Smolagents 版本入口
├── app.log                          # 后台运行日志（新增）
│
├── graphrag/
│   ├── agents/
│   │   ├── chat_agent.py           # 核心编排器（已更新）
│   │   │                          # - 集成 MCP GitHub 客户端
│   │   │                          # - GitHub 灵感模式支持双模式
│   │   ├── memory_generator_agent.py
│   │   ├── memory_inserter_agent.py
│   │   └── memory_retriever_agent.py
│   │
│   ├── models/
│   │   ├── llm.py                  # LLM接口（Qwen-plus）
│   │   └── embedding.py            # 向量嵌入
│   │
│   ├── db/
│   │   └── database.py             # PostgreSQL操作
│   │
│   ├── ui/
│   │   └── gradio_ui.py            # 前端界面
│   │
│   └── utils/
│       ├── cache_helper.py         # Redis语义缓存
│       ├── mcp_github_client.py    # 双模式GitHub客户端（新增/更新）
│       ├── mcp_github_server_client.py  # MCP Server底层客户端（新增）
│       └── stepfun_client.py       # StepFun客户端（已更新）
│                                   # - 添加Qwen-plus备用翻译
│
├── test_mcp_github.py              # MCP客户端测试脚本（新增）
├── test_github_inspiration.py      # GitHub灵感模式测试（新增）
├── test_stepfun_timing.py          # StepFun响应时间测试（新增）
│
├── GITHUB_MCP_SERVER_MIGRATION.md  # MCP迁移文档（新增）
├── MCP_MIGRATION_TODO.md           # 任务清单（新增）
└── AGENT_HANDOVER_DOCUMENT.md      # 本文件（新增）
```

---

## 🔧 核心架构

### 系统架构
```
┌─────────────────┐
│   Gradio UI     │
└────────┬────────┘
         │
┌────────▼────────┐
│   ChatAgent     │
│  (核心编排器)    │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐  ┌──▼────────┐
│StepFun│  │  MCP      │
│ LLM   │  │  GitHub   │
│Client │  │  Server   │
│       │  │  Client   │
└───────┘  └─────┬─────┘
    │            │
    ▼            ▼
┌───────┐  ┌──────────┐
│step-  │  │  GitHub  │
│3.5-   │  │   API    │
│flash  │  └──────────┘
└───────┘
```

### 意图分类系统

| 意图值 | 模式 | 功能描述 |
|--------|------|----------|
| 1 | 闲聊模式 | 日常对话，AI自动识别记忆查询并检索数据库 |
| 2 | 记录记忆 | 提取结构化信息存入知识图谱 |
| 3 | 搜索记忆 | 专门用于记忆检索查询 |
| 4 | GitHub灵感 | 结合GitHub开源项目+本地记忆生成技术报告 |

### GitHub 灵感模式流程
```
用户输入（中文）
    ↓
StepFun 翻译 → 如果失败 → Qwen-plus 备用翻译
    ↓
英文关键词
    ↓
MCP Server 搜索仓库（3倍数量）
    ↓
LLM 相似度评估（逐个评估）
    ↓
筛选 >= 90% 相似度的仓库
    ↓
获取 README
    ↓
检索本地记忆
    ↓
StepFun 生成技术灵感报告
```

---

## ⚙️ 关键配置

### 环境变量 (.env)
```bash
# 数据库配置
DATABASE_URL=postgresql://root:123456@localhost:5432/ai_notepad

# DashScope API配置（Qwen-plus）
DASHSCOPE_API_KEY=your_dashscope_key

# 阶跃星辰配置（StepFun）
STEPFUN_API_KEY=your_stepfun_api_key_here
STEPFUN_BASE_URL=https://api.stepfun.com/step_plan/v1
STEPFUN_MODEL=step-3.5-flash

# GitHub 配置
GITHUB_TOKEN=your_github_token_here
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token_here

# GitHub MCP Server 配置
USE_GITHUB_MCP=true
GITHUB_MCP_SERVER_PATH=github-mcp-custom

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
```

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

---

## 🧪 测试脚本

### 1. 测试 MCP GitHub 客户端
```bash
cd /root/fugeg/app && python3 test_mcp_github.py
```

### 2. 测试 GitHub 灵感模式
```bash
cd /root/fugeg/app && python3 test_github_inspiration.py
```

### 3. 测试 StepFun 响应时间
```bash
cd /root/fugeg/app && python3 test_stepfun_timing.py
```

---

## 📊 性能数据

### StepFun API 响应时间（参考）

| 测试项目 | 平均响应时间 | 说明 |
|---------|-------------|------|
| 简单对话 | 2.04 秒 | 约 2 秒左右 |
| 查询翻译 | 0.96 秒 | 但返回空结果 |
| 灵感报告生成 | 11.47 秒 | 长文本生成 |

### GitHub MCP Server 空间占用

| 项目 | 大小 |
|------|------|
| 包本身 | 75.4 kB |
| 总依赖 | 约 5-10 MB |
| 运行时内存 | 50-100 MB |

---

## ⚠️ 已知问题与解决方案

### 问题 1: StepFun API 返回空字符串
**现象**: `step-3.5-flash` 模型在翻译任务中返回空结果  
**解决方案**: ✅ 已添加 Qwen-plus 备用翻译  
**状态**: 已修复

### 问题 2: GitHub 搜索返回 0 个结果
**现象**: 中文查询无法匹配到英文项目  
**解决方案**: ✅ 使用 Qwen-plus 将中文翻译为英文关键词  
**状态**: 已修复

### 问题 3: Gradio 版本警告
**现象**: `UserWarning: The parameters have been moved from the Blocks constructor to the launch() method`  
**影响**: 不影响功能，仅为警告  
**解决方案**: 可选修复，将 `theme` 和 `css` 参数从 `gr.Blocks()` 移到 `launch()`

---

## 📝 下一步建议

### 高优先级
1. **测试 LLM 相似度评估** - 验证新功能是否正常工作
2. **性能优化** - 如果 LLM 评估太慢，考虑批量评估或缓存
3. **添加更多翻译字典** - 扩展 Qwen-plus 备用翻译的词汇量

### 中优先级
4. **更新文档** - 更新 README.md 和 INTERVIEW_DOCUMENTATION.md
5. **添加更多 MCP 功能** - 如代码搜索、创建 Issue 等
6. **性能对比测试** - REST API vs MCP Server 响应时间对比

### 低优先级
7. **修复 Gradio 警告** - 将 theme/css 参数移到 launch()
8. **添加单元测试** - 为 MCP 客户端添加测试用例
9. **优化翻译质量** - 改进查询翻译的准确性

---

## 🔗 参考链接

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [GitHub MCP Server](https://github.com/anthropics/anthropic-cookbook/tree/main/mcp-servers/github)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

---

## 📞 联系信息

如有问题，请参考以下文档：
- `GITHUB_MCP_SERVER_MIGRATION.md` - MCP 迁移完整文档
- `MCP_MIGRATION_TODO.md` - 任务清单
- `CONVERSATION_CONTEXT.md` - 项目上下文文档
- `INTERVIEW_DOCUMENTATION.md` - 面试文档

---

**文档版本**: 1.0  
**最后更新**: 2026-04-12  
**状态**: ✅ 项目运行中，所有核心功能已完成
