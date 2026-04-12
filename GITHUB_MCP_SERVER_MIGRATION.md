# GitHub MCP Server 迁移文档

## 概述

本文档描述了从当前 GitHub API 客户端迁移到 **GitHub MCP Server** 的完整计划和实施步骤。

**架构说明**：
- **GitHub 操作**：使用 Anthropic GitHub MCP Server（通过 MCP 协议）
- **LLM 模型**：继续使用 阶跃星辰 step-3.5-flash（通过 StepFun API）
- **不需要 Anthropic API Key**

---

## 什么是 MCP?

**MCP (Model Context Protocol)** 是 Anthropic 推出的开放协议，用于标准化 AI 模型与外部工具/数据源的集成。GitHub MCP Server 提供了更丰富的 GitHub 操作能力。

### MCP 的优势

1. **标准化接口** - 统一的工具调用方式
2. **更丰富功能** - 支持代码搜索、文件内容获取、PR 操作等
3. **更好的类型安全** - Schema 定义的工具接口
4. **实时数据** - 直接访问 GitHub 最新数据

---

## 当前实现 vs MCP Server

| 功能 | 当前实现 | MCP Server |
|------|---------|-----------|
| 搜索仓库 | ✅ REST API | ✅ `search_repositories` |
| 获取 README | ✅ Raw 内容 | ✅ `get_file_contents` |
| 获取 Issues | ✅ REST API | ✅ `list_issues` |
| 获取代码文件 | ❌ 不支持 | ✅ `get_file_contents` |
| 代码搜索 | ❌ 不支持 | ✅ `search_code` |
| 创建 Issue | ❌ 不支持 | ✅ `create_issue` |
| 创建 PR | ❌ 不支持 | ✅ `create_pull_request` |
| 分支操作 | ❌ 不支持 | ✅ `create_branch` |

---

## 系统架构

```
┌─────────────────┐
│   Gradio UI     │
└────────┬────────┘
         │
┌────────▼────────┐
│   ChatAgent     │
│  (编排器)        │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐  ┌──▼────┐
│StepFun│  │  MCP  │
│ LLM   │  │ GitHub│
│Client │  │Server │
└───────┘  └───────┘
    │          │
    ▼          ▼
┌───────┐  ┌──────────┐
│step-  │  │  GitHub  │
│3.5-   │  │   API    │
│flash  │  └──────────┘
└───────┘
```

---

## 迁移步骤

### 阶段 1: 环境准备

#### 1.1 安装 MCP SDK

```bash
# 安装 MCP SDK（只需要这个，不需要 anthropic）
pip install mcp --break-system-packages
```

#### 1.2 配置环境变量

在 `.env` 文件中添加：

```bash
# GitHub MCP Server 配置
USE_GITHUB_MCP=true
GITHUB_MCP_SERVER_PATH=@anthropics/github-mcp-server
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token
```

#### 1.3 安装 GitHub MCP Server

**方式 A: 使用 npx (推荐)**
```bash
npx -y @anthropics/github-mcp-server
```

**方式 B: 本地安装**
```bash
git clone https://github.com/anthropics/anthropic-cookbook.git
cd anthropic-cookbook/mcp-servers/github
npm install
npm run build
```

---

### 阶段 2: 代码实现

#### 2.1 创建 MCP GitHub 客户端

创建新文件 `graphrag/utils/mcp_github_server_client.py`:

```python
"""
GitHub MCP Server 客户端
基于 Model Context Protocol 的 GitHub 集成

注意：此客户端只负责 GitHub 操作，LLM 仍使用 StepFun
"""
import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPGitHubServerClient:
    """
    GitHub MCP Server 客户端
    
    功能：
    1. 搜索 GitHub 仓库
    2. 获取文件内容（包括 README）
    3. 搜索代码
    4. 获取 Issues 和 PRs
    5. 创建 Issue 和 PR
    """
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        self.server_path = os.getenv('GITHUB_MCP_SERVER_PATH', '@anthropics/github-mcp-server')
        self.session = None
        self.exit_stack = None
        
    async def connect(self):
        """连接到 MCP Server"""
        if not self.github_token:
            raise ValueError("GitHub Token 未设置")
            
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", self.server_path],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": self.github_token}
        )
        
        self.exit_stack = await stdio_client(server_params).__aenter__()
        stdio, write = self.exit_stack
        self.session = await ClientSession(stdio, write).__aenter__()
        await self.session.initialize()
        
    async def disconnect(self):
        """断开连接"""
        if self.session:
            await self.session.__aexit__(None, None, None)
        if self.exit_stack:
            await self.exit_stack.__aexit__(None, None, None)
    
    async def search_repositories(self, query: str, per_page: int = 5) -> List[Dict[str, Any]]:
        """搜索 GitHub 仓库"""
        if not self.session:
            await self.connect()
            
        result = await self.session.call_tool(
            "search_repositories",
            {"query": query, "perPage": per_page}
        )
        return json.loads(result.content[0].text)
    
    async def get_file_contents(self, owner: str, repo: str, 
                                path: str = "README.md",
                                branch: str = "main") -> Optional[str]:
        """获取文件内容"""
        if not self.session:
            await self.connect()
            
        try:
            result = await self.session.call_tool(
                "get_file_contents",
                {
                    "owner": owner,
                    "repo": repo,
                    "path": path,
                    "branch": branch
                }
            )
            return result.content[0].text
        except Exception as e:
            # 尝试 master 分支
            if branch == "main":
                return await self.get_file_contents(owner, repo, path, "master")
            return None
    
    async def search_code(self, query: str, per_page: int = 5) -> List[Dict[str, Any]]:
        """搜索代码"""
        if not self.session:
            await self.connect()
            
        result = await self.session.call_tool(
            "search_code",
            {"query": query, "perPage": per_page}
        )
        return json.loads(result.content[0].text)
    
    async def list_issues(self, owner: str, repo: str, 
                         state: str = "open") -> List[Dict[str, Any]]:
        """获取 Issues"""
        if not self.session:
            await self.connect()
            
        result = await self.session.call_tool(
            "list_issues",
            {
                "owner": owner,
                "repo": repo,
                "state": state
            }
        )
        return json.loads(result.content[0].text)
    
    async def create_issue(self, owner: str, repo: str, 
                          title: str, body: str) -> Dict[str, Any]:
        """创建 Issue"""
        if not self.session:
            await self.connect()
            
        result = await self.session.call_tool(
            "create_issue",
            {
                "owner": owner,
                "repo": repo,
                "title": title,
                "body": body
            }
        )
        return json.loads(result.content[0].text)
```

#### 2.2 更新现有 GitHub 客户端

修改 `graphrag/utils/mcp_github_client.py`，添加对 MCP Server 的支持：

```python
# 添加导入
from graphrag.utils.mcp_github_server_client import MCPGitHubServerClient

class MCPGitHubClient:
    def __init__(self, github_token: Optional[str] = None, use_mcp_server: bool = False):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.use_mcp_server = use_mcp_server
        
        # 原有 REST API 客户端
        self.base_url = "https://api.github.com"
        self.headers = {...}
        
        # MCP Server 客户端
        self.mcp_client = None
        if use_mcp_server:
            self.mcp_client = MCPGitHubServerClient(self.github_token)
    
    async def search_repositories(self, query: str, per_page: int = 5):
        """搜索仓库 - 支持双模式"""
        if self.use_mcp_server and self.mcp_client:
            return await self.mcp_client.search_repositories(query, per_page)
        # 原有 REST API 实现...
```

---

### 阶段 3: 集成到系统

#### 3.1 更新 ChatAgent

在 `graphrag/agents/chat_agent.py` 中：

```python
class ChatAgent:
    def __init__(self, db, logger, use_github_mcp: bool = False):
        self.db = db
        self.logger = logger
        self.llm = LLMClient()  # StepFun/Qwen
        self.github_client = MCPGitHubClient(use_mcp_server=use_github_mcp)
```

#### 3.2 更新配置

在 `.env` 中添加：

```bash
# GitHub MCP Server 配置
USE_GITHUB_MCP=true
GITHUB_MCP_SERVER_PATH=@anthropics/github-mcp-server
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token
```

---

### 阶段 4: 测试

#### 4.1 创建测试脚本

```python
# test_mcp_github.py
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from graphrag.utils.mcp_github_server_client import MCPGitHubServerClient

async def test():
    """测试 MCP GitHub 客户端"""
    client = MCPGitHubServerClient()
    
    try:
        print("连接到 MCP Server...")
        await client.connect()
        print("✅ 连接成功\n")
        
        # 测试搜索仓库
        print("测试 1: 搜索仓库 'LoRa gateway'")
        repos = await client.search_repositories("LoRa gateway", 3)
        print(f"✅ 找到 {len(repos)} 个仓库")
        for repo in repos:
            print(f"  - {repo.get('full_name')}: {repo.get('stars', 0)} stars")
        
        # 测试获取 README
        if repos:
            owner, repo = repos[0]['full_name'].split('/')
            print(f"\n测试 2: 获取 {owner}/{repo} 的 README")
            readme = await client.get_file_contents(owner, repo)
            if readme:
                print(f"✅ README 长度: {len(readme)} 字符")
                print(f"   预览: {readme[:150]}...")
            else:
                print("❌ 未找到 README")
        
        # 测试代码搜索
        print("\n测试 3: 搜索代码 'def hello_world'")
        code_results = await client.search_code("def hello_world", 3)
        print(f"✅ 找到 {len(code_results)} 个代码片段")
        
        print("\n✅ 所有测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.disconnect()
        print("\n已断开连接")

if __name__ == "__main__":
    asyncio.run(test())
```

---

## 任务清单

### 🔴 高优先级

- [ ] 1. 安装 MCP SDK
  ```bash
  pip install mcp --break-system-packages
  ```
- [ ] 2. 检查 GitHub Token 权限（确认有 `repo` 权限）
- [ ] 3. 检查 Node.js 版本（需要 v18+）
  ```bash
  node --version
  ```
- [ ] 4. 测试 GitHub MCP Server 安装
  ```bash
  npx -y @anthropics/github-mcp-server --help
  ```
- [ ] 5. 创建 `mcp_github_server_client.py`

### 🟡 中优先级

- [ ] 6. 更新 `mcp_github_client.py` 支持双模式
- [ ] 7. 更新 `chat_agent.py` 集成 MCP 客户端
- [ ] 8. 更新 `.env` 配置
- [ ] 9. 创建并运行测试脚本

### 🟢 低优先级

- [ ] 10. 添加 MCP 专属功能（代码搜索、创建 Issue 等）
- [ ] 11. 更新文档
- [ ] 12. 性能对比测试
- [ ] 13. 移除旧实现（可选）

---

## 依赖安装

```bash
# 只需要 MCP SDK
pip install mcp --break-system-packages

# 检查 Node.js（MCP Server 需要）
node --version  # 需要 v18+

# 安装 GitHub MCP Server
npx -y @anthropics/github-mcp-server
```

---

## 配置示例

### 完整 .env 配置

```bash
# 数据库配置
DATABASE_URL=postgresql://root:123456@localhost:5432/ai_notepad

# DashScope API配置（用于一般对话）
DASHSCOPE_API_KEY=your_dashscope_key

# 阶跃星辰配置（用于 GitHub 灵感模式）
STEPFUN_API_KEY=your_stepfun_key
STEPFUN_BASE_URL=https://api.stepfun.com/step_plan/v1
STEPFUN_MODEL=step-3.5-flash

# GitHub 配置
GITHUB_TOKEN=your_github_token

# GitHub MCP Server 配置（新增）
USE_GITHUB_MCP=true
GITHUB_MCP_SERVER_PATH=@anthropics/github-mcp-server
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token
```

---

## 注意事项

1. **Node.js 要求**: MCP Server 需要 Node.js v18+
2. **Token 权限**: GitHub Token 需要有 `repo` 权限
3. **异步代码**: MCP 客户端使用 async/await，需要相应调整调用代码
4. **连接管理**: 注意正确管理 MCP 连接的生命周期
5. **LLM 不变**: 继续使用 StepFun step-3.5-flash，不需要 Anthropic API Key

---

## 参考链接

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [GitHub MCP Server](https://github.com/anthropics/anthropic-cookbook/tree/main/mcp-servers/github)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

---

**创建时间**: 2026-04-12  
**状态**: 待实施  
**架构**: StepFun LLM + MCP GitHub
