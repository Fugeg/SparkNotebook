"""
GitHub MCP Server 客户端
基于 Model Context Protocol 的 GitHub 集成

注意：此客户端只负责 GitHub 操作，LLM 仍使用 StepFun
"""
import os
import json
import asyncio
import re
from typing import Optional, Dict, Any, List
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPGitHubServerClient:
    """
    GitHub MCP Server 客户端
    
    功能：
    1. 搜索 GitHub 仓库（带 LLM 相似度评估）
    2. 获取文件内容（包括 README）
    3. 搜索代码
    4. 获取 Issues 和 PRs
    5. 创建 Issue 和 PR
    """
    
    def __init__(self, github_token: Optional[str] = None, use_llm_evaluation: bool = True, similarity_threshold: float = 0.9):
        self.github_token = github_token or os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN') or os.getenv('GITHUB_TOKEN')
        self.server_path = os.getenv('GITHUB_MCP_SERVER_PATH', 'github-mcp-custom')
        self.session = None
        self.exit_stack = None
        self.use_llm_evaluation = use_llm_evaluation
        self.similarity_threshold = similarity_threshold  # 默认 90%
        
    async def connect(self):
        """连接到 MCP Server"""
        if not self.github_token:
            raise ValueError("GitHub Token 未设置")
        
        # 创建退出栈
        self.exit_stack = AsyncExitStack()
        
        # 设置服务器参数
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", self.server_path, "stdio"],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": self.github_token}
        )
        
        # 连接到服务器
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        stdio, write = stdio_transport
        
        # 创建会话
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        
        # 初始化
        await self.session.initialize()
        
    async def disconnect(self):
        """断开连接"""
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.exit_stack = None
            self.session = None
    
    async def search_repositories(self, query: str, per_page: int = 5) -> List[Dict[str, Any]]:
        """
        搜索 GitHub 仓库（带 LLM 相似度评估）
        
        流程：
        1. 使用 MCP Server 搜索仓库
        2. 使用 LLM 评估每个仓库与用户查询的相似度
        3. 只返回相似度 >= 90% 的仓库
        4. 如果都不满足，继续搜索更多结果
        """
        if not self.session:
            raise RuntimeError("未连接到 MCP Server，请先调用 connect()")
        
        print(f"[GitHub MCP] 开始搜索: '{query}'")
        
        # 第一步：搜索仓库（获取更多结果用于筛选）
        search_per_page = per_page * 3  # 搜索3倍数量，用于筛选
        result = await self.session.call_tool(
            "search_repositories",
            {"query": query, "perPage": search_per_page}
        )
        data = json.loads(result.content[0].text)
        items = data.get("items", []) if isinstance(data, dict) else data
        
        print(f"[GitHub MCP] 原始搜索结果: {len(items)} 个仓库")
        
        if not items:
            return []
        
        # 第二步：使用 LLM 评估相似度
        if self.use_llm_evaluation:
            filtered_items = await self._evaluate_similarity_with_llm(query, items, per_page)
            print(f"[GitHub MCP] LLM 筛选后: {len(filtered_items)} 个仓库 (阈值: {self.similarity_threshold*100:.0f}%)")
            return filtered_items
        else:
            # 不使用 LLM 评估，直接返回前 per_page 个
            return items[:per_page]
    
    async def _evaluate_similarity_with_llm(self, query: str, repos: List[Dict[str, Any]], target_count: int) -> List[Dict[str, Any]]:
        """
        使用 LLM 评估仓库与用户查询的相似度
        
        Args:
            query: 用户查询
            repos: 仓库列表
            target_count: 目标返回数量
            
        Returns:
            相似度 >= 阈值的仓库列表
        """
        from graphrag.models.llm import LLMModel
        
        llm = LLMModel()
        qualified_repos = []
        
        print(f"[GitHub MCP] 开始 LLM 相似度评估 (阈值: {self.similarity_threshold*100:.0f}%)")
        
        for i, repo in enumerate(repos):
            if len(qualified_repos) >= target_count:
                break
                
            # 构建仓库信息
            repo_info = {
                'full_name': repo.get('full_name', 'N/A'),
                'description': repo.get('description', '无描述')[:200],
                'language': repo.get('language', '未知'),
                'stars': repo.get('stargazers_count', 0),
                'topics': repo.get('topics', [])[:5]
            }
            
            # 使用 LLM 评估相似度
            similarity_score = await self._get_llm_similarity_score(llm, query, repo_info)
            
            print(f"  [{i+1}/{len(repos)}] {repo_info['full_name']}: {similarity_score*100:.1f}%")
            
            # 如果相似度 >= 阈值，保留该仓库
            if similarity_score >= self.similarity_threshold:
                repo['_similarity_score'] = similarity_score  # 保存相似度分数
                qualified_repos.append(repo)
                print(f"    ✓ 符合条件 (已选 {len(qualified_repos)}/{target_count})")
            else:
                print(f"    ✗ 不符合条件，继续评估下一个")
        
        # 按相似度排序
        qualified_repos.sort(key=lambda x: x.get('_similarity_score', 0), reverse=True)
        
        return qualified_repos
    
    async def _get_llm_similarity_score(self, llm: 'LLMModel', query: str, repo_info: Dict[str, Any]) -> float:
        """
        使用 LLM 评估单个仓库与用户查询的相似度
        
        Returns:
            相似度分数 (0.0 - 1.0)
        """
        system_prompt = """You are a technical relevance evaluator. Your task is to evaluate how relevant a GitHub repository is to the user's query.

Evaluation Criteria:
1. Semantic relevance - Does the repository's topic match the user's intent?
2. Technical relevance - Is the technology stack appropriate?
3. Use case relevance - Does the project solve the user's problem?

Scoring Guidelines:
- 0.95-1.00: Perfect match, exactly what the user is looking for
- 0.90-0.94: Highly relevant, very good match
- 0.80-0.89: Relevant, but not perfect
- 0.70-0.79: Somewhat relevant
- 0.60-0.69: Weakly relevant
- 0.00-0.59: Not relevant

You must respond with ONLY a number between 0.00 and 1.00, representing the similarity score.
Example: 0.92"""

        user_prompt = f"""User Query: {query}

GitHub Repository:
- Name: {repo_info['full_name']}
- Description: {repo_info['description']}
- Language: {repo_info['language']}
- Stars: {repo_info['stars']}
- Topics: {', '.join(repo_info['topics']) if repo_info['topics'] else 'None'}

Evaluate the relevance (respond with only a number 0.00-1.00):"""

        try:
            # 使用 LLM 进行评估
            response = llm.chat(
                user_prompt,
                username="系统"
            )
            
            # 提取数字
            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                score = float(match.group(1))
                # 归一化到 0-1 范围
                if score > 1.0:
                    score = score / 100.0
                return min(max(score, 0.0), 1.0)
            else:
                print(f"    [警告] 无法解析 LLM 响应: '{response}'，使用默认分数 0.5")
                return 0.5
        except Exception as e:
            print(f"    [错误] LLM 评估失败: {e}，使用默认分数 0.5")
            return 0.5
    
    async def get_file_contents(self, owner: str, repo: str, 
                                path: str = "README.md",
                                branch: str = "main") -> Optional[str]:
        """获取文件内容"""
        if not self.session:
            raise RuntimeError("未连接到 MCP Server，请先调用 connect()")
            
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
            raise RuntimeError("未连接到 MCP Server，请先调用 connect()")
            
        result = await self.session.call_tool(
            "search_code",
            {"query": query, "perPage": per_page}
        )
        data = json.loads(result.content[0].text)
        return data.get("items", []) if isinstance(data, dict) else data
    
    async def list_issues(self, owner: str, repo: str, 
                         state: str = "open") -> List[Dict[str, Any]]:
        """获取 Issues"""
        if not self.session:
            raise RuntimeError("未连接到 MCP Server，请先调用 connect()")
            
        result = await self.session.call_tool(
            "list_issues",
            {
                "owner": owner,
                "repo": repo,
                "state": state
            }
        )
        data = json.loads(result.content[0].text)
        return data if isinstance(data, list) else []
    
    async def create_issue(self, owner: str, repo: str, 
                          title: str, body: str) -> Dict[str, Any]:
        """创建 Issue"""
        if not self.session:
            raise RuntimeError("未连接到 MCP Server，请先调用 connect()")
            
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
