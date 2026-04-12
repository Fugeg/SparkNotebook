"""
MCP GitHub 客户端
支持两种模式：
1. REST API 模式（原有实现）
2. MCP Server 模式（新增）
"""
import os
import json
import asyncio
from typing import Optional, Dict, Any, List
from urllib.parse import quote

# 导入 MCP Server 客户端
try:
    from graphrag.utils.mcp_github_server_client import MCPGitHubServerClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


class MCPGitHubClient:
    """
    GitHub MCP 客户端
    
    支持双模式：
    - REST API 模式：直接使用 GitHub REST API
    - MCP Server 模式：通过 Model Context Protocol 连接 GitHub MCP Server
    
    功能：
    1. 搜索 GitHub 仓库
    2. 获取仓库 README
    3. 获取仓库 Issues
    4. 获取用户 Starred 项目
    """
    
    def __init__(self, github_token: Optional[str] = None, use_mcp_server: bool = False):
        """
        初始化 GitHub 客户端
        
        Args:
            github_token: GitHub Personal Access Token
            use_mcp_server: 是否使用 MCP Server 模式
        """
        self.github_token = github_token or os.getenv('GITHUB_TOKEN') or os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN')
        self.use_mcp_server = use_mcp_server and MCP_AVAILABLE and os.getenv('USE_GITHUB_MCP', 'false').lower() == 'true'
        
        # MCP Server 客户端
        self.mcp_client = None
        if self.use_mcp_server:
            if MCP_AVAILABLE:
                self.mcp_client = MCPGitHubServerClient(self.github_token)
                print("[GitHub] 使用 MCP Server 模式")
            else:
                print("[GitHub] MCP SDK 未安装，回退到 REST API 模式")
                self.use_mcp_server = False
        
        # REST API 配置
        if not self.use_mcp_server:
            self.base_url = "https://api.github.com"
            self.headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Fugeg-AI-Notepad"
            }
            if self.github_token:
                self.headers["Authorization"] = f"token {self.github_token}"
            print("[GitHub] 使用 REST API 模式")
    
    async def connect(self):
        """连接到 MCP Server（仅在 MCP 模式下需要）"""
        if self.use_mcp_server and self.mcp_client:
            await self.mcp_client.connect()
    
    async def disconnect(self):
        """断开 MCP Server 连接"""
        if self.use_mcp_server and self.mcp_client:
            await self.mcp_client.disconnect()
    
    async def search_repositories(self, query: str, sort: str = "stars", 
                                   order: str = "desc", per_page: int = 5) -> List[Dict[str, Any]]:
        """
        搜索 GitHub 仓库
        
        Args:
            query: 搜索关键词
            sort: 排序方式 (stars, forks, updated)
            order: 排序顺序 (desc, asc)
            per_page: 返回数量
            
        Returns:
            仓库列表
        """
        if self.use_mcp_server and self.mcp_client:
            return await self._search_repositories_mcp(query, per_page)
        else:
            return await self._search_repositories_rest(query, sort, order, per_page)
    
    async def _search_repositories_mcp(self, query: str, per_page: int = 5) -> List[Dict[str, Any]]:
        """使用 MCP Server 搜索仓库"""
        try:
            items = await self.mcp_client.search_repositories(query, per_page)
            
            # 简化返回的数据
            results = []
            for item in items:
                results.append({
                    "name": item.get('name'),
                    "full_name": item.get('full_name'),
                    "description": item.get('description'),
                    "stars": item.get('stargazers_count'),
                    "forks": item.get('forks_count'),
                    "language": item.get('language'),
                    "url": item.get('html_url'),
                    "readme_url": f"https://raw.githubusercontent.com/{item.get('full_name')}/main/README.md"
                })
            return results
        except Exception as e:
            print(f"[GitHub MCP] 搜索错误: {e}，回退到 REST API")
            return await self._search_repositories_rest(query, per_page=per_page)
    
    async def _search_repositories_rest(self, query: str, sort: str = "stars", 
                                         order: str = "desc", per_page: int = 5) -> List[Dict[str, Any]]:
        """使用 REST API 搜索仓库"""
        import aiohttp
        
        encoded_query = quote(query)
        url = f"{self.base_url}/search/repositories?q={encoded_query}&sort={sort}&order={order}&per_page={per_page}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get('items', [])
                        
                        # 简化返回的数据
                        results = []
                        for item in items:
                            results.append({
                                "name": item.get('name'),
                                "full_name": item.get('full_name'),
                                "description": item.get('description'),
                                "stars": item.get('stargazers_count'),
                                "forks": item.get('forks_count'),
                                "language": item.get('language'),
                                "url": item.get('html_url'),
                                "readme_url": f"https://raw.githubusercontent.com/{item.get('full_name')}/main/README.md"
                            })
                        return results
                    else:
                        print(f"[GitHub] 搜索失败: {response.status}")
                        return []
        except Exception as e:
            print(f"[GitHub] 搜索错误: {e}")
            return []
    
    async def get_readme(self, owner: str, repo: str) -> Optional[str]:
        """
        获取仓库 README 内容
        
        Args:
            owner: 仓库所有者
            repo: 仓库名
            
        Returns:
            README 内容或 None
        """
        if self.use_mcp_server and self.mcp_client:
            return await self._get_readme_mcp(owner, repo)
        else:
            return await self._get_readme_rest(owner, repo)
    
    async def _get_readme_mcp(self, owner: str, repo: str) -> Optional[str]:
        """使用 MCP Server 获取 README"""
        try:
            content = await self.mcp_client.get_file_contents(owner, repo, "README.md")
            if content:
                return content[:3000] + "..." if len(content) > 3000 else content
            return None
        except Exception as e:
            print(f"[GitHub MCP] 获取 README 错误: {e}，回退到 REST API")
            return await self._get_readme_rest(owner, repo)
    
    async def _get_readme_rest(self, owner: str, repo: str) -> Optional[str]:
        """使用 REST API 获取 README"""
        import aiohttp
        
        # 尝试多个分支
        branches = ['main', 'master', 'develop']
        
        for branch in branches:
            url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content = await response.text()
                            # 限制长度
                            return content[:3000] + "..." if len(content) > 3000 else content
            except Exception:
                continue
        
        return None
    
    async def get_repo_issues(self, owner: str, repo: str, 
                              state: str = "open", per_page: int = 5) -> List[Dict[str, Any]]:
        """
        获取仓库 Issues
        
        Args:
            owner: 仓库所有者
            repo: 仓库名
            state: 状态 (open, closed, all)
            per_page: 返回数量
            
        Returns:
            Issues 列表
        """
        if self.use_mcp_server and self.mcp_client:
            return await self._get_repo_issues_mcp(owner, repo, state, per_page)
        else:
            return await self._get_repo_issues_rest(owner, repo, state, per_page)
    
    async def _get_repo_issues_mcp(self, owner: str, repo: str, 
                                    state: str = "open", per_page: int = 5) -> List[Dict[str, Any]]:
        """使用 MCP Server 获取 Issues"""
        try:
            items = await self.mcp_client.list_issues(owner, repo, state)
            
            results = []
            for item in items[:per_page]:
                # 过滤掉 PR
                if 'pull_request' not in item:
                    results.append({
                        "title": item.get('title'),
                        "body": item.get('body', '')[:500] if item.get('body') else "",
                        "state": item.get('state'),
                        "url": item.get('html_url')
                    })
            return results
        except Exception as e:
            print(f"[GitHub MCP] 获取 Issues 错误: {e}，回退到 REST API")
            return await self._get_repo_issues_rest(owner, repo, state, per_page)
    
    async def _get_repo_issues_rest(self, owner: str, repo: str, 
                                     state: str = "open", per_page: int = 5) -> List[Dict[str, Any]]:
        """使用 REST API 获取 Issues"""
        import aiohttp
        
        url = f"{self.base_url}/repos/{owner}/{repo}/issues?state={state}&per_page={per_page}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        for item in data:
                            # 过滤掉 PR
                            if 'pull_request' not in item:
                                results.append({
                                    "title": item.get('title'),
                                    "body": item.get('body', '')[:500] if item.get('body') else "",
                                    "state": item.get('state'),
                                    "url": item.get('html_url')
                                })
                        return results
                    else:
                        return []
        except Exception as e:
            print(f"[GitHub] 获取 Issues 错误: {e}")
            return []
    
    async def get_trending_repos(self, language: Optional[str] = None, 
                                  since: str = "daily") -> List[Dict[str, Any]]:
        """
        获取 Trending 仓库（使用搜索 API 模拟）
        
        Args:
            language: 编程语言筛选
            since: 时间范围 (daily, weekly, monthly)
            
        Returns:
            热门仓库列表
        """
        # 构建查询
        query_parts = ["stars:>100"]
        if language:
            query_parts.append(f"language:{language}")
        
        # 根据时间范围添加条件
        if since == "daily":
            query_parts.append("pushed:>2024-01-01")  # 简化处理
        
        query = " ".join(query_parts)
        return await self.search_repositories(query, sort="stars", per_page=5)
    
    async def get_repo_details(self, full_name: str) -> Dict[str, Any]:
        """
        获取仓库详细信息
        
        Args:
            full_name: 完整仓库名 (owner/repo)
            
        Returns:
            仓库详情
        """
        import aiohttp
        
        url = f"{self.base_url}/repos/{full_name}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "name": data.get('name'),
                            "full_name": data.get('full_name'),
                            "description": data.get('description'),
                            "stars": data.get('stargazers_count'),
                            "forks": data.get('forks_count'),
                            "language": data.get('language'),
                            "topics": data.get('topics', []),
                            "url": data.get('html_url'),
                            "created_at": data.get('created_at'),
                            "updated_at": data.get('updated_at')
                        }
                    else:
                        return {}
        except Exception as e:
            print(f"[GitHub] 获取仓库详情错误: {e}")
            return {}


# 同步包装函数（方便非异步环境调用）
def search_github_repos_sync(query: str, per_page: int = 5) -> List[Dict[str, Any]]:
    """同步方式搜索 GitHub 仓库"""
    import requests
    
    github_token = os.getenv('GITHUB_TOKEN')
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Fugeg-AI-Notepad"
    }
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    encoded_query = quote(query)
    url = f"https://api.github.com/search/repositories?q={encoded_query}&sort=stars&order=desc&per_page={per_page}"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            results = []
            for item in items:
                results.append({
                    "name": item.get('name'),
                    "full_name": item.get('full_name'),
                    "description": item.get('description'),
                    "stars": item.get('stargazers_count'),
                    "forks": item.get('forks_count'),
                    "language": item.get('language'),
                    "url": item.get('html_url'),
                    "readme_url": f"https://raw.githubusercontent.com/{item.get('full_name')}/main/README.md"
                })
            return results
        else:
            print(f"[GitHub] 搜索失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"[GitHub] 搜索错误: {e}")
        return []


if __name__ == "__main__":
    # 测试
    async def test():
        # 测试 REST API 模式
        print("=== 测试 REST API 模式 ===")
        client_rest = MCPGitHubClient(use_mcp_server=False)
        
        print("搜索 'LoRa gateway'...")
        results = await client_rest.search_repositories("LoRa gateway", per_page=3)
        for repo in results:
            print(f"  - {repo['full_name']}: {repo['description'][:60]}... ({repo['stars']} stars)")
        
        # 测试 MCP Server 模式（如果启用）
        if os.getenv('USE_GITHUB_MCP', 'false').lower() == 'true':
            print("\n=== 测试 MCP Server 模式 ===")
            client_mcp = MCPGitHubClient(use_mcp_server=True)
            await client_mcp.connect()
            
            print("搜索 'LoRa gateway'...")
            results = await client_mcp.search_repositories("LoRa gateway", per_page=3)
            for repo in results:
                print(f"  - {repo['full_name']}: {repo['description'][:60]}... ({repo['stars']} stars)")
            
            await client_mcp.disconnect()
    
    asyncio.run(test())
