"""
测试 MCP GitHub Server 客户端
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag.utils.mcp_github_server_client import MCPGitHubServerClient


async def test():
    """测试 MCP GitHub 客户端"""
    print("="*60)
    print("MCP GitHub Server 客户端测试")
    print("="*60)
    
    client = MCPGitHubServerClient()
    
    # 检查 Token
    if not client.github_token:
        print("\n❌ 错误: GitHub Token 未设置")
        print("请检查 .env 文件中的 GITHUB_TOKEN 或 GITHUB_PERSONAL_ACCESS_TOKEN")
        return
    
    print(f"\n✅ GitHub Token 已配置: {client.github_token[:10]}...")
    print(f"   MCP Server: {client.server_path}")
    
    try:
        print("\n连接到 MCP Server...")
        await client.connect()
        print("✅ 连接成功!\n")
        
        # 测试 1: 搜索仓库
        print("-"*60)
        print("测试 1: 搜索仓库 'LoRa gateway'")
        print("-"*60)
        repos = await client.search_repositories("LoRa gateway", 3)
        print(f"✅ 找到 {len(repos)} 个仓库")
        print(f"   数据类型: {type(repos)}")
        print(f"   数据预览: {str(repos)[:500]}...")
        
        # 测试 2: 获取 README
        if repos and isinstance(repos, list) and len(repos) > 0:
            print("\n" + "-"*60)
            repo = repos[0]
            if isinstance(repo, dict):
                full_name = repo.get('full_name', '')
                if full_name and '/' in full_name:
                    owner, repo_name = full_name.split('/')
                    print(f"测试 2: 获取 {owner}/{repo_name} 的 README")
                    print("-"*60)
                    readme = await client.get_file_contents(owner, repo_name)
                    if readme:
                        print(f"✅ README 获取成功")
                        print(f"   长度: {len(readme)} 字符")
                        print(f"   预览:\n   {readme[:200]}...")
                    else:
                        print("⚠️ 未找到 README")
            else:
                print(f"⚠️ 仓库数据格式不正确: {type(repo)}")
        
        print("\n" + "="*60)
        print("✅ 所有测试完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n断开连接...")
        await client.disconnect()
        print("✅ 已断开连接")


if __name__ == "__main__":
    asyncio.run(test())
