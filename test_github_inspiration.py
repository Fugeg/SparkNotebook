"""
测试 GitHub 灵感模式（带 MCP Server 支持）
"""
import asyncio
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag.agents.chat_agent import ChatAgent
from graphrag.db.database import Database
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_github_inspiration():
    """测试 GitHub 灵感模式"""
    print("="*60)
    print("GitHub 灵感模式测试")
    print("="*60)
    
    # 检查配置
    use_mcp = os.getenv('USE_GITHUB_MCP', 'false').lower() == 'true'
    print(f"\n配置:")
    print(f"  USE_GITHUB_MCP: {use_mcp}")
    print(f"  GITHUB_TOKEN: {'已配置' if os.getenv('GITHUB_TOKEN') else '未配置'}")
    print(f"  STEPFUN_API_KEY: {'已配置' if os.getenv('STEPFUN_API_KEY') else '未配置'}")
    
    # 初始化 ChatAgent
    print("\n初始化 ChatAgent...")
    db = Database()
    agent = ChatAgent(db, logger)
    
    # 测试查询
    test_query = "我想做一个图像识别的项目"
    
    print(f"\n测试查询: {test_query}")
    print("-"*60)
    
    try:
        response = await agent.handle_github_inspiration(test_query, user_id=1)
        print("\n" + "="*60)
        print("生成的报告:")
        print("="*60)
        print(response[:500] + "..." if len(response) > 500 else response)
        print("\n✅ 测试完成!")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_github_inspiration())
