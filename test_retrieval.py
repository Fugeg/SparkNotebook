"""
测试记忆检索功能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graphrag.db.database import Database
from graphrag.agents.memory_retriever_agent import MemoryRetrieverAgent
from graphrag.utils.logger import Logger

# 初始化
db = Database()
logger = Logger()
retriever = MemoryRetrieverAgent(db, logger)

# 测试检索
test_queries = [
    "我曾经喜欢过谁",
    "我喜欢的人是谁",
    "fgw是谁",
    "我喜欢fgw"
]

print("=" * 60)
print("记忆检索测试 (用户ID=2)")
print("=" * 60)

for query in test_queries:
    print(f"\n查询: '{query}'")
    print("-" * 40)
    result = retriever.retrieve_memory(query, user_id=2)
    if result:
        print(f"找到 {len(result)} 条记忆:")
        for item in result[:3]:
            print(f"  - [{item.get('type', 'unknown')}] {item.get('content', '')[:60]}...")
    else:
        print("未找到相关记忆")

print("\n" + "=" * 60)
