# ~/fugeg/app/test_db_full.py
from graphrag.utils import llm
from graphrag.db import db_handler

test_text = "今天想到了一个关于 AI 自动记事本的绝妙灵感。"
print("1. 正在获取向量...")
vec = llm.get_embedding(test_text)

if vec:
    print(f"2. 向量获取成功 (长度: {len(vec)})，正在存入数据库...")
    new_id = db_handler.insert_inspiration(test_text, vec)
    if new_id:
        print(f"✅ 成功！数据已存入 inspirations 表，ID 为: {new_id}")
    else:
        print("❌ 数据库存储失败")