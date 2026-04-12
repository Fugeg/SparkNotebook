# graphrag/utils/llm_client.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class DashScopeClient:
    def __init__(self):
        # 自动读取环境变量中的 Key
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        if not self.api_key:
            raise ValueError("❌ 未找到 DASHSCOPE_API_KEY，请检查环境变量配置")
            
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(self, prompt, system_prompt=None, model="qwen-plus-latest", json_mode=False):
        """通用对话接口，支持 JSON 模式（毕设解析必备）"""
        # 默认系统提示词
        if system_prompt is None:
            system_prompt = """# Role
你是「Fugeg · AI 记事本助手」，Fugeg 的个人 AI 伙伴。

# 你的任务
根据用户的不同意图，采取合适的回复方式：

## 模式 1：记录新记忆
- 简短确认 + 总结提取的关键信息
- 示例："已为你记下。核心节点：[项目: HarmonyOS], [状态: Debugging]。"

## 模式 2：搜索记忆
- 基于上下文直接给出答案
- 可做有价值的延展提醒
- 找不到时坦诚回答

## 模式 3：日常闲聊
- 简洁明快，像朋友一样对话
- 符合极客/开发者的语境

# 回答风格
- 简短、直接、自然
- 不要使用 JSON、代码块或技术性语言
- 第一人称：称呼用户为"你"，自称为"我"""

        response_format = {"type": "json_object"} if json_mode else None
        
        try:
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"LLM Chat Error: {e}")
            return None

    def get_embedding(self, text, model="text-embedding-v3"):
        """获取 1536 维向量（论文 3.3.4 节核心选型）"""
        try:
            # 过滤掉空字符串或仅包含空格的文本
            if not text or not text.strip():
                return None
                
            response = self.client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding Error: {e}")
            return None

# 单例模式，方便其他模块直接调用
llm = DashScopeClient()