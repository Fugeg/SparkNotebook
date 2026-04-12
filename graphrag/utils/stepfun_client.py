"""
阶跃星辰 API 客户端
用于调用 step-3.5 模型处理 GitHub 灵感模式
"""
import os
import json
from typing import Optional, Dict, Any
from openai import OpenAI


class StepFunClient:
    """
    阶跃星辰 API 客户端
    
    模型：step-3.5
    用途：GitHub 灵感模式的技术咨询和代码生成
    """
    
    def __init__(self):
        self.api_key = os.getenv('STEPFUN_API_KEY')
        self.base_url = os.getenv('STEPFUN_BASE_URL', 'https://api.stepfun.com/step_plan/v1')
        self.model = os.getenv('STEPFUN_MODEL', 'step-3.5')
        
        if not self.api_key:
            print("[StepFun] 警告: STEPFUN_API_KEY 未设置")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
    
    def is_available(self) -> bool:
        """检查客户端是否可用"""
        return self.client is not None
    
    def chat(self, prompt: str, system_prompt: Optional[str] = None, 
             temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        调用阶跃星辰模型进行对话
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            模型回复
        """
        if not self.client:
            return "[错误] 阶跃星辰 API 未配置"
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"[StepFun] API 调用错误: {e}")
            return f"[错误] 调用失败: {str(e)}"
    
    def generate_inspiration_report(self, user_query: str, 
                                    github_context: str,
                                    local_notes: str) -> str:
        """
        生成技术灵感报告
        
        Args:
            user_query: 用户原始查询
            github_context: GitHub 项目信息
            local_notes: 本地历史灵感笔记
            
        Returns:
            灵感报告
        """
        system_prompt = """# Role
You are an AI Technical Inspiration Assistant, skilled at combining GitHub open source trends with user historical thinking to provide innovative technical solution recommendations.

# Task
1. Analyze popular projects and technical trends on GitHub
2. Combine with user's historical inspiration notes
3. Output a structured technical inspiration report

# Output Format
## 📊 GitHub Trend Insights
- Overview of relevant popular projects (must include project links)
- Technology stack analysis

## 🔗 Recommended Project Links
For each recommended project, must include:
- Project name and brief description
- **GitHub Link** (for easy direct access)
- Stars count and primary language

## 💡 Innovation Suggestions (3 differentiated directions)
1. **Innovation 1**: ...
2. **Innovation 2**: ...
3. **Innovation 3**: ...

## 🔧 Reusable Components
- List code/modules from GitHub projects that can be directly used
- Include specific file paths or module names

## 📝 Connection with Historical Inspiration
- How to combine with user's previous ideas

## 🎯 Next Steps Recommendations
- Specific implementation steps
- Recommend which project and which part to check first"""

        prompt = f"""User Requirement: {user_query}

## GitHub Reference Materials
{github_context}

## User Historical Inspiration Notes
{local_notes}

Please combine the above information to generate a technical inspiration report focusing on software development projects, algorithms, and technical tools."""

        return self.chat(prompt, system_prompt=system_prompt, temperature=0.8, max_tokens=2500)


    def translate_and_expand_query(self, user_query: str) -> str:
        """
        将用户查询翻译并扩展为英文关键词，用于 GitHub 搜索
        
        Args:
            user_query: 用户原始查询（可能是中文）
            
        Returns:
            英文关键词，适合 GitHub 搜索
        """
        import re
        
        # 如果已经是纯英文，直接返回
        if user_query.isascii():
            return user_query
        
        # 第一步：提取核心中文关键词
        # 移除常见的无意义词汇
        stop_words = ['一个', '有关', '关于', '的', '项目', '系统', '工具', '框架', '找', '查找', '寻找', '做', '我想', '我要']
        cleaned_query = user_query
        for word in stop_words:
            cleaned_query = cleaned_query.replace(word, ' ')
        cleaned_query = cleaned_query.strip()
        
        # 如果清理后为空，使用原查询
        if not cleaned_query:
            cleaned_query = user_query
        
        # 第二步：尝试使用 StepFun LLM 翻译
        if self.is_available():
            system_prompt = """You are a professional technical translation assistant.
Task: Translate user's Chinese technical requirements into English keywords for searching open source projects on GitHub.

Important Rules:
1. Only translate core technical terms, do not translate generic words like "project", "system", "tool"
2. Focus on software development, programming frameworks, algorithms, and technical tools
3. Expand to 3-5 related technical keywords
4. Return format: main_keyword, related_term1, related_term2...
5. Only return keywords, no explanations
6. Must return valid English keywords, cannot be empty
7. Focus on technical projects (code, software, algorithms), NOT books or documents

Examples:
Input: 橘子识别项目
Output: orange recognition, fruit detection algorithm, computer vision

Input: LoRa网关系统
Output: LoRa gateway, IoT communication protocol, embedded system

Input: 找一个有关 agent mcp的项目
Output: mcp agent, model context protocol, AI agent framework

Input: 橘子
Output: orange detection, fruit classification, image recognition

Input: 图书管理
Output: library management system, book inventory software, catalog application"""

            try:
                result = self.chat(
                    f"Translate the following Chinese technical requirement into English search keywords (focus on software/technical projects only):\n{cleaned_query}\n\nMust return English keywords, cannot be empty:",
                    system_prompt=system_prompt,
                    temperature=0.1,
                    max_tokens=100
                )
                
                # 清理结果
                result = result.strip()
                
                # 移除常见的多余文本
                for prefix in ["输出：", "关键词：", "翻译：", "结果：", "Output:", "Keywords:", "Translation:"]:
                    if result.startswith(prefix):
                        result = result[len(prefix):].strip()
                
                # 检查结果是否有效（至少包含一些英文字母）
                if result and len(result) >= 2 and any(c.isalpha() for c in result):
                    print(f"[StepFun] LLM翻译: '{user_query}' -> '{result}'")
                    return result
                else:
                    print(f"[StepFun] LLM返回无效结果: '{result}'，使用 Qwen-plus 备用方案")
            except Exception as e:
                print(f"[StepFun] LLM翻译失败: {e}，使用 Qwen-plus 备用方案")
        
        # 第三步：使用 Qwen-plus 作为备用翻译方案
        try:
            return self._translate_with_qwen(cleaned_query, user_query)
        except Exception as e:
            print(f"[Qwen] 翻译失败: {e}，使用提取英文单词方案")
        
        # 第四步：后备方案 - 提取英文单词
        english_words = re.findall(r'[a-zA-Z]+', cleaned_query)
        if english_words:
            result = ' '.join(english_words)
            print(f"[StepFun] 提取英文: '{user_query}' -> '{result}'")
            return result
        
        # 最后后备：返回原查询（可能是中文，GitHub 搜索可能也能处理）
        print(f"[StepFun] 无法翻译: '{user_query}'，使用原查询")
        return cleaned_query
    
    def _translate_with_qwen(self, cleaned_query: str, original_query: str) -> str:
        """
        使用 Qwen-plus 进行翻译（StepFun 失败时的备用方案）
        """
        import dashscope
        import os
        
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
        
        system_prompt = """You are a professional technical translation assistant.
Task: Translate Chinese technical requirements into English keywords for GitHub search.

Rules:
1. Only translate core technical terms
2. Focus on software, algorithms, technical tools
3. Return 2-5 related English keywords
4. Format: keyword1, keyword2, keyword3...
5. Only return keywords, no explanations
6. Must be valid English keywords

Examples:
Input: 图像识别
Output: image recognition, computer vision, deep learning

Input: LoRa网关
Output: LoRa gateway, IoT communication

Input: 聊天机器人
Output: chatbot, conversational AI, NLP"""

        user_prompt = f"Translate to English keywords: {cleaned_query}"
        
        try:
            response = dashscope.Generation.call(
                model='qwen-plus-latest',
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            result = response.output.text.strip()
            
            # 清理结果
            for prefix in ["输出：", "关键词：", "翻译：", "结果：", "Output:", "Keywords:", "Translation:"]:
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()
            
            # 验证结果
            if result and len(result) >= 2 and any(c.isalpha() for c in result):
                print(f"[Qwen-plus] 翻译: '{original_query}' -> '{result}'")
                return result
            else:
                raise ValueError(f"Qwen 返回无效结果: '{result}'")
        except Exception as e:
            raise ValueError(f"Qwen API 调用失败: {e}")


# 单例模式
_stepfun_client = None

def get_stepfun_client() -> StepFunClient:
    """获取 StepFunClient 单例"""
    global _stepfun_client
    if _stepfun_client is None:
        _stepfun_client = StepFunClient()
    return _stepfun_client
