"""
GraphRAG Chat Agent - Smolagents 版本

使用 smolagents 的 CodeAgent 实现多步推理和工具调用
"""
import json
import os
import time
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel
from smolagents.default_tools import FinalAnswerTool

from graphrag.models.llm import LLMModel
from graphrag.smolagents_tools.memory_tools import (
    ToolContext,
    classify_intent,
    extract_information,
    generate_embedding,
    insert_memory,
    retrieve_memory,
    generate_chat_response,
    save_raw_input
)


class SmolChatAgent:
    """
    基于 smolagents 的聊天 Agent
    
    使用 CodeAgent 实现多步推理，自动选择合适的工具处理用户输入
    支持对话上下文记忆
    """
    
    def __init__(self, db, logger, use_code_agent: bool = True):
        """
        初始化 SmolChatAgent
        
        Args:
            db: 数据库实例
            logger: 日志器
            use_code_agent: 是否使用 CodeAgent（否则使用 ToolCallingAgent）
        """
        self.db = db
        self.logger = logger
        
        # 设置工具上下文
        ToolContext.set_context(db, logger)
        
        # 初始化 LLM
        self.llm = LLMModel()
        
        # 创建 smolagents 模型适配器
        self.model = self._create_model()
        
        # 创建工具列表
        self.tools = self._create_tools()
        
        # 创建 Agent
        self.use_code_agent = use_code_agent
        self.agent = self._create_agent()
        
        # 对话上下文管理
        self.conversation_history = {}  # user_id -> 对话历史列表
        self.max_history_length = 10    # 保留最近10轮对话
        
        self.logger.info("SmolChatAgent 初始化完成")
    
    def _create_model(self):
        """创建 smolagents 模型适配器"""
        # 使用 LiteLLMModel 支持 DashScope
        try:
            # 配置 DashScope API
            api_key = os.getenv('DASHSCOPE_API_KEY')
            if not api_key:
                self.logger.warning("DASHSCOPE_API_KEY 未设置，使用默认配置")

            model = LiteLLMModel(
                model_id="dashscope/qwen-plus-latest",
                api_key=api_key,
                temperature=0.4
            )
            return model
        except Exception as e:
            self.logger.error(f"创建 LiteLLMModel 失败: {e}")
            # 降级到使用本地 transformers
            try:
                from smolagents import TransformersModel
                self.logger.info("降级到 TransformersModel")
                return TransformersModel(model_id="Qwen/Qwen2.5-7B-Instruct")
            except Exception as e2:
                self.logger.error(f"创建 TransformersModel 也失败: {e2}")
                raise
    
    def _create_tools(self) -> List[Any]:
        """创建工具列表"""
        # 定义工具函数 - 使用与原有 Agent 逻辑一致的完整版工具
        tools = [
            classify_intent,
            extract_information,
            generate_embedding,
            insert_memory,      # 完整版：包含插入节点、建立关系、自动连接
            retrieve_memory,    # 完整版：包含多跳检索、LLM筛选
            generate_chat_response,
            save_raw_input,
            FinalAnswerTool()
        ]
        return tools
    
    def _create_agent(self):
        """创建 smolagents Agent"""

        # 系统提示词
        system_prompt = """# Role
你是「Fugeg · AI 记事本助手」，一个基于 GraphRAG（知识图谱检索增强生成）技术的下一代个人知识管理（PKM）中枢。你的核心使命是帮助 Fugeg 构建、维护和探索他的个人数字大脑。

# Core Mindset (图谱思维)
你不仅仅是一个对话机器人，你是一个"知识编织者"。在处理信息时，你必须始终保持图谱思维：
1. **实体感知 (Entities)：** 敏锐地捕捉文本中的关键节点（如：人物、地点、项目、技术栈、时间节点、核心概念）。
2. **关系洞察 (Relationships)：** 识别实体之间的连接（如：属于、包含、发生于、相关联、不喜欢）。
3. **全局视角 (Global Context)：** 懂得将零散的信息片段，放入 Fugeg 整体的生命线和知识网络中去理解，寻找隐藏的关联（Insights）。

# Available Tools (可用工具)
1. classify_intent - 分类用户意图（1=闲聊，2=记录记忆，3=搜索记忆）
2. extract_information - 从文本中提取结构化信息（实体和关系），自动添加 embedding
3. generate_embedding - 生成文本的向量嵌入
4. insert_memory - 完整版记忆插入（包含：插入节点、建立关系、自动连接相似记忆）
5. retrieve_memory - 完整版记忆检索（包含：语义搜索、多跳检索、LLM筛选）
6. generate_chat_response - 生成聊天回复
7. save_raw_input - 保存用户输入到对话历史

# Operational Workflows (操作流)

## 模式 1：记录新记忆 (知识入库)
当用户陈述新的事实、想法或经历时（classify_intent 返回 2）：
- **你的任务：** 从输入中精准提取【实体】和【关系】，并以结构化的方式保存到知识图谱。
- **执行步骤：**
  1. 使用 extract_information 提取信息单元（返回的JSON已包含embedding）
  2. 使用 insert_memory 插入完整记忆（自动处理节点插入、关系建立、相似记忆连接）
  3. 使用 generate_chat_response 生成确认回复
  4. 使用 save_raw_input 保存对话记录

## 模式 2：搜索记忆 (知识检索)
当用户提问过去的事情或需要你回忆时（classify_intent 返回 3）：
- **你的任务：** 基于检索到的上下文，进行多跳推理，综合生成准确的答案。绝对不要编造 Fugeg 的个人信息（反幻觉）。
- **执行步骤：**
  1. 使用 retrieve_memory 进行完整检索（包含语义搜索、多跳扩展、LLM筛选）
  2. 使用 generate_chat_response 基于检索到的上下文生成回答
  3. 使用 save_raw_input 保存对话记录
- **注意：** 如果找不到相关信息，坦诚回答"在你的知识库中没有找到关于此的记录"，不要瞎猜。

## 模式 3：日常闲聊 (通用对话)
当用户只是在打招呼或进行非个人知识的通用讨论时（classify_intent 返回 1）：
- **你的任务：** 保持自然、友好、懂技术的交流。
- **执行步骤：**
  1. 使用 generate_chat_response 直接生成回复
  2. 使用 save_raw_input 保存对话记录
- **重要：** 回复必须是自然、对话式的语言，不要返回 JSON 格式或技术性的数据结构。像朋友一样聊天。

# Tone & Style (语气与风格)
- **极客且优雅：** 语气专业、冷静、高效，不使用过分夸张的情感词汇。
- **结构化表达：** 尽量使用 Markdown（如列表、加粗）来展示复杂信息，让结果一目了然。
- **第一人称视角：** 称呼用户为 "你"，自称为 "我" 或 "你的记事本"。

# Guardrails (边界限制)
- 永远以提供的检索上下文为事实基准，不要用你自带的通用预训练数据来覆盖或猜测 Fugeg 的个人隐私和偏好。
- 在记录记忆时，确保准确提取实体和关系，不要遗漏关键信息。
- 在搜索记忆时，如果上下文不足，明确告知用户，不要编造信息。

# Output Format (输出格式要求)
- **绝对禁止**返回 JSON、代码块、或任何机器可读的数据格式。
- **必须**使用自然、流畅的中文对话形式回复用户。
- 即使是确认记忆已保存，也应该用自然的语言，例如："好的，我记下了。"而不是技术性的描述。
- 回复应该简洁、友好、符合日常对话习惯。

请仔细分析用户需求，按照上述工作流程选择合适的工具完成任务。"""

        # 创建 PromptTemplates - 使用默认模板但替换 system_prompt
        from smolagents import EMPTY_PROMPT_TEMPLATES
        prompt_templates = EMPTY_PROMPT_TEMPLATES.copy()
        prompt_templates['system_prompt'] = system_prompt

        if self.use_code_agent:
            agent = CodeAgent(
                tools=self.tools,
                model=self.model,
                prompt_templates=prompt_templates,
                additional_authorized_imports=["json", "os", "sys"],
                max_steps=10
            )
        else:
            agent = ToolCallingAgent(
                tools=self.tools,
                model=self.model,
                prompt_templates=prompt_templates,
                max_steps=10
            )
        
        return agent
    
    def _get_conversation_context(self, user_id: int) -> str:
        """获取对话上下文"""
        if user_id not in self.conversation_history:
            return ""
        
        history = self.conversation_history[user_id]
        if not history:
            return ""
        
        # 构建上下文字符串
        context_parts = []
        for turn in history[-self.max_history_length:]:
            context_parts.append(f"用户: {turn['user']}")
            context_parts.append(f"AI: {turn['assistant']}")
        
        return "\n".join(context_parts)
    
    def _add_to_history(self, user_id: int, user_input: str, assistant_response: str):
        """添加对话到历史"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # 限制历史长度
        if len(self.conversation_history[user_id]) > self.max_history_length:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history_length:]
    
    def clear_history(self, user_id: int = None):
        """清空对话历史"""
        if user_id is None:
            self.conversation_history.clear()
        else:
            self.conversation_history.pop(user_id, None)
    
    def _extract_final_answer(self, result) -> str:
        """
        从 Agent 执行结果中提取最终答案
        过滤掉思考过程、工具调用等内部信息
        """
        import re
        import json
        
        # 如果结果是字符串，直接处理
        if isinstance(result, str):
            text = result
        else:
            # 如果是其他类型，转换为字符串
            text = str(result)
        
        # 尝试提取 JSON 中的 final_answer 字段
        try:
            # 查找代码块中的 JSON
            json_match = re.search(r'```(?:json)?\s*\n?(\{[\s\S]*?\})\n?```', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                data = json.loads(json_str)
                if 'final_answer' in data:
                    return data['final_answer'].strip()
                if 'response' in data:
                    return data['response'].strip()
        except (json.JSONDecodeError, KeyError):
            pass
        
        # 尝试直接匹配 final_answer 字段
        final_answer_match = re.search(r'"final_answer":\s*"([^"]*(?:\\.[^"]*)*)"', text, re.DOTALL)
        if final_answer_match:
            answer = final_answer_match.group(1)
            answer = answer.replace('\\"', '"').replace('\\n', '\n')
            return answer.strip()
        
        # 尝试提取 FinalAnswerTool 的内容
        patterns = [
            # 匹配 FinalAnswerTool 的 response 字段
            r'"response":\s*"([^"]*(?:\\.[^"]*)*)"',
            # 匹配 FinalAnswerTool 的文本内容
            r'FinalAnswerTool.*?(?::|=)\s*([\s\S]+?)(?=\n\s*\{|\Z)',
            # 匹配 :::tool::: 之后的最终输出
            r':::tool:::\s*\{[^}]*"response":\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1)
                # 处理转义的引号
                answer = answer.replace('\\"', '"')
                return answer.strip()
        
        # 如果没有找到 FinalAnswerTool，尝试清理其他内容
        # 移除思考过程（Thoughts:...）
        text = re.sub(r'Thoughts?:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除工具调用详情
        text = re.sub(r'\{[^}]*"tool"[^}]*\}', '', text, flags=re.DOTALL)
        
        # 移除代码块标记
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # 移除 :::tool::: 标记
        text = re.sub(r':::tool:::.*?:::tool:::', '', text, flags=re.DOTALL)
        
        # 清理多余空白
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        # 如果清理后还有内容，返回它
        if text and len(text) > 10:
            return text
        
        # 最后的备选：返回原始结果的简化版本
        # 只保留前500个字符
        return str(result)[:500] if result else "处理完成，但没有获取到明确的回复。"
    
    def handle_input(self, user_input: str, user_id: int = 1) -> str:
        """
        处理用户文本输入（带上下文记忆）
        
        Args:
            user_input: 用户输入文本
            user_id: 用户ID
            
        Returns:
            AI 回复
        """
        try:
            # 获取对话上下文
            context = self._get_conversation_context(user_id)
            
            # 构建任务描述（包含上下文）
            if context:
                task = f"""以下是之前的对话历史：
{context}

当前用户输入: "{user_input}"
用户ID: {user_id}

请结合上下文理解用户需求，并按照以下步骤处理：
1. 使用 classify_intent 分析用户意图
2. 根据意图选择后续操作：
   - 意图=1（闲聊）：结合上下文生成回复
   - 意图=2（记录记忆）：提取信息并保存到数据库
   - 意图=3（搜索记忆）：检索相关记忆并回复
3. 保存对话记录
4. 返回最终回复

请使用 FinalAnswerTool 返回最终答案。"""
            else:
                task = f"""用户输入: "{user_input}"
用户ID: {user_id}

请按照以下步骤处理：
1. 使用 classify_intent 分析用户意图
2. 根据意图选择后续操作：
   - 意图=1（闲聊）：直接生成回复
   - 意图=2（记录记忆）：提取信息并保存到数据库
   - 意图=3（搜索记忆）：检索相关记忆并回复
3. 保存对话记录
4. 返回最终回复

请使用 FinalAnswerTool 返回最终答案。"""
            
            # 运行 Agent
            result = self.agent.run(task)
            
            # 提取最终答案（从 Agent 的输出中）
            final_answer = self._extract_final_answer(result)
            
            # 保存到对话历史
            self._add_to_history(user_id, user_input, final_answer)
            
            return final_answer
            
        except Exception as e:
            self.logger.error(f"处理输入失败: {e}")
            return "处理您的请求时出现错误，请稍后重试。"
    
    def handle_audio_input(self, audio_file_path: str, user_id: int = 1) -> Tuple[Optional[str], str]:
        """
        处理语音输入
        
        Args:
            audio_file_path: 音频文件路径
            user_id: 用户ID
            
        Returns:
            (转换后的文本, AI回复)
        """
        try:
            # 语音转文本
            text = self.llm.speech_to_text(audio_file_path)
            
            if text is None:
                return None, "语音识别失败，请重试。"
            
            if not text.strip():
                return None, "未能识别到语音内容，请重试。"
            
            # 处理转换后的文本
            response = self.handle_input(text, user_id)
            
            return text, response
            
        except Exception as e:
            self.logger.error(f"处理语音输入失败: {e}")
            return None, "处理语音时出现错误，请稍后重试。"
    
    def run_direct_task(self, task: str) -> str:
        """
        直接运行任务（用于复杂的多步任务）
        
        Args:
            task: 任务描述
            
        Returns:
            任务执行结果
        """
        try:
            result = self.agent.run(task)
            return result
        except Exception as e:
            self.logger.error(f"执行任务失败: {e}")
            return f"任务执行失败: {str(e)}"


# 保持向后兼容的别名
ChatAgent = SmolChatAgent
