from datetime import datetime
from graphrag.models.llm import LLMModel
from graphrag.models.embedding import EmbeddingModel
from graphrag.agents.memory_generator_agent import MemoryGeneratorAgent
from graphrag.agents.memory_inserter_agent import MemoryInserterAgent
from graphrag.agents.memory_retriever_agent import MemoryRetrieverAgent
from graphrag.utils.cache_helper import SemanticCache


class ChatAgent:
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        self.llm = LLMModel()
        self.embedding = EmbeddingModel()
        self.memory_generator = MemoryGeneratorAgent(self.db, self.logger)
        self.memory_inserter = MemoryInserterAgent(self.db, self.logger)
        self.memory_retriever = MemoryRetrieverAgent(self.db, self.logger)
        
        # 语义缓存 - 用于缓存 LLM 响应
        self.cache = SemanticCache(
            host='localhost',
            port=6379,
            db=0,
            expire_time=3600*24,  # 24小时过期
            similarity_threshold=0.95  # 相似度阈值
        )

        # 对话上下文管理
        self.conversation_history = {}  # user_id -> 对话历史列表
        self.max_history_length = 10    # 保留最近10轮对话

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

    def _get_username(self, user_id: int) -> str:
        """根据用户ID获取用户名"""
        try:
            # 确保数据库连接
            if not self.db.connect():
                return "用户"
            
            # 从数据库查询用户信息
            query = "SELECT username FROM users WHERE id = %s"
            self.db.cursor.execute(query, (user_id,))
            result = self.db.cursor.fetchone()
            if result:
                return result[0]
        except Exception as e:
            self.logger.error(f"获取用户名失败: {e}")
        finally:
            # 查询完成后断开连接
            self.db.disconnect()
        return "用户"

    async def handle_input(self, user_input, user_id=1):
        """处理用户文本输入（带上下文记忆）"""
        return await self._process_input(user_input, input_method='text', user_id=user_id)

    async def handle_audio_input(self, audio_file_path, user_id=1):
        """处理语音输入，返回 (转换文本, AI回复) 元组"""
        # 语音转文本
        text = self.llm.speech_to_text(audio_file_path)
        if text is None:
            return None, "语音识别失败，请重试。"

        if not text.strip():
            return None, "未能识别到语音内容，请重试。"

        # 处理转换后的文本
        response = await self._process_input(text, input_method='audio', user_id=user_id)
        return text, response

    async def _process_input(self, user_input, input_method='text', user_id=1):
        """统一处理输入（内部方法，带上下文）"""
        try:
            # 获取用户名
            username = self._get_username(user_id)

            # 获取对话上下文
            context = self._get_conversation_context(user_id)

            # 意图识别
            intent = self.llm.classify_intent(user_input)
            self.logger.info(f"用户输入: {user_input}")
            self.logger.info(f"识别意图: {intent}")
            self.logger.info(f"用户名: {username}")
            if context:
                self.logger.info(f"对话上下文长度: {len(context)} 字符")

            response = ""
            if intent == 1:
                # 普通闲聊（带上下文、用户名、用户ID）
                # 如果用户询问记忆，handle_chat 内部会调用记忆检索
                response = self.handle_chat(user_input, context, username, user_id=user_id)
            elif intent == 2:
                # 记录新记忆（带上下文和用户名）
                response = self.handle_memory_creation(user_input, input_method, user_id, context, username)
            elif intent == 3:
                # 搜索/检索记忆（带上下文和用户名）- 专门用于记忆查询
                response = self.handle_memory_retrieval(user_input, user_id, context, username)
            elif intent == 4:
                # GitHub 灵感模式（异步）
                response = await self.handle_github_inspiration(user_input, user_id, context)
            else:
                response = "抱歉，我无法理解您的请求。"

            # 保存到对话历史
            self._add_to_history(user_id, user_input, response)

            # 保存到数据库
            self.db.insert_raw_input(user_input, input_method=input_method, user_id=user_id, response_content=response)

            return response
        except Exception as e:
            self.logger.error(f"处理用户输入失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return "处理您的请求时出现错误，请稍后重试。"

    def _is_asking_for_memories(self, user_input: str) -> bool:
        """
        使用 LLM 判断用户是否在询问记忆/灵感
        比关键词匹配更准确
        """
        system_prompt = """You are an intent classifier. Determine if the user is asking about their stored memories, notes, or past records.

Respond with ONLY "YES" or "NO":
- YES: User is asking about their memories, notes, what they said before, their preferences, or past records
- NO: User is just chatting, asking general questions, or making statements

Examples:
Input: "你还记得我的灵感有哪些吗" -> YES
Input: "我之前说过什么" -> YES
Input: "我喜欢什么颜色" -> YES
Input: "我的笔记里写了什么" -> YES
Input: "你记得我告诉过你的事吗" -> YES
Input: "你好" -> NO
Input: "今天天气怎么样" -> NO
Input: "帮我写个Python排序" -> NO
Input: "什么是人工智能" -> NO"""

        try:
            result = self.llm.chat(
                f"Input: {user_input}\nIs the user asking about their memories?",
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=10
            )
            is_asking = "YES" in result.upper()
            if is_asking:
                print(f"\n🧠 [意图识别] 用户正在询问记忆: '{user_input[:40]}...'\n")
            return is_asking
        except Exception as e:
            self.logger.warning(f"记忆意图识别失败: {e}")
            return False

    def handle_chat(self, user_input, context="", username="用户", use_cache=True, user_id=1):
        """处理普通闲聊（带上下文和用户名，支持语义缓存，支持记忆检索）"""
        
        # 使用 LLM 检测用户是否在询问记忆/灵感（更专业的方式）
        if not context and self._is_asking_for_memories(user_input):
            retrieval_result = self.memory_retriever.retrieve_memory(user_input, user_id=user_id)
            
            if retrieval_result:
                # 有记忆，生成基于记忆的回答
                memory_context = self._format_memories_for_chat(retrieval_result)
                prompt = f"用户问：{user_input}\n\n我从记忆中找到：\n{memory_context}\n\n请基于以上记忆回答用户，自然地提及相关记忆内容。"
                response = self.llm.chat(prompt, username=username)
                
                # 存入缓存
                if use_cache and self.cache.enabled:
                    try:
                        query_vector = self.embedding.get_embedding(user_input)
                        self.cache.set(user_input, query_vector, response, metadata={"type": "chat_with_memory"})
                    except:
                        pass
                
                return response
        
        # 1. 生成查询向量（用于语义缓存）
        query_vector = None
        if use_cache and self.cache.enabled:
            try:
                query_vector = self.embedding.get_embedding(user_input)
            except Exception as e:
                self.logger.warning(f"生成查询向量失败: {e}")
        
        # 2. 尝试从缓存获取（仅当没有上下文时，有上下文说明是多轮对话，不缓存）
        if use_cache and not context and query_vector:
            cached_response = self.cache.get(user_input, query_vector)
            if cached_response:
                log_msg = f"[Redis缓存命中] 查询: '{user_input[:30]}...' -> 返回缓存结果"
                self.logger.info(log_msg)
                print(f"\n🚀 {log_msg}\n")
                return f"[来自缓存] {cached_response}"
        
        # 3. 缓存未命中，调用 LLM
        if context:
            # 如果有上下文，将上下文加入提示
            prompt = f"以下是之前的对话历史：\n{context}\n\n当前用户输入：{user_input}\n\n请结合上下文回复："
            response = self.llm.chat(prompt, username=username)
        else:
            response = self.llm.chat(user_input, username=username)
        
        # 4. 存入缓存（仅当没有上下文时）
        if use_cache and not context and query_vector:
            self.cache.set(user_input, query_vector, response, 
                          metadata={"type": "chat", "username": username})
        
        return response

    def handle_memory_creation(self, user_input, input_method='text', user_id=1, context="", username="用户"):
        """处理记忆创建（带上下文和用户名）"""
        # 先保存原始输入到raw_inputs表
        self.db.insert_raw_input(user_input, input_method=input_method, user_id=user_id)

        # 生成记忆
        structured_info = self.memory_generator.process_input(user_input)
        if structured_info is None:
            # None 表示处理过程中发生错误
            return "处理您的记忆时出现错误，请稍后重试。"

        # 检查是否提取到有效信息
        if len(structured_info) == 0:
            # 空数组表示没有可提取的信息（不是错误）
            return "我仔细看了，这句话里没有需要记录的具体信息呢。你可以说得更详细一点，比如'记住我下周三要开会'。"

        # 插入记忆
        insertion_result = self.memory_inserter.insert_memory(structured_info, user_id=user_id)
        if not insertion_result:
            return "存储您的记忆时出现错误，请稍后重试。"

        # 使用 LLM 生成自然的确认回复
        try:
            # 构建一个简单的确认提示
            confirm_prompt = f"用户说：{user_input}\n\n记忆已成功保存。请用一句话自然、简洁地确认，不要显示技术性的类型标签（如[经历]、[人物]等）。"
            natural_response = self.llm.chat(confirm_prompt, username=username)
            return natural_response
        except:
            # 如果 LLM 调用失败，使用默认回复
            return "好的，我记下了。"

    def handle_memory_retrieval(self, user_input, user_id=1, context="", username="用户", use_cache=True):
        """处理记忆检索（带上下文和用户名，支持语义缓存）"""
        
        # 1. 生成查询向量（用于语义缓存）
        query_vector = None
        if use_cache and self.cache.enabled and not context:
            try:
                query_vector = self.embedding.get_embedding(user_input)
            except Exception as e:
                self.logger.warning(f"生成查询向量失败: {e}")
        
        # 2. 尝试从缓存获取（仅当没有上下文时）
        if use_cache and not context and query_vector:
            cached_response = self.cache.get(user_input, query_vector)
            if cached_response:
                log_msg = f"[Redis缓存命中] 查询: '{user_input[:30]}...' -> 返回缓存结果"
                self.logger.info(log_msg)
                print(f"\n🚀 {log_msg}\n")
                return f"[来自缓存] {cached_response}"
        
        # 3. 缓存未命中，执行检索
        retrieval_result = self.memory_retriever.retrieve_memory(user_input, user_id=user_id)
        if not retrieval_result:
            return "未找到相关记忆。"

        # 4. 生成回答（传入用户名）
        response = self.llm.generate_response(user_input, retrieval_result, username=username)

        # 5. 存入缓存（仅当没有上下文时，且找到了有效记忆）
        if use_cache and not context and query_vector and retrieval_result:
            # 只缓存成功找到记忆的回复，不缓存"未找到"的结果
            if "未找到" not in response and "找不到" not in response:
                self.cache.set(user_input, query_vector, response,
                              metadata={"type": "retrieval", "username": username, "user_id": user_id})

        return response

    def get_cache_stats(self):
        """获取缓存统计信息"""
        return self.cache.get_stats()

    def clear_cache(self):
        """清空所有缓存"""
        self.cache.clear()

    def get_cache_status(self):
        """获取缓存状态"""
        return {
            "enabled": self.cache.enabled,
            "threshold": self.cache.similarity_threshold,
            "expire_time": self.cache.expire_time
        }

    async def handle_github_inspiration(self, user_input: str, user_id: int = 1, context: str = "") -> str:
        """
        处理 GitHub 灵感模式（意图 4）
        
        流程：
        1. 从 GitHub 搜索相关开源项目（支持 MCP Server 模式）
        2. 检索本地历史灵感笔记
        3. 使用阶跃星辰模型生成技术灵感报告
        """
        import asyncio
        import os
        from graphrag.utils.mcp_github_client import MCPGitHubClient, search_github_repos_sync
        from graphrag.utils.stepfun_client import get_stepfun_client

        print(f"\n🔍 [GitHub灵感模式] 正在处理: {user_input[:50]}...\n")

        # 0. 使用阶跃星辰将中文查询翻译为英文关键词
        search_query = user_input
        stepfun = get_stepfun_client()
        if stepfun.is_available():
            print("[0/4] 转换查询关键词...")
            try:
                # 在同步环境中运行异步方法
                loop = asyncio.get_event_loop()
                search_query = await loop.run_in_executor(
                    None, stepfun.translate_and_expand_query, user_input
                )
                print(f"      转换结果: {search_query}")
            except Exception as e:
                print(f"      转换失败: {e}，使用原查询")
                search_query = user_input

        # 1. 搜索 GitHub 项目（支持 MCP Server 模式和 REST API 模式）
        print("[1/4] 搜索 GitHub 相关项目...")
        github_repos = []
        readme_content = ""
        
        # 检查是否启用 MCP Server 模式
        use_mcp = os.getenv('USE_GITHUB_MCP', 'false').lower() == 'true'
        
        if use_mcp:
            # 使用 MCP Server 模式
            print("      [模式: MCP Server]")
            try:
                client = MCPGitHubClient(use_mcp_server=True)
                await client.connect()
                github_repos = await client.search_repositories(search_query, per_page=3)
                print(f"      找到 {len(github_repos)} 个相关项目")
                
                # 2. 获取第一个项目的 README
                if github_repos:
                    print("[2/4] 获取项目详情...")
                    first_repo = github_repos[0]
                    owner, repo = first_repo['full_name'].split('/')
                    readme_content = await client.get_readme(owner, repo)
                    if readme_content:
                        print(f"      已获取 {first_repo['full_name']} 的 README")
                    else:
                        print(f"      未找到 README")
                
                await client.disconnect()
            except Exception as e:
                print(f"      MCP Server 模式失败: {e}，回退到 REST API 模式")
                use_mcp = False
        
        if not use_mcp:
            # 使用 REST API 模式（原有实现）
            print("      [模式: REST API]")
            try:
                github_repos = await asyncio.get_event_loop().run_in_executor(
                    None, search_github_repos_sync, search_query, 3
                )
                print(f"      找到 {len(github_repos)} 个相关项目")
            except Exception as e:
                print(f"      GitHub 搜索失败: {e}")
                github_repos = []

            # 2. 获取第一个项目的 README（如果有）
            if github_repos:
                print("[2/4] 获取项目详情...")
                try:
                    client = MCPGitHubClient(use_mcp_server=False)
                    first_repo = github_repos[0]
                    owner, repo = first_repo['full_name'].split('/')
                    readme_content = await client.get_readme(owner, repo)
                    if readme_content:
                        print(f"      已获取 {first_repo['full_name']} 的 README")
                    else:
                        print(f"      未找到 README")
                except Exception as e:
                    print(f"      获取 README 失败: {e}")

        # 3. 检索本地历史灵感
        print("[3/4] 检索本地历史灵感...")
        try:
            local_notes = self.memory_retriever.retrieve_memory(user_input, user_id=user_id)
            if local_notes:
                print(f"      找到 {len(local_notes)} 条相关笔记")
            else:
                print("      未找到相关本地笔记")
                local_notes = []
        except Exception as e:
            print(f"      本地检索失败: {e}")
            local_notes = []

        # 4. 构建上下文
        github_context = self._format_github_context(github_repos, readme_content)
        local_context = self._format_local_context(local_notes)

        # 5. 使用阶跃星辰生成灵感报告
        print("[4/4] 生成技术灵感报告...")
        stepfun = get_stepfun_client()

        if not stepfun.is_available():
            # 如果阶跃星辰不可用，使用默认的 Qwen 模型
            print("      [警告] 阶跃星辰 API 未配置，使用默认模型")
            prompt = f"""用户需求：{user_input}

## GitHub 参考资料
{github_context}

## 用户历史灵感笔记
{local_context}

请结合以上信息，给出技术建议。"""
            response = self.llm.chat(prompt, username=self._get_username(user_id))
        else:
            response = stepfun.generate_inspiration_report(user_input, github_context, local_context)

        print("      ✓ 报告生成完成\n")
        return response

    def _format_memories_for_chat(self, memories: list) -> str:
        """格式化记忆用于闲聊回复"""
        if not memories:
            return ""
        
        context = ""
        for i, mem in enumerate(memories[:5], 1):
            mem_type = mem.get('type', '笔记')
            mem_content = mem.get('content', '')
            context += f"{i}. [{mem_type}] {mem_content}\n"
        
        return context

    def _format_github_context(self, repos: list, readme: str = "") -> str:
        """格式化 GitHub 项目信息"""
        if not repos:
            return "未找到相关 GitHub 项目。"

        context = "### 相关 GitHub 项目\n\n"
        for i, repo in enumerate(repos[:3], 1):
            context += f"{i}. **{repo['full_name']}**\n"
            context += f"   - 描述: {repo.get('description', '无描述')}\n"
            context += f"   - ⭐ Stars: {repo.get('stars', 0)}\n"
            context += f"   - 语言: {repo.get('language', '未知')}\n"
            context += f"   - 链接: {repo.get('url', '')}\n\n"

        if readme:
            context += f"\n### 项目 README 摘要\n{readme[:1500]}..."

        return context

    def _format_local_context(self, notes: list) -> str:
        """格式化本地灵感笔记"""
        if not notes:
            return "用户暂无相关历史灵感笔记。"

        context = "### 用户历史灵感\n\n"
        for i, note in enumerate(notes[:5], 1):
            note_type = note.get('type', '笔记')
            note_content = note.get('content', '')
            context += f"{i}. [{note_type}] {note_content[:200]}\n\n"

        return context
