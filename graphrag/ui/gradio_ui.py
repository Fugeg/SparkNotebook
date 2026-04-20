import gradio as gr

def create_interface(chat_agent, db):
    """创建 Gradio 界面 - 纯聊天对话式布局（带用户权限控制和模式切换）"""
    
    # 当前登录用户
    current_user = {"username": None, "is_admin": False}
    
    # 当前模式："chat" = 闲聊模式(意图1), "notebook" = 记事本模式(意图2,3)
    current_mode = {"mode": "chat"}
    
    def get_user_id(username):
        """根据用户名获取用户 ID"""
        user = db.get_user_by_username(username)
        return user['id'] if user else 1
    
    def get_chat_history_for_ui(username):
        """获取对话历史用于显示"""
        if not username:
            return []
        user_id = get_user_id(username)
        history = db.get_chat_history(user_id, limit=50)
        chat_history = []
        for item in history:
            chat_history.append({"role": "user", "content": item['user_input']})
            chat_history.append({"role": "assistant", "content": item['response']})
        return chat_history
    
    def login(username):
        """用户登录"""
        if not username or not username.strip():
            return "请输入用户名", gr.update(interactive=False), "未登录", "", [], gr.update(value="闲聊模式")
        
        username = username.strip()
        user = db.get_user_by_username(username)
        
        if not user:
            return f"用户 '{username}' 不存在，请先注册", gr.update(interactive=False), "未登录", "", [], gr.update(value="闲聊模式")
        
        # 登录成功
        current_user["username"] = username
        current_user["is_admin"] = (username == "fugeg")
        
        # 管理员可以切换用户，普通用户只能使用自己的账户
        dropdown_interactive = current_user["is_admin"]
        permission = "管理员" if current_user["is_admin"] else "普通用户"
        
        # 返回登录状态、用户下拉框、权限、当前登录用户、聊天历史、模式选择
        return f"登录成功！欢迎 {username}", gr.update(interactive=dropdown_interactive), permission, username, get_chat_history_for_ui(username), gr.update(value="闲聊模式")
    
    def register_user(username, chatbot_history):
        """创建新用户"""
        if not username or not username.strip():
            return "请输入用户名", "", gr.update(choices=db.list_users()), chatbot_history
        
        username = username.strip()
        user_id = db.create_user(username, "")
        if user_id:
            users = db.list_users()
            choices = [u['username'] for u in users]
            return f"用户 '{username}' 创建成功！请登录", "", gr.update(choices=choices), chatbot_history
        else:
            return "创建用户失败，用户名可能已存在", "", gr.update(), chatbot_history
    
    def refresh_users():
        """刷新用户列表（仅管理员可用）"""
        users = db.list_users()
        choices = [u['username'] for u in users] if users else ['default_user']
        return gr.update(choices=choices)
    
    def switch_user(new_username, current_login_user):
        """切换用户（仅管理员可用）"""
        if not current_login_user or current_login_user != "fugeg":
            return "只有管理员可以切换用户", current_login_user, "普通用户", []
        
        return f"已切换到用户：{new_username}", new_username, "管理员", get_chat_history_for_ui(new_username)
    
    def on_user_change(username, current_login_user):
        """用户切换时加载历史"""
        if current_login_user != "fugeg":
            # 普通用户不能通过下拉切换
            return f"当前用户：{current_login_user}", current_login_user, []
        
        return f"已切换到：{username}", username, get_chat_history_for_ui(username)
    
    def clear_chat():
        """清空聊天"""
        return []
    
    def switch_mode(mode):
        """切换模式"""
        # mode 是 Radio 按钮的值："闲聊模式"、"记事本模式" 或 "GitHub灵感模式"
        if mode == "闲聊模式":
            current_mode["mode"] = "chat"
            return gr.update(placeholder="💬 闲聊模式：日常对话，AI会自动识别你是否在询问记忆"), "已切换到闲聊模式 - AI会自动识别记忆查询"
        elif mode == "记事本模式":
            current_mode["mode"] = "notebook"
            return gr.update(placeholder="📝 记事本模式：所有输入自动保存为记忆，可随时查询"), "已切换到记事本模式 - 自动记录所有内容"
        else:  # GitHub灵感模式
            current_mode["mode"] = "github"
            return gr.update(placeholder="🐙 GitHub灵感模式：技术咨询，查询GitHub开源项目"), "已切换到GitHub灵感模式"
    
    async def handle_input_with_user(text, username, chatbot_history):
        """处理用户输入（带用户名和模式）"""
        import asyncio
        
        # 检查是否已登录
        if not username or username == "":
            return "", chatbot_history
        
        if not text or not text.strip():
            return "", chatbot_history
        
        # 先显示用户消息
        chatbot_history.append({"role": "user", "content": text})
        
        user_id = get_user_id(username)
        
        # 根据当前模式处理
        if current_mode["mode"] == "chat":
            # 闲聊模式：使用 handle_input 进行意图分类（支持意图4）
            response = await chat_agent.handle_input(text, user_id)
        elif current_mode["mode"] == "github":
            # GitHub灵感模式：强制使用意图4，直接调用 handle_github_inspiration
            # 存储技术灵感报告到 raw_inputs
            response = await chat_agent.handle_github_inspiration(text, user_id)
            
            # 保存技术灵感报告到数据库
            chat_agent.db.insert_raw_input(
                f"[GitHub灵感查询] {text}", 
                input_method='text', 
                user_id=user_id, 
                response_content=response
            )
        else:
            # 记事本模式：用户输入存储为结构化记忆，AI回复只存原始对话
            context = chat_agent._get_conversation_context(user_id)
            username_actual = chat_agent._get_username(user_id)

            # 1. 先存储用户输入的结构化记忆
            user_response = chat_agent.handle_memory_creation(text, 'text', user_id, context, username_actual)

            # 2. 生成AI回复（基于上下文）
            response = chat_agent.handle_chat(text, context, username_actual, user_id=user_id)

            # 3. 存储AI回复的原始内容（不提取结构化信息，只存raw_inputs）
            chat_agent.db.insert_raw_input(f"AI回复：{response}", input_method='text', user_id=user_id, response_content=None)

            # 4. 保存到对话历史
            chat_agent._add_to_history(user_id, text, response)
            chat_agent.db.insert_raw_input(text, input_method='text', user_id=user_id, response_content=response)
        
        # 再显示 AI 回复
        chatbot_history.append({"role": "assistant", "content": response})
        
        return "", chatbot_history
    
    async def transcribe_and_submit(audio, username, chatbot_history):
        """转录语音并提交"""
        # 检查是否已登录
        if not username or username == "":
            return "", chatbot_history
        
        if audio is None:
            return "", chatbot_history
        
        audio_file_path = audio if isinstance(audio, str) else audio[0]
        
        user_id = get_user_id(username)
        
        # 语音转文本
        text = chat_agent.llm.speech_to_text(audio_file_path)
        if text is None or not text.strip():
            chatbot_history.append({"role": "user", "content": f"[语音识别失败]"})
            chatbot_history.append({"role": "assistant", "content": "语音识别失败，请重试。"})
            return "", chatbot_history
        
        # 显示转换后的文本
        chatbot_history.append({"role": "user", "content": f"🎤 {text}"})
        
        # 根据当前模式处理
        if current_mode["mode"] == "chat":
            # 闲聊模式：使用 handle_input 进行意图分类（支持意图4）
            response = await chat_agent.handle_input(text, user_id)
        elif current_mode["mode"] == "github":
            # GitHub灵感模式：强制使用意图4，直接调用 handle_github_inspiration
            # 存储技术灵感报告到 raw_inputs
            response = await chat_agent.handle_github_inspiration(text, user_id)
            
            # 保存技术灵感报告到数据库
            chat_agent.db.insert_raw_input(
                f"[GitHub灵感查询-语音] {text}", 
                input_method='audio', 
                user_id=user_id, 
                response_content=response
            )
        else:
            # 记事本模式：用户语音存储为结构化记忆，AI回复只存原始对话
            context = chat_agent._get_conversation_context(user_id)
            username_actual = chat_agent._get_username(user_id)

            # 1. 存储用户语音转文字的结构化记忆
            chat_agent.handle_memory_creation(text, 'audio', user_id, context, username_actual)

            # 2. 生成AI回复
            response = chat_agent.handle_chat(text, context, username_actual, user_id=user_id)

            # 3. 存储AI回复的原始内容（不提取结构化信息）
            chat_agent.db.insert_raw_input(f"AI回复：{response}", input_method='text', user_id=user_id, response_content=None)

            # 4. 保存到对话历史
            chat_agent._add_to_history(user_id, text, response)
            chat_agent.db.insert_raw_input(text, input_method='audio', user_id=user_id, response_content=response)
        
        # 显示 AI 回复
        chatbot_history.append({"role": "assistant", "content": response})
        
        return "", chatbot_history
    
    async def on_audio_upload(audio, username, chatbot_history):
        """音频文件上传自动转录"""
        # 检查是否已登录
        if not username or username == "":
            return "", chatbot_history
        
        if audio is None:
            return "", chatbot_history
        
        audio_file_path = audio if isinstance(audio, str) else audio[0]
        
        user_id = get_user_id(username)
        
        # 语音转文本
        text = chat_agent.llm.speech_to_text(audio_file_path)
        if text is None or not text.strip():
            chatbot_history.append({"role": "user", "content": f"[语音识别失败]"})
            chatbot_history.append({"role": "assistant", "content": "语音识别失败，请重试。"})
            return "", chatbot_history
        
        # 显示转换后的文本
        chatbot_history.append({"role": "user", "content": f"🎤 {text}"})
        
        # 根据当前模式处理
        if current_mode["mode"] == "chat":
            # 闲聊模式：使用 handle_input 进行意图分类（支持意图4）
            response = await chat_agent.handle_input(text, user_id)
        elif current_mode["mode"] == "github":
            # GitHub灵感模式：强制使用意图4，直接调用 handle_github_inspiration
            # 存储技术灵感报告到 raw_inputs
            response = await chat_agent.handle_github_inspiration(text, user_id)
            
            # 保存技术灵感报告到数据库
            chat_agent.db.insert_raw_input(
                f"[GitHub灵感查询-语音上传] {text}", 
                input_method='audio', 
                user_id=user_id, 
                response_content=response
            )
        else:
            # 记事本模式：用户语音存储为结构化记忆，AI回复只存原始对话
            context = chat_agent._get_conversation_context(user_id)
            username_actual = chat_agent._get_username(user_id)

            # 1. 存储用户语音转文字的结构化记忆
            chat_agent.handle_memory_creation(text, 'audio', user_id, context, username_actual)

            # 2. 生成AI回复
            response = chat_agent.handle_chat(text, context, username_actual, user_id=user_id)

            # 3. 存储AI回复的原始内容（不提取结构化信息）
            chat_agent.db.insert_raw_input(f"AI回复：{response}", input_method='text', user_id=user_id, response_content=None)

            # 4. 保存到对话历史
            chat_agent._add_to_history(user_id, text, response)
            chat_agent.db.insert_raw_input(text, input_method='audio', user_id=user_id, response_content=response)

        # 显示 AI 回复
        chatbot_history.append({"role": "assistant", "content": response})

        return "", chatbot_history

    # CSS 样式 - 清新淡蓝主题
    css = """
    /* 全局背景 - 亮白色 */
    body {
        background: #f8fafc !important;
        color: #334155 !important;
    }
    
    .gradio-container {
        background: #f8fafc !important;
    }
    
    /* 主容器 */
    @media screen and (max-width: 768px) {
        .main-container {
            flex-direction: column !important;
        }
        .left-panel {
            width: 100% !important;
            min-width: 100% !important;
        }
    }
    
    @media screen and (min-width: 769px) {
        .main-container {
            display: flex !important;
            gap: 20px !important;
        }
        .left-panel {
            width: 22% !important;
            min-width: 220px !important;
        }
    }
    
    /* 左侧面板 - 淡蓝卡片效果 */
    .left-panel {
        background: #ffffff !important;
        border: 1px solid #e0f2fe !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 4px 20px rgba(186, 230, 253, 0.4) !important;
    }
    
    /* 标题样式 - 淡蓝色 */
    h1, h2, h3 {
        color: #0ea5e9 !important;
    }
    
    /* 聊天容器 */
    .chat-container {
        height: 75vh !important;
        background: #ffffff !important;
        border: 1px solid #e0f2fe !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(186, 230, 253, 0.4) !important;
    }
    
    .gr-chatbot {
        height: 100% !important;
        background: transparent !important;
    }
    
    /* 聊天消息样式 - 淡蓝背景 */
    .gr-chatbot .message {
        background: #f0f9ff !important;
        border: 1px solid #bae6fd !important;
        border-radius: 12px !important;
        margin: 8px 0 !important;
        padding: 12px 16px !important;
    }
    
    .gr-chatbot .message.user {
        background: #e0f2fe !important;
        border-color: #7dd3fc !important;
    }
    
    .gr-chatbot .message.bot {
        background: #f0f9ff !important;
        border-color: #bae6fd !important;
    }
    
    /* 输入框样式 - 淡蓝边框 */
    input, textarea {
        background: #ffffff !important;
        border: 1px solid #bae6fd !important;
        color: #334155 !important;
        border-radius: 10px !important;
    }
    
    input:focus, textarea:focus {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1) !important;
    }
    
    /* 按钮样式 - 淡蓝色 */
    button {
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        color: white !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        background: linear-gradient(135deg, #7dd3fc 0%, #38bdf8 100%) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Radio 按钮组样式 */
    .gr-radio {
        background: #f0f9ff !important;
        border: 1px solid #bae6fd !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    
    .gr-radio label {
        color: #0ea5e9 !important;
    }
    
    .gr-radio input[type="radio"]:checked + label {
        color: #0284c7 !important;
        font-weight: bold !important;
    }
    
    /* 状态文本 */
    .status-text {
        color: #38bdf8 !important;
        font-size: 14px !important;
    }
    
    /* 分割线 */
    hr {
        border-color: #e0f2fe !important;
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f9ff !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7dd3fc 0%, #38bdf8 100%) !important;
        border-radius: 4px !important;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #38bdf8 0%, #0ea5e9 100%) !important;
    }
    
    /* 模式切换按钮样式 */
    .mode-switcher {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 15px;
        padding: 10px;
        background: #f0f9ff !important;
        border: 1px solid #bae6fd !important;
        border-radius: 25px !important;
    }
    
    .mode-btn {
        padding: 8px 20px;
        border-radius: 20px;
        border: none;
        cursor: pointer;
        font-size: 14px;
        transition: all 0.3s;
        background: #e0f2fe !important;
        color: #0ea5e9 !important;
    }
    
    .mode-btn.active {
        background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%) !important;
        color: white !important;
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3) !important;
    }

    .mode-btn:not(.active):hover {
        background: #bae6fd !important;
        color: #0284c7 !important;
    }

    /* 文本框标签 */
    label {
        color: #0ea5e9 !important;
        font-weight: 500 !important;
    }

    /* Markdown 文本 */
    .markdown-text {
        color: #334155 !important;
    }

    /* 链接样式 */
    a {
        color: #0ea5e9 !important;
        text-decoration: none !important;
    }

    a:hover {
        color: #38bdf8 !important;
    }
    """

    with gr.Blocks(css=css, theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 2.5em; margin: 0; color: #0ea5e9;">
                ⚡ SparkNotebook 
            </h1>
            <p style="color: #38bdf8; font-size: 1.1em; margin-top: 10px; letter-spacing: 2px;">
                ▰▰▰ 基于GraphRAG的知识管理与需求决策平台 ▰▰▰
            </p>
        </div>
        """)
        
        with gr.Row():
            # 左侧面板 - 用户管理和设置
            with gr.Column(scale=1, min_width=200):
                gr.Markdown("### 👤 用户管理")
                
                # 登录区域
                gr.Markdown("**登录**")
                login_input = gr.Textbox(
                    label="", 
                    placeholder="输入用户名",
                    lines=1
                )
                login_btn = gr.Button("🔑 登录", variant="primary", size="sm")
                login_status = gr.Textbox(label="状态", value="未登录", interactive=False)
                
                gr.Markdown("---")
                
                # 注册区域
                gr.Markdown("**注册新用户**")
                register_input = gr.Textbox(
                    label="", 
                    placeholder="新用户名",
                    lines=1
                )
                register_btn = gr.Button("📝 注册", variant="secondary", size="sm")
                
                gr.Markdown("---")
                
                # 用户切换（仅管理员）
                gr.Markdown("### 🔄 切换用户")
                gr.Markdown("*仅管理员可用*")
                users = db.list_users()
                user_choices = [u['username'] for u in users] if users else ['default_user']
                user_dropdown = gr.Dropdown(
                    choices=user_choices, 
                    label="选择用户", 
                    value=user_choices[0] if user_choices else None,
                    interactive=False  # 默认禁用
                )
                switch_btn = gr.Button("切换", size="sm")
                
                user_permission = gr.Textbox(label="权限", value="未登录", interactive=False)
                current_login_user = gr.Textbox(label="当前登录", value="", visible=False)
                
                gr.Markdown("---")
                
                # 语音输入
                gr.Markdown("### 🎤 语音输入")
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="",
                    editable=True
                )
                transcribe_btn = gr.Button("🎤 转录发送", variant="primary", size="sm")
                
                gr.Markdown("---")
                
                # 操作按钮
                gr.Markdown("### ⚙️ 操作")
                clear_btn = gr.Button("🗑️ 清空聊天", variant="secondary", size="sm")
            
            # 右侧面板 - 聊天对话区域
            with gr.Column(scale=4):
                # 模式切换区域
                gr.Markdown("### 🔄 对话模式")
                with gr.Row():
                    mode_selector = gr.Radio(
                        choices=["闲聊模式", "记事本模式", "GitHub灵感模式"],
                        value="闲聊模式",
                        label="",
                        interactive=True
                    )
                mode_status = gr.Textbox(
                    label="当前模式",
                    value="已切换到闲聊模式",
                    interactive=False,
                    show_label=True
                )
                
                gr.Markdown("---")
                
                # 聊天历史（包含输入框）
                chatbot = gr.Chatbot(
                    label="",
                    height=550,
                    placeholder="💬 请先登录...（未登录只能查看，无法对话）"
                )
                
                # 输入框（集成到聊天界面底部）
                with gr.Row():
                    input_text = gr.Textbox(
                        label="",
                        placeholder="💬 闲聊模式：随便聊聊，问问题，或者打个招呼...",
                        lines=1,
                        max_lines=4,
                        show_label=False,
                        scale=5,
                        autofocus=True,
                        container=False
                    )
                    submit_btn = gr.Button("🚀", variant="primary", scale=1, size="lg")
        
        # 事件处理 - 登录
        login_btn.click(
            fn=login,
            inputs=[login_input],
            outputs=[login_status, user_dropdown, user_permission, current_login_user, chatbot, mode_selector]
        )
        
        # 事件处理 - 注册
        register_btn.click(
            fn=register_user,
            inputs=[register_input, chatbot],
            outputs=[login_status, register_input, user_dropdown, chatbot]
        )
        
        # 事件处理 - 切换用户
        switch_btn.click(
            fn=switch_user,
            inputs=[user_dropdown, current_login_user],
            outputs=[login_status, current_login_user, user_permission, chatbot]
        )
        
        # 事件处理 - 用户切换时更新
        user_dropdown.change(
            fn=on_user_change,
            inputs=[user_dropdown, current_login_user],
            outputs=[login_status, current_login_user, chatbot]
        )
        
        # 事件处理 - 模式切换
        mode_selector.change(
            fn=switch_mode,
            inputs=[mode_selector],
            outputs=[input_text, mode_status]
        )
        
        # 事件处理 - 提交输入
        submit_btn.click(
            fn=handle_input_with_user,
            inputs=[input_text, current_login_user, chatbot],
            outputs=[input_text, chatbot]
        )
        
        input_text.submit(
            fn=handle_input_with_user,
            inputs=[input_text, current_login_user, chatbot],
            outputs=[input_text, chatbot]
        )
        
        # 语音转录功能
        transcribe_btn.click(
            fn=transcribe_and_submit,
            inputs=[audio_input, current_login_user, chatbot],
            outputs=[input_text, chatbot]
        )
        
        # 音频文件上传自动转录
        audio_input.change(
            fn=on_audio_upload,
            inputs=[audio_input, current_login_user, chatbot],
            outputs=[input_text, chatbot]
        )
        
        # 清空聊天
        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot]
        )
    
    return interface
