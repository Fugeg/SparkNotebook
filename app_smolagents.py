"""
GraphRAG 系统 - Smolagents 版本

使用 smolagents 框架重新实现的 GraphRAG 系统
"""
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 使用 smolagents 版本的 ChatAgent
from graphrag.smolagents_tools import SmolChatAgent
from graphrag.db.database import Database
from graphrag.utils.logger import Logger

class GraphRAGSystemSmolagents:
    def __init__(self, use_code_agent: bool = True):
        self.logger = Logger()
        self.db = Database()
        # 使用 smolagents 版本的 ChatAgent
        self.chat_agent = SmolChatAgent(self.db, self.logger, use_code_agent=use_code_agent)

    def run(self):
        """启动系统"""
        self.logger.info("Starting GraphRAG System (Smolagents Version)...")
        self.db.initialize()

        # 导入Gradio界面
        from graphrag.ui.gradio_ui import create_interface
        interface = create_interface(self.chat_agent, self.db)

        # 启动HTTP服务
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            max_threads=40,
            show_error=True
        )

if __name__ == "__main__":
    # 可以通过环境变量控制使用 CodeAgent 还是 ToolCallingAgent
    use_code = os.getenv('USE_CODE_AGENT', 'true').lower() == 'true'
    system = GraphRAGSystemSmolagents(use_code_agent=use_code)
    system.run()
