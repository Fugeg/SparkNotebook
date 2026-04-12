import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag.agents.chat_agent import ChatAgent
from graphrag.db.database import Database
from graphrag.utils.logger import Logger

class GraphRAGSystem:
    def __init__(self):
        self.logger = Logger()
        self.db = Database()
        self.chat_agent = ChatAgent(self.db, self.logger)

    def run(self):
        """启动系统"""
        self.logger.info("Starting GraphRAG System...")
        self.db.initialize()

        # 导入Gradio界面
        from graphrag.ui.gradio_ui import create_interface
        interface = create_interface(self.chat_agent, self.db)

        # 启动HTTP服务（稳定版本）
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            max_threads=40,
            show_error=True
        )

if __name__ == "__main__":
    system = GraphRAGSystem()
    system.run()
