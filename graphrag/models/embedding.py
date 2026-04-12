import os
import dashscope
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class EmbeddingModel:
    def __init__(self):
        dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.model = os.getenv('EMBEDDING_MODEL', 'text-embedding-v3')
    
    def get_embedding(self, text):
        try:
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=text
            )
            return response.output['embeddings'][0]['embedding']
        except Exception as e:
            print(f"获取嵌入失败: {e}")
            return np.random.rand(1024).tolist()
