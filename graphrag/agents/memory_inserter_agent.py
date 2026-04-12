import json
from graphrag.models.llm import LLMModel
from graphrag.models.embedding import EmbeddingModel

class MemoryInserterAgent:
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        self.llm = LLMModel()
        self.embedding = EmbeddingModel()
        self.similarity_threshold = 0.5
    
    def insert_memory(self, structured_info, user_id=1):
        """插入记忆到数据库"""
        try:
            if not structured_info or len(structured_info) == 0:
                # 空数组不算错误，只是没有可提取的信息
                return True
            
            # 存储映射: temp_id -> 实际数据库ID
            id_mapping = {}
            
            # 第一步：持久化所有信息单元
            for item in structured_info:
                temp_id = item.get('temp_id')
                node_type = self._map_type(item.get('type', ''))
                content = item.get('content', '')
                embedding = item.get('embedding')
                
                if node_type and content:
                    node_id = self.db.insert_node(node_type, content, embedding=embedding, user_id=user_id)
                    if node_id:
                        id_mapping[temp_id] = {
                            'id': node_id,
                            'type': node_type,
                            'content': content
                        }
            
            # 第二步：建立关联关系
            for item in structured_info:
                temp_id = item.get('temp_id')
                related_ids = item.get('related_ids', [])
                item_type = item.get('type', '')
                
                # 跳过非链接类型的单元
                if item_type not in ['线索', '关系']:
                    continue
                
                # 为每个相关ID建立链接
                for related_temp_id in related_ids:
                    if related_temp_id in id_mapping:
                        # 使用实际数据库ID而不是temp_id
                        self.db.insert_edge(
                            self._map_type(item.get('type', 'clue')),
                            id_mapping[item.get('temp_id')]['id'],
                            id_mapping[related_temp_id]['type'],
                            id_mapping[related_temp_id]['id'],
                            'related_to'
                        )
            
            # 第三步：查找相关记忆并建立新连接
            self._establish_connections(id_mapping, user_id)
            
            return True
        except Exception as e:
            self.logger.error(f"插入记忆失败: {e}")
            return False
    
    def _map_type(self, type_str):
        """类型映射"""
        type_map = {
            '经历': 'experiences',
            '灵感': 'inspirations',
            '提醒': 'reminders',
            '闲绪': 'miscellaneous_thoughts',
            '人物': 'people',
            '事件': 'events',
            '地点': 'places',
            '关系': 'connections',
            '线索': 'connections'
        }
        return type_map.get(type_str, type_str)
    
    def _establish_connections(self, new_nodes, user_id=1):
        """建立新节点与现有节点的连接"""
        for temp_id, node_info in new_nodes.items():
            content = node_info['content']
            node_type = node_info['type']
            
            # 生成嵌入并搜索相似节点（只搜索当前用户的）
            embedding = self.embedding.get_embedding(content)
            similar_nodes = self.db.search_similar_nodes(embedding, top_k=3, user_id=user_id)
            
            # 遍历相似节点
            for similar_node in similar_nodes:
                # 跳过自身
                if similar_node['id'] == node_info['id']:
                    continue
                
                # 低于相似度阈值才建立连接
                if similar_node.get('distance', 1) < self.similarity_threshold:
                    # 使用LLM判断是否建立连接
                    if self._should_connect(content, similar_node['content']):
                        # 创建线索单元
                        clue_content = f"{content} 与 {similar_node['content']} 相关"
                        clue_type = self._get_opposite_type(node_type)
                        
                        clue_id = self.db.insert_node(clue_type, clue_content, embedding=self.embedding.get_embedding(clue_content), user_id=user_id)
                        if clue_id:
                            # 建立连接
                            self.db.insert_edge(
                                clue_type,
                                clue_id,
                                node_type,
                                node_info['id'],
                                'related_to'
                            )
                            self.db.insert_edge(
                                clue_type,
                                clue_id,
                                similar_node['type'],
                                similar_node['id'],
                                'related_to'
                            )
    
    def _should_connect(self, new_content, existing_content):
        """使用LLM判断是否应该建立连接"""
        prompt = f"""判断以下两条信息是否存在有意义的关联：

新信息：{new_content}
已有信息：{existing_content}

如果存在有意义的关联（如因果、上下文、主题相关等），请回答"是"，否则回答"否"。"""
        try:
            response = self.llm.chat(prompt)
            return "是" in response
        except Exception as e:
            print(f"判断连接失败: {e}")
            return False
    
    def _get_opposite_type(self, node_type):
        """获取链接表类型"""
        return 'connections'