"""
记忆插入器 - 将结构化信息存入数据库
集成实体归一化功能
"""
import json
from graphrag.models.llm import LLMModel
from graphrag.models.embedding import EmbeddingModel
from graphrag.utils.entity_resolver import EntityResolver


class MemoryInserterAgent:
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        self.llm = LLMModel()
        self.embedding = EmbeddingModel()
        self.similarity_threshold = 0.5
        self.entity_resolver = EntityResolver(
            redis_host='localhost',
            redis_port=6379,
            redis_db=1,
            embedding_threshold=0.85
        )

    def insert_memory(self, structured_info, user_id=1):
        """插入记忆到数据库"""
        try:
            if not structured_info or len(structured_info) == 0:
                return True

            id_mapping = {}

            for item in structured_info:
                temp_id = item.get('temp_id')
                node_type = self._map_type(item.get('type', ''))
                content = item.get('content', '')
                embedding = item.get('embedding')

                if node_type and content:
                    canonical_content, is_alias = self._resolve_entity(
                        content, node_type, embedding
                    )

                    if is_alias:
                        self.logger.info(f"实体归一化: '{content}' -> '{canonical_content}'")

                    existing_id = self._find_existing_entity(
                        canonical_content, node_type, user_id
                    )

                    if existing_id:
                        id_mapping[temp_id] = {
                            'id': existing_id,
                            'type': node_type,
                            'content': canonical_content,
                            'is_alias': is_alias
                        }
                    else:
                        node_id = self.db.insert_node(
                            node_type, canonical_content,
                            embedding=embedding, user_id=user_id
                        )
                        if node_id:
                            id_mapping[temp_id] = {
                                'id': node_id,
                                'type': node_type,
                                'content': canonical_content,
                                'is_alias': False
                            }
                            self._register_entity_alias(
                                canonical_content, content, node_type, node_id
                            )

            for item in structured_info:
                temp_id = item.get('temp_id')
                related_ids = item.get('related_ids', [])
                item_type = item.get('type', '')

                if item_type not in ['线索', '关系']:
                    continue

                for related_temp_id in related_ids:
                    if related_temp_id in id_mapping and temp_id in id_mapping:
                        self.db.insert_edge(
                            self._map_type(item.get('type', 'clue')),
                            id_mapping[temp_id]['id'],
                            id_mapping[related_temp_id]['type'],
                            id_mapping[related_temp_id]['id'],
                            'related_to'
                        )

            self._establish_connections(id_mapping, user_id)

            return True
        except Exception as e:
            self.logger.error(f"插入记忆失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _resolve_entity(self, content, entity_type, embedding=None):
        """使用实体归一化器解析实体"""
        try:
            entity_info = {
                'name': content,
                'content': content,
                'embedding': embedding or self.embedding.get_embedding(content),
                'type': entity_type
            }

            canonical_name, is_alias = self.entity_resolver.resolve(entity_info)
            return canonical_name, is_alias
        except Exception as e:
            self.logger.error(f"实体解析失败: {e}")
            return content, False

    def _find_existing_entity(self, content, entity_type, user_id):
        """查找已存在的实体"""
        try:
            embedding = self.embedding.get_embedding(content)
            similar_nodes = self.db.search_similar_nodes(
                embedding, top_k=5, user_id=user_id
            )

            for node in similar_nodes:
                if node['type'] == entity_type:
                    name_sim = self._string_similarity(content, node['content'])
                    if name_sim >= 0.85:
                        return node['id']

            return None
        except Exception as e:
            self.logger.error(f"查找已有实体失败: {e}")
            return None

    def _string_similarity(self, s1, s2):
        """计算字符串相似度（编辑距离）"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return 0.0

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        distance = previous_row[-1]
        max_len = max(len(s1), len(s2))
        return 1 - (distance / max_len) if max_len > 0 else 1.0

    def _register_entity_alias(self, canonical, alias, entity_type, entity_id):
        """注册实体别名关系"""
        try:
            entity_info = {
                'name': alias,
                'content': canonical,
                'type': entity_type
            }
            self.entity_resolver.resolve(entity_info)
        except Exception as e:
            self.logger.error(f"注册别名失败: {e}")

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
            if node_info.get('is_alias'):
                continue

            content = node_info['content']
            node_type = node_info['type']

            embedding = self.embedding.get_embedding(content)
            similar_nodes = self.db.search_similar_nodes(
                embedding, top_k=3, user_id=user_id
            )

            for similar_node in similar_nodes:
                if similar_node['id'] == node_info['id']:
                    continue

                if similar_node.get('distance', 1) < self.similarity_threshold:
                    if self._should_connect(content, similar_node['content']):
                        clue_content = f"{content} 与 {similar_node['content']} 相关"
                        clue_type = 'connections'

                        clue_id = self.db.insert_node(
                            clue_type, clue_content,
                            embedding=self.embedding.get_embedding(clue_content),
                            user_id=user_id
                        )
                        if clue_id:
                            self.db.insert_edge(
                                clue_type, clue_id,
                                node_type, node_info['id'],
                                'related_to'
                            )
                            self.db.insert_edge(
                                clue_type, clue_id,
                                similar_node['type'], similar_node['id'],
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
