import os
from graphrag.models.llm import LLMModel
from graphrag.models.embedding import EmbeddingModel
from dotenv import load_dotenv

load_dotenv()

class MemoryRetrieverAgent:
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        self.llm = LLMModel()
        self.embedding = EmbeddingModel()
        self.max_hops = int(os.getenv('MAX_HOPS', 3))
        self.max_tokens = int(os.getenv('MAX_TOKENS', 2000))
        self.top_k = int(os.getenv('TOP_K', 5))
    
    def retrieve_memory(self, query, user_id=1):
        """检索记忆 - 基于论文实现"""
        try:
            # 生成查询嵌入
            query_embedding = self.embedding.get_embedding(query)
            
            # 步骤1: 基于Embedding的语义相似度检索（召回初始相关节点，只搜索当前用户的）
            seed_nodes = self.db.search_similar_nodes(query_embedding, top_k=self.top_k, user_id=user_id)
            self.logger.info(f"用户ID: {user_id}, 找到种子节点: {len(seed_nodes)}")
            
            if not seed_nodes:
                return []
            
            # 步骤2: 基于图的扩展检索（丰富上下文）
            relevant_nodes = self._multi_hop_retrieval(query, seed_nodes, user_id)
            
            return relevant_nodes
        except Exception as e:
            self.logger.error(f"检索记忆失败: {e}")
            return []
    
    def _multi_hop_retrieval(self, query, seed_nodes, user_id=1):
        """多跳检索 - 论文实现"""
        visited = set()
        relevant_nodes = []
        current_hop = 0
        current_nodes = seed_nodes
        
        while current_hop < self.max_hops and current_nodes:
            # LLM引导的筛选：在每一跳扩展后，由LLM判断哪些节点具有再次查询的价值
            valuable_nodes = self._llm_filter_nodes(query, current_nodes)
            
            # 添加有价值的节点到结果
            for node in valuable_nodes:
                node_key = f"{node['type']}:{node['id']}"
                if node_key not in visited:
                    visited.add(node_key)
                    relevant_nodes.append(node)
            
            # 获取下一跳节点
            next_nodes = []
            for node in valuable_nodes:
                neighbors = self.db.get_node_neighbors(node['type'], node['id'], user_id=user_id)
                for neighbor in neighbors:
                    neighbor_node = self.db.get_node_by_id(neighbor['type'], neighbor['id'], user_id=user_id)
                    if neighbor_node:
                        neighbor_key = f"{neighbor['type']}:{neighbor_node['id']}"
                        if neighbor_key not in visited:
                            next_nodes.append({
                                'id': neighbor_node['id'],
                                'type': neighbor['type'],
                                'content': neighbor_node['content'],
                                'metadata': neighbor_node['metadata'],
                                'relationship': neighbor.get('relationship', '')
                            })
            
            current_nodes = next_nodes
            current_hop += 1
        
        return relevant_nodes
    
    def _llm_filter_nodes(self, query, nodes):
        """LLM引导的节点筛选"""
        if not nodes:
            return []
        
        # 构造节点信息
        node_info = []
        for i, node in enumerate(nodes):
            node_info.append(f"节点{i+1}: [{node.get('type', 'unknown')}] {node.get('content', '')[:100]}")
        
        prompt = f"""用户查询：{query}

请评估以下节点与查询的相关性，并将它们分类：
1. 值得再次扩展查询的节点（与查询高度相关，可能有更多相关信息）
2. 作为上下文补充的节点（与查询相关，但不是扩展源）
3. 不相关的节点（应被剪枝）

{chr(10).join(node_info)}

请按以下格式输出（每行一个节点）：
节点1: 分类 (扩展/补充/剪枝)
节点2: 分类
..."""
        try:
            # 使用 llm.chat 方法，传入空字符串作为 system_prompt（使用默认提示词）
            response = self.llm.chat(prompt, username="系统")
            return self._parse_llm_filter_response(nodes, response)
        except Exception as e:
            self.logger.error(f"LLM筛选失败: {e}")
            # 如果失败，返回所有节点
            return nodes[:3]
    
    def _parse_llm_filter_response(self, nodes, response):
        """解析LLM筛选响应"""
        lines = response.strip().split('\n')
        expand_nodes = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 提取节点编号
            try:
                if '节点' in line and ':' in line:
                    node_part = line.split(':')[0]
                    node_num = int(''.join(filter(str.isdigit, node_part.split('节点')[1])))
                    
                    if 1 <= node_num <= len(nodes):
                        if '扩展' in line:
                            expand_nodes.append(nodes[node_num - 1])
                        elif '补充' in line:
                            expand_nodes.append(nodes[node_num - 1])
            except (ValueError, IndexError):
                continue
        
        # 如果没有识别出任何节点，返回前3个
        if not expand_nodes:
            expand_nodes = nodes[:3]
        
        return expand_nodes
