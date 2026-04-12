"""
GraphRAG 记忆管理工具 - Smolagents 版本

使用 @tool 装饰器定义的工具，用于记忆生成、插入和检索
与原有 Agent 实现保持逻辑一致
"""
import json
import re
import os
from typing import Optional, List, Dict, Any
from smolagents import tool

# 导入原有的模型和数据库
from graphrag.models.llm import LLMModel
from graphrag.models.embedding import EmbeddingModel


class ToolContext:
    """工具上下文，用于共享数据库和日志器"""
    db = None
    logger = None
    llm = None
    embedding = None
    
    @classmethod
    def set_context(cls, db, logger):
        cls.db = db
        cls.logger = logger
        cls.llm = LLMModel()
        cls.embedding = EmbeddingModel()


def _clean_json_response(text: str) -> str:
    """清理 LLM 返回的 JSON，移除 Markdown 代码块标记"""
    # 移除 ```json 或 ``` 开头的标记
    text = re.sub(r'^\s*```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*```\s*', '', text)
    # 移除结尾的 ``` 标记
    text = re.sub(r'\s*```\s*$', '', text)
    # 去除前后空白
    text = text.strip()
    return text


def _map_type(type_str: str) -> str:
    """类型映射：中文类型 -> 数据库类型"""
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


def _add_embeddings(structured_info: List[Dict]) -> None:
    """为信息单元添加嵌入向量"""
    if ToolContext.embedding is None:
        ToolContext.embedding = EmbeddingModel()
    
    for item in structured_info:
        content = item.get('content', '')
        item['embedding'] = ToolContext.embedding.get_embedding(content)


@tool
def classify_intent(text: str) -> int:
    """
    分类用户输入的意图
    
    Args:
        text: 用户输入的文本
        
    Returns:
        意图类别: 1=闲聊, 2=记录新记忆, 3=搜索记忆
    """
    if ToolContext.llm is None:
        ToolContext.llm = LLMModel()
    return ToolContext.llm.classify_intent(text)


@tool
def extract_information(text: str) -> str:
    """
    从用户输入中提取结构化信息，生成记忆单元
    与 MemoryGeneratorAgent.process_input 逻辑一致
    
    Args:
        text: 用户输入的自然语言文本
        
    Returns:
        JSON 字符串，包含提取的信息单元数组（已添加 embedding）
        
    Example:
        >>> extract_information("今天在深圳遇到了张悦")
        '[{"temp_id": "1", "type": "地点", "content": "深圳", "embedding": [...]}, ...]'
    """
    try:
        if ToolContext.llm is None:
            ToolContext.llm = LLMModel()
        if ToolContext.logger:
            ToolContext.logger.info(f"提取信息: {text[:50]}...")
        
        # 提取信息
        extracted_info = ToolContext.llm.extract_information(text)
        
        if ToolContext.logger:
            ToolContext.logger.info(f"提取的信息: {extracted_info[:100]}...")
        
        # 清理 Markdown 代码块标记
        cleaned_json = _clean_json_response(extracted_info)
        
        if ToolContext.logger:
            ToolContext.logger.info(f"清理后的JSON: {cleaned_json[:100]}...")
        
        # 解析JSON
        structured_info = json.loads(cleaned_json)
        
        # 为每个信息单元生成嵌入
        _add_embeddings(structured_info)
        
        return json.dumps(structured_info, ensure_ascii=False)
    except Exception as e:
        if ToolContext.logger:
            ToolContext.logger.error(f"提取信息失败: {e}")
        return "[]"


@tool
def generate_embedding(text: str) -> List[float]:
    """
    为文本生成向量嵌入
    
    Args:
        text: 需要生成嵌入的文本
        
    Returns:
        向量嵌入数组
    """
    if ToolContext.embedding is None:
        ToolContext.embedding = EmbeddingModel()
    return ToolContext.embedding.get_embedding(text)


@tool
def insert_memory(structured_info_json: str, user_id: int = 1) -> str:
    """
    插入记忆到数据库（完整版，与 MemoryInserterAgent.insert_memory 逻辑一致）
    
    Args:
        structured_info_json: JSON 字符串，包含信息单元数组
        user_id: 用户ID，默认为1
        
    Returns:
        JSON 字符串，包含插入结果和统计信息
    """
    if ToolContext.db is None:
        raise ValueError("数据库未初始化")
    
    try:
        # 解析输入
        structured_info = json.loads(structured_info_json)
        
        if not structured_info or len(structured_info) == 0:
            return json.dumps({"success": False, "message": "没有要插入的信息"})
        
        if ToolContext.logger:
            ToolContext.logger.info(f"开始插入 {len(structured_info)} 个记忆单元")
        
        # 存储映射: temp_id -> 实际数据库ID
        id_mapping = {}
        inserted_count = 0
        
        # 第一步：持久化所有信息单元
        for item in structured_info:
            temp_id = item.get('temp_id')
            node_type = _map_type(item.get('type', ''))
            content = item.get('content', '')
            embedding = item.get('embedding')
            
            if node_type and content:
                node_id = ToolContext.db.insert_node(node_type, content, embedding=embedding, user_id=user_id)
                if node_id:
                    id_mapping[temp_id] = {
                        'id': node_id,
                        'type': node_type,
                        'content': content
                    }
                    inserted_count += 1
        
        if ToolContext.logger:
            ToolContext.logger.info(f"已插入 {inserted_count} 个节点")
        
        # 第二步：建立关联关系
        edge_count = 0
        for item in structured_info:
            temp_id = item.get('temp_id')
            related_ids = item.get('related_ids', [])
            item_type = item.get('type', '')
            
            # 跳过非链接类型的单元
            if item_type not in ['线索', '关系']:
                continue
            
            # 为每个相关ID建立链接
            for related_temp_id in related_ids:
                if related_temp_id in id_mapping and temp_id in id_mapping:
                    # 使用实际数据库ID而不是temp_id
                    ToolContext.db.insert_edge(
                        _map_type(item.get('type', 'clue')),
                        id_mapping[temp_id]['id'],
                        id_mapping[related_temp_id]['type'],
                        id_mapping[related_temp_id]['id'],
                        'related_to'
                    )
                    edge_count += 1
        
        if ToolContext.logger:
            ToolContext.logger.info(f"已建立 {edge_count} 个关系")
        
        # 第三步：查找相关记忆并建立新连接
        connection_count = _establish_connections(id_mapping, user_id)
        
        return json.dumps({
            "success": True,
            "inserted_nodes": inserted_count,
            "established_edges": edge_count,
            "new_connections": connection_count,
            "id_mapping": id_mapping
        }, ensure_ascii=False)
        
    except Exception as e:
        if ToolContext.logger:
            ToolContext.logger.error(f"插入记忆失败: {e}")
        return json.dumps({"success": False, "error": str(e)})


def _should_connect(new_content: str, existing_content: str) -> bool:
    """使用LLM判断是否应该建立连接"""
    prompt = f"""判断以下两条信息是否存在有意义的关联：

新信息：{new_content}
已有信息：{existing_content}

如果存在有意义的关联（如因果、上下文、主题相关等），请回答"是"，否则回答"否"。"""
    try:
        if ToolContext.llm is None:
            ToolContext.llm = LLMModel()
        response = ToolContext.llm.chat(prompt)
        return "是" in response
    except Exception as e:
        if ToolContext.logger:
            ToolContext.logger.error(f"判断连接失败: {e}")
        return False


def _establish_connections(new_nodes: Dict, user_id: int = 1) -> int:
    """建立新节点与现有节点的连接（与原有 Agent 逻辑一致）"""
    if ToolContext.embedding is None:
        ToolContext.embedding = EmbeddingModel()
    
    similarity_threshold = 0.5
    connection_count = 0
    
    for temp_id, node_info in new_nodes.items():
        content = node_info['content']
        node_type = node_info['type']
        
        # 生成嵌入并搜索相似节点（只搜索当前用户的）
        embedding = ToolContext.embedding.get_embedding(content)
        similar_nodes = ToolContext.db.search_similar_nodes(embedding, top_k=3, user_id=user_id)
        
        # 遍历相似节点
        for similar_node in similar_nodes:
            # 跳过自身
            if similar_node['id'] == node_info['id']:
                continue
            
            # 低于相似度阈值才建立连接
            if similar_node.get('distance', 1) < similarity_threshold:
                # 使用LLM判断是否建立连接
                if _should_connect(content, similar_node['content']):
                    # 创建线索单元
                    clue_content = f"{content} 与 {similar_node['content']} 相关"
                    clue_type = 'connections'
                    
                    clue_id = ToolContext.db.insert_node(
                        clue_type, 
                        clue_content, 
                        embedding=ToolContext.embedding.get_embedding(clue_content), 
                        user_id=user_id
                    )
                    if clue_id:
                        # 建立连接
                        ToolContext.db.insert_edge(
                            clue_type,
                            clue_id,
                            node_type,
                            node_info['id'],
                            'related_to'
                        )
                        ToolContext.db.insert_edge(
                            clue_type,
                            clue_id,
                            similar_node['type'],
                            similar_node['id'],
                            'related_to'
                        )
                        connection_count += 1
    
    return connection_count


@tool
def retrieve_memory(query: str, user_id: int = 1) -> str:
    """
    检索记忆（完整版，与 MemoryRetrieverAgent.retrieve_memory 逻辑一致）
    
    Args:
        query: 用户查询
        user_id: 用户ID
        
    Returns:
        JSON 字符串，包含检索到的相关节点列表
    """
    if ToolContext.db is None:
        raise ValueError("数据库未初始化")
    
    try:
        if ToolContext.logger:
            ToolContext.logger.info(f"开始检索记忆: {query[:50]}...")
        
        if ToolContext.embedding is None:
            ToolContext.embedding = EmbeddingModel()
        
        # 配置参数
        max_hops = int(os.getenv('MAX_HOPS', 3))
        top_k = int(os.getenv('TOP_K', 5))
        
        # 生成查询嵌入
        query_embedding = ToolContext.embedding.get_embedding(query)
        
        # 步骤1: 基于Embedding的语义相似度检索
        seed_nodes = ToolContext.db.search_similar_nodes(query_embedding, top_k=top_k, user_id=user_id)
        
        if ToolContext.logger:
            ToolContext.logger.info(f"找到 {len(seed_nodes)} 个种子节点")
        
        if not seed_nodes:
            return json.dumps([], ensure_ascii=False)
        
        # 步骤2: 基于图的扩展检索（多跳检索）
        relevant_nodes = _multi_hop_retrieval(query, seed_nodes, max_hops, user_id)
        
        if ToolContext.logger:
            ToolContext.logger.info(f"多跳检索后得到 {len(relevant_nodes)} 个相关节点")
        
        return json.dumps(relevant_nodes, ensure_ascii=False)
        
    except Exception as e:
        if ToolContext.logger:
            ToolContext.logger.error(f"检索记忆失败: {e}")
        return "[]"


def _llm_filter_nodes(query: str, nodes: List[Dict]) -> List[Dict]:
    """LLM引导的节点筛选"""
    if not nodes:
        return []
    
    if ToolContext.llm is None:
        ToolContext.llm = LLMModel()
    
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
        response = ToolContext.llm.generate(prompt)
        return _parse_llm_filter_response(nodes, response)
    except Exception as e:
        if ToolContext.logger:
            ToolContext.logger.error(f"LLM筛选失败: {e}")
        # 如果失败，返回所有节点
        return nodes[:3]


def _parse_llm_filter_response(nodes: List[Dict], response: str) -> List[Dict]:
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
                    if '扩展' in line or '补充' in line:
                        expand_nodes.append(nodes[node_num - 1])
        except (ValueError, IndexError):
            continue
    
    # 如果没有识别出任何节点，返回前3个
    if not expand_nodes:
        expand_nodes = nodes[:3]
    
    return expand_nodes


def _multi_hop_retrieval(query: str, seed_nodes: List[Dict], max_hops: int, user_id: int) -> List[Dict]:
    """多跳检索（与原有 Agent 逻辑一致）"""
    visited = set()
    relevant_nodes = []
    current_hop = 0
    current_nodes = seed_nodes
    
    while current_hop < max_hops and current_nodes:
        # LLM引导的筛选
        valuable_nodes = _llm_filter_nodes(query, current_nodes)
        
        # 添加有价值的节点到结果
        for node in valuable_nodes:
            node_key = f"{node['type']}:{node['id']}"
            if node_key not in visited:
                visited.add(node_key)
                relevant_nodes.append(node)
        
        # 获取下一跳节点
        next_nodes = []
        for node in valuable_nodes:
            neighbors = ToolContext.db.get_node_neighbors(node['type'], node['id'], user_id=user_id)
            for neighbor in neighbors:
                neighbor_node = ToolContext.db.get_node_by_id(neighbor['type'], neighbor['id'], user_id=user_id)
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


@tool
def generate_chat_response(query: str, context: str = "") -> str:
    """
    生成聊天回复
    
    Args:
        query: 用户查询
        context: 上下文信息（JSON字符串）
        
    Returns:
        AI 回复文本
    """
    try:
        if ToolContext.llm is None:
            ToolContext.llm = LLMModel()
        
        if context:
            context_list = json.loads(context)
            return ToolContext.llm.generate_response(query, context_list)
        else:
            return ToolContext.llm.chat(query)
    except Exception as e:
        if ToolContext.logger:
            ToolContext.logger.error(f"生成回复失败: {e}")
        return "抱歉，我现在有点累，请稍后再试。"


@tool
def save_raw_input(user_input: str, input_method: str = "text", user_id: int = 1, response_content: str = "") -> bool:
    """
    保存原始输入到数据库
    
    Args:
        user_input: 用户输入
        input_method: 输入方式 (text, audio)
        user_id: 用户ID
        response_content: AI回复内容
        
    Returns:
        是否保存成功
    """
    if ToolContext.db is None:
        raise ValueError("数据库未初始化")
    
    try:
        ToolContext.db.insert_raw_input(user_input, input_method=input_method, user_id=user_id, response_content=response_content)
        return True
    except Exception as e:
        if ToolContext.logger:
            ToolContext.logger.error(f"保存原始输入失败: {e}")
        return False


# 为了兼容性，提供类形式的工具包装
class MemoryGeneratorTool:
    """记忆生成工具类"""
    
    def __init__(self):
        self.name = "memory_generator"
        self.description = "从用户输入中提取结构化信息，生成记忆单元"
    
    def __call__(self, text: str) -> str:
        return extract_information(text)


class MemoryInserterTool:
    """记忆插入工具类"""
    
    def __init__(self):
        self.name = "memory_inserter"
        self.description = "将提取的记忆单元插入到数据库中"
    
    def insert(self, structured_info_json: str, user_id: int = 1) -> str:
        return insert_memory(structured_info_json, user_id)


class MemoryRetrieverTool:
    """记忆检索工具类"""
    
    def __init__(self):
        self.name = "memory_retriever"
        self.description = "从数据库中检索相关记忆"
    
    def retrieve(self, query: str, user_id: int = 1) -> str:
        return retrieve_memory(query, user_id)
