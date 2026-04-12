import json
import re
from graphrag.models.llm import LLMModel
from graphrag.models.embedding import EmbeddingModel

class MemoryGeneratorAgent:
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        self.llm = LLMModel()
        self.embedding = EmbeddingModel()
    
    def process_input(self, user_input):
        """处理用户输入，生成结构化信息"""
        try:
            # 提取信息
            extracted_info = self.llm.extract_information(user_input)
            self.logger.info(f"提取的信息: {extracted_info}")

            # 清理 Markdown 代码块标记
            cleaned_json = self._clean_json_response(extracted_info)
            self.logger.info(f"清理后的JSON: {cleaned_json}")

            # 解析JSON
            parsed_data = json.loads(cleaned_json)

            # 兼容旧格式（对象格式）和新格式（数组格式）
            if isinstance(parsed_data, dict):
                # 旧格式：转换为新格式
                structured_info = self._convert_legacy_format(parsed_data)
            elif isinstance(parsed_data, list):
                # 新格式：直接使用
                structured_info = parsed_data
            else:
                self.logger.error(f"未知的JSON格式: {type(parsed_data)}")
                return None

            # 为每个信息单元生成嵌入
            self._add_embeddings(structured_info)

            return structured_info
        except Exception as e:
            self.logger.error(f"处理输入失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _convert_legacy_format(self, legacy_data):
        """将旧格式（对象）转换为新格式（数组）"""
        structured_info = []
        temp_id_counter = 1

        # 处理 direct_information
        for item in legacy_data.get("direct_information", []):
            structured_info.append({
                "temp_id": str(temp_id_counter),
                "type": self._map_type(item.get("type", "")),
                "content": item.get("content", ""),
                "related_ids": []
            })
            temp_id_counter += 1

        # 处理 indirect_entities
        for item in legacy_data.get("indirect_entities", []):
            structured_info.append({
                "temp_id": str(temp_id_counter),
                "type": self._map_type(item.get("type", "")),
                "content": item.get("content", ""),
                "related_ids": []
            })
            temp_id_counter += 1

        # 处理 relations（转换为线索类型）
        for item in legacy_data.get("relations", []):
            structured_info.append({
                "temp_id": str(temp_id_counter),
                "type": "线索",
                "content": item.get("relationship", "") + ": " + item.get("source_content", "") + " -> " + item.get("target_content", ""),
                "related_ids": []
            })
            temp_id_counter += 1

        return structured_info

    def _map_type(self, type_str):
        """类型映射：英文到中文"""
        type_map = {
            "experience": "经历",
            "inspiration": "灵感",
            "reminder": "提醒",
            "thought": "闲绪",
            "event": "事件",
            "person": "人物",
            "place": "地点",
            "entity": "实体",
            "connection": "关系",
            "clue": "线索"
        }
        return type_map.get(type_str.lower(), type_str)
    
    def _clean_json_response(self, text):
        """清理 LLM 返回的 JSON，移除 Markdown 代码块标记"""
        # 移除 ```json 或 ``` 开头的标记
        text = re.sub(r'^\s*```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\s*```\s*', '', text)
        # 移除结尾的 ``` 标记
        text = re.sub(r'\s*```\s*$', '', text)
        # 去除前后空白
        text = text.strip()
        return text
    
    def _add_embeddings(self, structured_info):
        """为信息单元添加嵌入向量"""
        for item in structured_info:
            content = item.get('content', '')
            item['embedding'] = self.embedding.get_embedding(content)