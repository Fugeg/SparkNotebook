"""
实体归一化模块
基于并查集 (Disjoint Set Union) 和语义判定实现实体对齐
"""
import json
import redis
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime


class DisjointSetUnion:
    """
    并查集 (DSU) 数据结构
    用于高效维护等价实体关系
    """

    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.freq = {}
        self.canonical_name = {}

    def make_set(self, x: str):
        """创建新集合"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            self.freq[x] = 1
            self.canonical_name[x] = x

    def find(self, x: str) -> str:
        """路径压缩查找根节点"""
        if x not in self.parent:
            self.make_set(x)
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> bool:
        """
        按秩合并两个集合
        返回 True 表示成功合并，False 表示已在同一集合
        """
        self.make_set(x)
        self.make_set(y)

        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.freq[root_x] += self.freq[root_y]

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        if len(self.canonical_name[root_x]) < len(self.canonical_name[root_y]):
            self.canonical_name[root_x] = self.canonical_name[root_y]

        return True

    def is_same_set(self, x: str, y: str) -> bool:
        """判断是否在同一集合"""
        return self.find(x) == self.find(y)

    def get_canonical(self, x: str) -> str:
        """获取规范名称"""
        return self.canonical_name.get(self.find(x), x)

    def get_freq(self, x: str) -> int:
        """获取出现频率"""
        return self.freq.get(self.find(x), 1)


class EntityResolver:
    """
    实体归一化器
    结合语义判定和并查集实现实体对齐
    """

    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        redis_db: int = 1,
        embedding_threshold: float = 0.85,
        name_match_threshold: float = 0.75,
    ):
        self.dsu = DisjointSetUnion()
        self.embedding_threshold = embedding_threshold
        self.name_match_threshold = name_match_threshold

        self.redis_client = None
        try:
            self.redis_client = redis.StrictRedis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            self.cache_enabled = True
            print(f"[EntityResolver] Redis 连接成功: {redis_host}:{redis_port}/{redis_db}")
        except Exception as e:
            print(f"[EntityResolver] Redis 连接失败: {e}，使用内存存储")
            self.cache_enabled = False
            self._memory_cache = {}

        self.namespace = "entity_resolver:"

    def _get_cache_key(self, entity_name: str) -> str:
        """生成缓存键"""
        return f"{self.namespace}embedding:{hash(entity_name)}"

    def get_embedding_similarity(
        self,
        name1: str,
        name2: str,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """计算两个向量的余弦相似度"""
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """计算两个字符串的编辑距离相似度"""
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

        levenshtein_distance = previous_row[-1]
        max_len = max(len(s1), len(s2))
        return 1 - (levenshtein_distance / max_len) if max_len > 0 else 1.0

    def _semantic_llm_check(
        self,
        name1: str,
        name2: str,
        context1: str = "",
        context2: str = ""
    ) -> Tuple[bool, float]:
        """
        使用 LLM 判断两个实体是否语义等价
        返回 (是否等价, 置信度)
        """
        try:
            from graphrag.models.llm import LLMModel
            llm = LLMModel()

            prompt = f"""判断以下两个实体名称是否指向同一个现实世界中的实体：

实体1: {name1}
上下文1: {context1 or '无'}

实体2: {name2}
上下文2: {context2 or '无'}

判断依据：
1. 同一实体的不同称呼（如"陕科大"和"陕西科技大学"）
2. 包含关系（如"北京大学"和"北大"）
3. 缩写与全称（如"NASA"和"美国国家航空航天局"）

请只回答 JSON 格式：
{{"equivalent": true/false, "confidence": 0.0-1.0, "reason": "简短原因"}}
"""

            response = llm.chat(prompt)
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('equivalent', False), result.get('confidence', 0.0)
        except Exception as e:
            print(f"[EntityResolver] LLM 判断失败: {e}")

        return False, 0.0

    def check_equivalence(
        self,
        entity1: Dict[str, Any],
        entity2: Dict[str, Any],
        use_llm: bool = True
    ) -> Tuple[bool, float]:
        """
        综合判断两个实体是否等价

        Args:
            entity1: {"name": str, "content": str, "embedding": List[float], "neighbors": List[str]}
            entity2: 同上
            use_llm: 是否使用 LLM 进行最终确认

        Returns:
            (是否等价, 置信度)
        """
        name1 = entity1.get('name', entity1.get('content', ''))
        name2 = entity2.get('name', entity2.get('content', ''))
        embedding1 = entity1.get('embedding', [])
        embedding2 = entity2.get('embedding', [])
        context1 = entity1.get('content', '')
        context2 = entity2.get('content', '')

        if not embedding1 or not embedding2:
            name_sim = self._levenshtein_ratio(name1, name2)
            if name_sim >= self.name_match_threshold:
                return True, name_sim
            return False, name_sim

        embedding_sim = self.get_embedding_similarity(name1, name2, embedding1, embedding2)

        if embedding_sim >= 0.95:
            return True, embedding_sim

        if embedding_sim >= self.embedding_threshold:
            if use_llm:
                is_equivalent, llm_confidence = self._semantic_llm_check(
                    name1, name2, context1, context2
                )
                if is_equivalent:
                    combined_confidence = (embedding_sim + llm_confidence) / 2
                    return True, combined_confidence
            else:
                return True, embedding_sim

        name_sim = self._levenshtein_ratio(name1, name2)
        if name_sim >= self.name_match_threshold and embedding_sim >= 0.7:
            return True, (name_sim + embedding_sim) / 2

        return False, embedding_sim

    def resolve(self, new_entity: Dict[str, Any]) -> Tuple[str, bool]:
        """
        解析新实体，判断是否需要合并到现有实体

        Args:
            new_entity: {"name": str, "content": str, "embedding": List[float]}

        Returns:
            (规范名称, 是否为已有实体的别名)
        """
        new_name = new_entity.get('name', new_entity.get('content', ''))

        if not new_name:
            return new_name, False

        cached_canonical = self._get_cached_canonical(new_name)
        if cached_canonical:
            return cached_canonical, cached_canonical != new_name

        top_candidates = self._find_similar_entities(new_entity)

        for candidate_name, similarity in top_candidates:
            candidate_entity = self._get_entity_by_name(candidate_name)
            if not candidate_entity:
                continue

            is_equivalent, confidence = self.check_equivalence(
                new_entity, candidate_entity, use_llm=True
            )

            if is_equivalent and confidence >= self.embedding_threshold:
                self.dsu.union(new_name, candidate_name)
                canonical = self.dsu.get_canonical(new_name)
                self._cache_canonical(new_name, canonical)
                return canonical, True

        self.dsu.make_set(new_name)
        self._cache_canonical(new_name, new_name)
        return new_name, False

    def _get_cached_canonical(self, name: str) -> Optional[str]:
        """从缓存获取规范名称"""
        cache_key = self._get_cache_key(name)

        if self.cache_enabled:
            try:
                return self.redis_client.get(cache_key)
            except:
                pass
        else:
            return self._memory_cache.get(cache_key)

        return None

    def _cache_canonical(self, name: str, canonical: str):
        """缓存规范名称"""
        cache_key = self._get_cache_key(name)
        cache_data = json.dumps({
            "original": name,
            "canonical": canonical,
            "timestamp": datetime.now().isoformat()
        })

        if self.cache_enabled:
            try:
                self.redis_client.setex(cache_key, 86400, cache_data)
            except:
                pass
        else:
            self._memory_cache[cache_key] = cache_data

    def _find_similar_entities(self, entity: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, float]]:
        """查找相似的已有实体"""
        return []

    def _get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取实体信息"""
        return None

    def get_stats(self) -> Dict[str, Any]:
        """获取归一化统计信息"""
        unique_entities = len(set(self.dsu.find(k) for k in self.dsu.parent if k))
        total_aliases = sum(self.dsu.freq.get(self.dsu.find(k), 1)
                          for k in self.dsu.parent) - unique_entities

        return {
            "unique_entities": unique_entities,
            "total_aliases": total_aliases,
            "total_processed": len(self.dsu.parent),
            "deduplication_rate": f"{(total_aliases / (unique_entities + total_aliases) * 100):.1f}%" if unique_entities > 0 else "0%"
        }
