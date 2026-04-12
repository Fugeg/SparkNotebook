"""
Redis 语义缓存工具类
用于缓存 LLM 查询结果，通过向量相似度匹配避免重复调用
"""
import redis
import json
import numpy as np
import hashlib
from typing import Optional, List, Dict, Any
from datetime import datetime


class SemanticCache:
    """
    语义缓存类 - 基于向量相似度的智能缓存
    
    核心功能：
    1. 存储查询向量、原始查询文本和响应结果
    2. 基于余弦相似度匹配历史查询
    3. 支持缓存过期时间设置
    4. 提供缓存统计信息
    """
    
    def __init__(self, host='localhost', port=6379, db=0, password=None, 
                 expire_time=3600*24, similarity_threshold=0.95):
        """
        初始化 Redis 连接
        
        Args:
            host: Redis 服务器地址
            port: Redis 端口
            db: 数据库编号
            password: 密码（如果有）
            expire_time: 缓存过期时间（秒），默认24小时
            similarity_threshold: 相似度阈值，默认0.95
        """
        try:
            self.client = redis.StrictRedis(
                host=host, 
                port=port, 
                db=db, 
                password=password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # 测试连接
            self.client.ping()
            self.enabled = True
            print(f"[Redis] 缓存服务已连接: {host}:{port}/{db}")
        except Exception as e:
            print(f"[Redis] 连接失败: {e}，缓存功能已禁用")
            self.enabled = False
            self.client = None
        
        self.expire_time = expire_time
        self.similarity_threshold = similarity_threshold
        self.cache_prefix = "semantic_cache:"
        self.stats_prefix = "cache_stats:"
    
    def _get_cache_key(self, query: str) -> str:
        """生成缓存键"""
        # 使用查询文本的 MD5 哈希作为键的一部分
        hash_obj = hashlib.md5(query.encode('utf-8'))
        return f"{self.cache_prefix}{hash_obj.hexdigest()}"
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception as e:
            print(f"[Redis] 相似度计算错误: {e}")
            return 0.0
    
    def get(self, query: str, query_vector: List[float]) -> Optional[str]:
        """
        尝试从缓存获取响应
        
        Args:
            query: 查询文本
            query_vector: 查询向量
            
        Returns:
            缓存的响应或 None
        """
        if not self.enabled:
            return None
        
        try:
            # 首先尝试精确匹配
            cache_key = self._get_cache_key(query)
            cached_data = self.client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                print(f"[Redis] 精确缓存命中: {query[:30]}...")
                self._update_stats(hit=True)
                return data.get('response')
            
            # 精确未命中，尝试语义匹配
            return self._semantic_search(query_vector)
            
        except Exception as e:
            print(f"[Redis] 缓存查询错误: {e}")
            return None
    
    def _semantic_search(self, query_vector: List[float]) -> Optional[str]:
        """
        语义搜索 - 查找相似的历史查询
        
        Args:
            query_vector: 查询向量
            
        Returns:
            最相似的缓存响应或 None
        """
        try:
            # 获取所有缓存键
            pattern = f"{self.cache_prefix}*"
            keys = self.client.scan_iter(match=pattern, count=100)
            
            query_vec = np.array(query_vector)
            best_match = None
            best_similarity = 0.0
            
            for key in keys:
                try:
                    cached_data = self.client.get(key)
                    if not cached_data:
                        continue
                    
                    data = json.loads(cached_data)
                    cached_vec = np.array(data.get('vector', []))
                    
                    if len(cached_vec) == 0:
                        continue
                    
                    # 计算相似度
                    similarity = self._cosine_similarity(query_vec, cached_vec)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = data
                        
                except Exception as e:
                    continue
            
            # 检查最佳匹配是否超过阈值
            if best_match and best_similarity >= self.similarity_threshold:
                print(f"[Redis] 语义缓存命中 (相似度: {best_similarity:.4f}): {best_match.get('query', '')[:30]}...")
                self._update_stats(hit=True, semantic=True)
                return best_match.get('response')
            
            self._update_stats(hit=False)
            return None
            
        except Exception as e:
            print(f"[Redis] 语义搜索错误: {e}")
            return None
    
    def set(self, query: str, query_vector: List[float], response: str, 
            metadata: Optional[Dict[str, Any]] = None):
        """
        存储查询结果到缓存
        
        Args:
            query: 查询文本
            query_vector: 查询向量
            response: 响应内容
            metadata: 额外元数据（可选）
        """
        if not self.enabled:
            return
        
        try:
            cache_key = self._get_cache_key(query)
            
            cache_data = {
                "query": query,
                "vector": query_vector,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # 存储到 Redis，设置过期时间
            self.client.setex(
                cache_key, 
                self.expire_time, 
                json.dumps(cache_data, ensure_ascii=False)
            )
            
            print(f"[Redis] 缓存已存储: {query[:30]}... (过期时间: {self.expire_time}s)")
            
        except Exception as e:
            print(f"[Redis] 缓存存储错误: {e}")
    
    def _update_stats(self, hit: bool, semantic: bool = False):
        """更新缓存统计信息"""
        try:
            stats_key = f"{self.stats_prefix}total"
            stats = self.client.get(stats_key)
            
            if stats:
                stats_data = json.loads(stats)
            else:
                stats_data = {"total": 0, "hits": 0, "semantic_hits": 0, "misses": 0}
            
            stats_data["total"] += 1
            if hit:
                stats_data["hits"] += 1
                if semantic:
                    stats_data["semantic_hits"] += 1
            else:
                stats_data["misses"] += 1
            
            self.client.setex(stats_key, 86400, json.dumps(stats_data))
            
        except Exception:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            stats_key = f"{self.stats_prefix}total"
            stats = self.client.get(stats_key)
            
            if stats:
                stats_data = json.loads(stats)
            else:
                stats_data = {"total": 0, "hits": 0, "semantic_hits": 0, "misses": 0}
            
            # 计算命中率
            total = stats_data.get("total", 0)
            hits = stats_data.get("hits", 0)
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            return {
                "enabled": True,
                "total_queries": total,
                "cache_hits": hits,
                "semantic_hits": stats_data.get("semantic_hits", 0),
                "cache_misses": stats_data.get("misses", 0),
                "hit_rate": f"{hit_rate:.2f}%",
                "expire_time": self.expire_time,
                "similarity_threshold": self.similarity_threshold
            }
            
        except Exception as e:
            return {"enabled": True, "error": str(e)}
    
    def clear(self):
        """清空所有缓存"""
        if not self.enabled:
            return
        
        try:
            pattern = f"{self.cache_prefix}*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                self.client.delete(*keys)
                print(f"[Redis] 已清空 {len(keys)} 个缓存项")
            else:
                print("[Redis] 缓存为空")
                
        except Exception as e:
            print(f"[Redis] 清空缓存错误: {e}")
    
    def delete(self, query: str):
        """删除特定查询的缓存"""
        if not self.enabled:
            return
        
        try:
            cache_key = self._get_cache_key(query)
            self.client.delete(cache_key)
            print(f"[Redis] 已删除缓存: {query[:30]}...")
        except Exception as e:
            print(f"[Redis] 删除缓存错误: {e}")
