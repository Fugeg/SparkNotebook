"""
EvolutionaryNode - 基于 GEP (Genome Evolution Protocol) 的自我进化节点
实现 Agent 的自动优化与自我修复能力

GEP 核心理念：
1. Gene (基因): 高效的策略、Prompt、工具链封装为可迭代胶囊
2. Evolution (进化): 通过错误反馈自动生成改进策略
3. Fitness (适应度): LLM-as-a-Judge 评估策略效果
"""
import json
import hashlib
import time
import redis
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class GeneType(Enum):
    PROMPT = "prompt_gene"
    RETRIEVAL = "retrieval_gene"
    PARSER = "parser_gene"
    TOOLCHAIN = "toolchain_gene"


@dataclass
class GeneCapsule:
    """基因胶囊 - 封装可迭代的策略单元"""
    gene_id: str
    gene_type: str
    content: Dict[str, Any]
    fitness: float
    generations: int
    parent_gene_id: Optional[str]
    created_at: str
    metadata: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({
            "gene_id": self.gene_id,
            "gene_type": self.gene_type,
            "content": self.content,
            "fitness": self.fitness,
            "generations": self.generations,
            "parent_gene_id": self.parent_gene_id,
            "created_at": self.created_at,
            "metadata": self.metadata
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'GeneCapsule':
        data = json.loads(json_str)
        return cls(**data)


@dataclass
class EvolutionLog:
    """进化日志"""
    agent_id: str
    input_hash: str
    task_type: str
    input_context: str
    output_content: str
    output_quality: float
    error_pattern: Optional[str]
    error_message: Optional[str]
    token_consumed: int
    latency_ms: int
    generations: int
    is_successful: bool
    metadata: Dict[str, Any]


class EvolutionaryNode:
    """
    基于 GEP 协议的进化节点

    功能：
    1. 捕获运行错误并记录
    2. 调用 LLM 生成改进策略 (Gene)
    3. 将 Gene 存入 Redis 缓存
    4. 支持 10-50ms 毫秒级加载

    支持的 LLM:
    - DashScope (qwen-plus)
    - StepFun (step-3.5-flash)
    """

    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6379,
        llm_model=None,
        db=None,
        logger=None,
        use_stepfun: bool = True
    ):
        self.redis_client = None
        self.cache_enabled = False

        try:
            self.redis_client = redis.StrictRedis(
                host=redis_host,
                port=redis_port,
                db=2,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self.redis_client.ping()
            self.cache_enabled = True
            print(f"[EvolutionaryNode] Redis 连接成功: {redis_host}:{redis_port}/2")
        except Exception as e:
            print(f"[EvolutionaryNode] Redis 连接失败: {e}，使用内存缓存")
            self._memory_cache = {}

        self.llm = llm_model
        self.stepfun_client = None
        self.use_stepfun = use_stepfun

        if use_stepfun and not llm_model:
            try:
                from graphrag.utils.stepfun_client import StepFunClient
                self.stepfun_client = StepFunClient()
                if self.stepfun_client.is_available():
                    print("[EvolutionaryNode] StepFun Client 初始化成功 (step-3.5-flash)")
                else:
                    print("[EvolutionaryNode] StepFun Client 不可用")
            except Exception as e:
                print(f"[EvolutionaryNode] StepFun Client 初始化失败: {e}")

        self.db = db
        self.logger = logger

        self.gene_prefix = "gep:gene:"
        self.fitness_prefix = "gep:fitness:"
        self.cache_ttl = 86400 * 7

    def _generate_input_hash(self, input_text: str) -> str:
        """生成输入指纹"""
        return hashlib.md5(input_text.encode('utf-8')).hexdigest()

    def _generate_gene_id(self, gene_type: str) -> str:
        """生成基因 ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{gene_type}_{timestamp}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

    def capture_and_evolve(
        self,
        agent_id: str,
        task_type: str,
        input_context: str,
        output_content: str,
        output_quality: float,
        error_pattern: Optional[str] = None,
        error_message: Optional[str] = None,
        token_consumed: int = 0,
        latency_ms: int = 0
    ) -> Optional[GeneCapsule]:
        """
        捕获错误并触发进化

        Args:
            agent_id: Agent 标识
            task_type: 任务类型 (retrieval/parsing/generation)
            input_context: 输入上下文
            output_content: 输出内容
            output_quality: LLM-as-a-Judge 打分
            error_pattern: 错误模式
            error_message: 错误信息

        Returns:
            进化后的 GeneCapsule 或 None
        """
        is_successful = output_quality >= 0.7 and not error_pattern

        log = EvolutionLog(
            agent_id=agent_id,
            input_hash=self._generate_input_hash(input_context),
            task_type=task_type,
            input_context=input_context,
            output_content=output_content,
            output_quality=output_quality,
            error_pattern=error_pattern,
            error_message=error_message,
            token_consumed=token_consumed,
            latency_ms=latency_ms,
            generations=0,
            is_successful=is_successful,
            metadata={}
        )

        self._log_evolution(log)

        if not is_successful and error_pattern:
            return self._evolve_strategy(error_pattern, task_type, input_context)

        return None

    def _evolve_strategy(
        self,
        error_pattern: str,
        task_type: str,
        input_context: str
    ) -> Optional[GeneCapsule]:
        """基于错误模式生成改进策略"""
        gene_type = self._map_task_to_gene_type(task_type)

        evolution_prompt = f"""基于以下失败日志，分析错误模式并生成一个改进的策略胶囊（Gene）。

错误模式: {error_pattern}
任务类型: {task_type}
上下文: {input_context[:500]}...

请基于 GEP (Genome Evolution Protocol) 协议生成一个 JSON 格式的改进策略：

{{
    "gene_id": "自动生成的基因ID",
    "gene_type": "{gene_type}",
    "strategy_name": "策略名称",
    "prompt_template": "改进后的 Prompt 模板",
    "tool_config": {{"工具配置": "值"}},
    "retrieval_config": {{"检索配置": "值"}},
    "fallback_rules": ["回退规则列表"],
    "rationale": "改进理由说明"
}}

请只输出 JSON，不要包含其他解释性文字。"""

        response = None
        try:
            if self.stepfun_client and self.stepfun_client.is_available():
                print("[Evolver] 使用 StepFun (step-3.5-flash) 生成策略...")
                response = self.stepfun_client.chat(evolution_prompt)
            elif self.llm:
                print("[Evolver] 使用 DashScope (qwen-plus) 生成策略...")
                response = self.llm.chat(evolution_prompt, username="Evolver")
            else:
                print("[Evolver] 没有可用的 LLM，无法生成策略")
                return None

            json_match = None
            import re
            clean_response = response.strip()
            if clean_response.startswith('```'):
                clean_response = re.sub(r'^```(?:json)?\s*', '', clean_response)
                clean_response = re.sub(r'\s*```$', '', clean_response)
            json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)

            if json_match:
                gene_content = json.loads(json_match.group())
                gene_id = self._generate_gene_id(gene_type)

                gene = GeneCapsule(
                    gene_id=gene_id,
                    gene_type=gene_type,
                    content=gene_content,
                    fitness=0.5,
                    generations=1,
                    parent_gene_id=None,
                    created_at=datetime.now().isoformat(),
                    metadata={"error_pattern": error_pattern}
                )

                self._cache_gene(gene)
                return gene

        except Exception as e:
            print(f"[EvolutionaryNode] 进化失败: {e}")

        return None

    def _map_task_to_gene_type(self, task_type: str) -> str:
        """任务类型映射到基因类型"""
        mapping = {
            "retrieval": GeneType.RETRIEVAL.value,
            "extraction": GeneType.PARSER.value,
            "generation": GeneType.PROMPT.value,
            "github_mcp": GeneType.TOOLCHAIN.value,
        }
        return mapping.get(task_type, GeneType.PROMPT.value)

    def _log_evolution(self, log: EvolutionLog):
        """记录进化日志到数据库"""
        if not self.db:
            return

        try:
            self.db.log_evolution(
                agent_id=log.agent_id,
                input_hash=log.input_hash,
                task_type=log.task_type,
                input_context=log.input_context,
                output_content=log.output_content,
                output_quality=log.output_quality,
                error_pattern=log.error_pattern,
                error_message=log.error_message,
                token_consumed=log.token_consumed,
                latency_ms=log.latency_ms,
                generations=log.generations,
                is_successful=log.is_successful
            )
        except Exception as e:
            print(f"[EvolutionaryNode] 记录日志失败: {e}")

    def _cache_gene(self, gene: GeneCapsule):
        """缓存基因胶囊到 Redis"""
        cache_key = f"{self.gene_prefix}{gene.gene_id}"

        gene_data = {
            "gene": gene.to_json(),
            "fitness": gene.fitness,
            "generations": gene.generations
        }

        if self.cache_enabled:
            try:
                self.redis_client.hset(cache_key, mapping={
                    "gene": gene.to_json(),
                    "fitness": str(gene.fitness),
                    "generations": str(gene.generations),
                    "created_at": gene.created_at
                })
                self.redis_client.expire(cache_key, self.cache_ttl)

                fitness_key = f"{self.fitness_prefix}{gene.gene_type}"
                self.redis_client.zadd(fitness_key, {gene.gene_id: gene.fitness})

                print(f"[EvolutionaryNode] 基因已缓存: {gene.gene_id}")
            except Exception as e:
                print(f"[EvolutionaryNode] 缓存失败: {e}")
                self._memory_cache[cache_key] = gene_data
        else:
            self._memory_cache[cache_key] = gene_data

    def load_gene(self, gene_id: str) -> Optional[GeneCapsule]:
        """
        加载基因胶囊 (10-50ms 毫秒级加载)

        Redis Key 结构:
        gep:gene:{gene_id} -> Hash {
            gene: GeneCapsule JSON,
            fitness: float,
            generations: int,
            created_at: timestamp
        }
        """
        cache_key = f"{self.gene_prefix}{gene_id}"

        if self.cache_enabled:
            try:
                start_time = time.time()
                gene_data = self.redis_client.hgetall(cache_key)
                load_time = (time.time() - start_time) * 1000

                if gene_data and 'gene' in gene_data:
                    print(f"[EvolutionaryNode] 基因加载耗时: {load_time:.2f}ms")
                    return GeneCapsule.from_json(gene_data['gene'])
            except Exception as e:
                print(f"[EvolutionaryNode] 加载失败: {e}")
        else:
            if cache_key in self._memory_cache:
                return GeneCapsule.from_json(self._memory_cache[cache_key]['gene'])

        return None

    def get_best_gene(self, gene_type: str) -> Optional[GeneCapsule]:
        """
        获取最佳适应度的基因

        Redis Sorted Set:
        gep:fitness:{gene_type} -> Sorted Set {gene_id: fitness_score}
        """
        fitness_key = f"{self.fitness_prefix}{gene_type}"

        if self.cache_enabled:
            try:
                best_gene_id = self.redis_client.zrevrange(fitness_key, 0, 0)
                if best_gene_id:
                    return self.load_gene(best_gene_id[0])
            except Exception as e:
                print(f"[EvolutionaryNode] 获取最佳基因失败: {e}")

        return None

    def update_fitness(self, gene_id: str, new_fitness: float):
        """更新基因适应度"""
        cache_key = f"{self.gene_prefix}{gene_id}"

        if self.cache_enabled:
            try:
                self.redis_client.hset(cache_key, "fitness", str(new_fitness))

                gene_data = self.redis_client.hgetall(cache_key)
                if gene_data and 'gene' in gene_data:
                    gene = GeneCapsule.from_json(gene_data['gene'])
                    gene_type = gene.gene_type
                    fitness_key = f"{self.fitness_prefix}{gene_type}"
                    self.redis_client.zadd(fitness_key, {gene_id: new_fitness})
            except Exception as e:
                print(f"[EvolutionaryNode] 更新适应度失败: {e}")


class RedisGEPKeyStructure:
    """
    Redis GEP Key 结构设计

    目标: 10-50ms 毫秒级加载

    Key 结构:
    ┌─────────────────────────────────────────────────────────────┐
    │ 命名空间层级                                                │
    ├─────────────────────────────────────────────────────────────┤
    │ gep:                      # GEP 协议前缀                    │
    │   gene:{gene_id}:         # 单个基因胶囊                    │
    │   fitness:{gene_type}:    # 按类型分类的适应度排行榜        │
    │   lineage:{root_gene_id}: # 基因族谱链                     │
    │   stats:                  # 全局统计                       │
    └─────────────────────────────────────────────────────────────┘

    数据结构:
    ┌─────────────────────────────────────────────────────────────┐
    │ gep:gene:{gene_id} -> Hash                                 │
    │   gene: GeneCapsule JSON                                    │
    │   fitness: 0.85                                            │
    │   generations: 3                                            │
    │   created_at: 2026-04-16T10:30:00                          │
    │   parent: parent_gene_id (可选)                            │
    └─────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────┐
    │ gep:fitness:{gene_type} -> Sorted Set                      │
    │   gene_id_1: 0.95                                          │
    │   gene_id_2: 0.87                                          │
    │   gene_id_3: 0.76                                          │
    │   ...                                                       │
    └─────────────────────────────────────────────────────────────┘
    """

    @staticmethod
    def gene_key(gene_id: str) -> str:
        """单个基因 Key"""
        return f"gep:gene:{gene_id}"

    @staticmethod
    def fitness_key(gene_type: str) -> str:
        """适应度排行榜 Key"""
        return f"gep:fitness:{gene_type}"

    @staticmethod
    def lineage_key(root_gene_id: str) -> str:
        """基因族谱 Key"""
        return f"gep:lineage:{root_gene_id}"

    @staticmethod
    def stats_key() -> str:
        """全局统计 Key"""
        return "gep:stats"

    @staticmethod
    def gene_type_pattern() -> str:
        """基因类型通配模式"""
        return "gep:gene:*"

    @staticmethod
    def all_fitness_pattern() -> str:
        """所有适应度 Key 通配模式"""
        return "gep:fitness:*"
