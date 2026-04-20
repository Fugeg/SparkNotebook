"""
Evolver 模块 - GEP (Genome Evolution Protocol) 自我进化系统

包含两个实现:
1. EvolutionaryNode - 纯 Python 实现 (当前使用)
2. SparkNotebookEvolver - 与官方 Evolver CLI 集成
"""
from graphrag.evolver.evolutionary_node import (
    EvolutionaryNode,
    GeneCapsule,
    GeneType,
    RedisGEPKeyStructure,
    EvolutionLog
)
from graphrag.evolver.sparknotebook_evolver import (
    SparkNotebookEvolver,
    create_evolver_integration,
    EvolutionEvent
)

__all__ = [
    'EvolutionaryNode',
    'GeneCapsule',
    'GeneType',
    'RedisGEPKeyStructure',
    'EvolutionLog',
    'SparkNotebookEvolver',
    'create_evolver_integration',
    'EvolutionEvent'
]
