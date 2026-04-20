"""
Evolver Python 集成层
用于将 SparkNotebook 与 Evolver CLI 集成

Evolver 工作流程:
1. 将错误日志写入 memory/ 目录
2. 调用 evolver CLI 生成 GEP 提示词
3. 解析输出并应用到项目中
"""
import json
import subprocess
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class EvolutionEvent:
    """Evolver 事件格式"""
    timestamp: str
    event_type: str
    signal: str
    content: str
    metadata: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "signal": self.signal,
            "content": self.content,
            "metadata": self.metadata
        }


class SparkNotebookEvolver:
    """
    SparkNotebook 与 Evolver 的集成类

    功能:
    1. 将错误模式写入 memory/ 目录
    2. 调用 evolver 生成改进提示词
    3. 解析并应用进化结果
    """

    def __init__(
        self,
        evolver_path: str = "/root/fugeg/app/evolver",
        memory_dir: str = "/root/fugeg/app/evolver/memory"
    ):
        self.evolver_path = evolver_path
        self.memory_dir = memory_dir
        self.assets_dir = os.path.join(evolver_path, "assets", "gep")

    def log_error(
        self,
        agent_id: str,
        error_pattern: str,
        query: str,
        output_content: str = "",
        output_quality: float = 0.0
    ) -> str:
        """
        将错误记录写入 memory/ 目录

        Args:
            agent_id: Agent 标识
            error_pattern: 错误模式
            query: 用户查询
            output_content: 输出内容
            output_quality: 输出质量评分

        Returns:
            写入的日志文件路径
        """
        timestamp = datetime.now().isoformat()

        event = EvolutionEvent(
            timestamp=timestamp,
            event_type="error",
            signal=error_pattern,
            content=f"Query: {query}\nOutput: {output_content}\nQuality: {output_quality}",
            metadata={
                "agent_id": agent_id,
                "quality_score": output_quality
            }
        )

        log_file = os.path.join(
            self.memory_dir,
            f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(event.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"[Evolver] 错误日志已写入: {log_file}")
        return log_file

    def log_signal(
        self,
        signal_type: str,
        signal: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        将信号写入 memory/ 目录

        Args:
            signal_type: 信号类型 (info, warning, success)
            signal: 信号内容
            content: 详细描述
            metadata: 元数据

        Returns:
            写入的日志文件路径
        """
        timestamp = datetime.now().isoformat()

        event = EvolutionEvent(
            timestamp=timestamp,
            event_type=signal_type,
            signal=signal,
            content=content,
            metadata=metadata or {}
        )

        log_file = os.path.join(
            self.memory_dir,
            f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(event.to_dict(), f, ensure_ascii=False, indent=2)

        print(f"[Evolver] 信号日志已写入: {log_file}")
        return log_file

    def run_evolution(self, strategy: str = "balanced") -> Optional[Dict[str, Any]]:
        """
        运行 Evolver CLI 生成进化提示词

        Args:
            strategy: 进化策略 (balanced, innovate, harden, repair-only)

        Returns:
            进化结果字典，或 None 如果失败
        """
        env = os.environ.copy()
        env['EVOLVE_STRATEGY'] = strategy
        env['EVOLVER_REPO_ROOT'] = self.evolver_path

        try:
            result = subprocess.run(
                ['node', 'index.js'],
                cwd=self.evolver_path,
                capture_output=True,
                text=True,
                timeout=60,
                env=env
            )

            print(f"[Evolver] stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[Evolver] stderr:\n{result.stderr}")

            if result.returncode == 0:
                return self._parse_evolution_output(result.stdout)

            return None

        except subprocess.TimeoutExpired:
            print("[Evolver] 进化超时")
            return None
        except Exception as e:
            print(f"[Evolver] 进化失败: {e}")
            return None

    def run_review_mode(self) -> Optional[Dict[str, Any]]:
        """
        运行审查模式，等待人工确认

        Returns:
            进化结果字典，或 None
        """
        env = os.environ.copy()
        env['EVOLVER_REPO_ROOT'] = self.evolver_path

        try:
            result = subprocess.run(
                ['node', 'index.js', '--review'],
                cwd=self.evolver_path,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            if result.returncode == 0:
                return self._parse_evolution_output(result.stdout)

            return None

        except Exception as e:
            print(f"[Evolver] 审查模式失败: {e}")
            return None

    def _parse_evolution_output(self, output: str) -> Dict[str, Any]:
        """
        解析 Evolver 输出

        Evolver 输出格式:
        - GEP 引导的提示词
        - sessions_spawn(...) 协议
        """
        result = {
            "raw_output": output,
            "gep_prompt": None,
            "suggestions": []
        }

        lines = output.split('\n')
        gep_prompt_lines = []
        in_gep_section = False

        for line in lines:
            if '🧬' in line or 'GEP' in line.upper():
                in_gep_section = True

            if in_gep_section:
                gep_prompt_lines.append(line)

            if 'suggestion' in line.lower() or 'recommendation' in line.lower():
                result["suggestions"].append(line.strip())

        if gep_prompt_lines:
            result["gep_prompt"] = '\n'.join(gep_prompt_lines)

        return result

    def get_genes(self) -> List[Dict[str, Any]]:
        """获取现有的 Genes"""
        genes_file = os.path.join(self.assets_dir, "genes.json")

        if os.path.exists(genes_file):
            with open(genes_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else data.get("genes", [])

        return []

    def get_capsules(self) -> List[Dict[str, Any]]:
        """获取现有的 Capsules"""
        capsules_file = os.path.join(self.assets_dir, "capsules.json")

        if os.path.exists(capsules_file):
            with open(capsules_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else data.get("capsules", [])

        return []

    def get_events(self) -> List[Dict[str, Any]]:
        """获取进化事件历史"""
        events_file = os.path.join(self.assets_dir, "events.jsonl")

        events = []
        if os.path.exists(events_file):
            with open(events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            events.append(json.loads(line))
                        except:
                            pass

        return events


def create_evolver_integration() -> SparkNotebookEvolver:
    """创建默认配置的 Evolver 集成实例"""
    return SparkNotebookEvolver(
        evolver_path="/root/fugeg/app/evolver",
        memory_dir="/root/fugeg/app/evolver/memory"
    )
