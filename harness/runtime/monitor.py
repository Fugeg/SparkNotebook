"""
Agent 运行时监控器
追踪耗时、Token 消耗、成功率等关键指标
"""

import json
import time
import functools
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class AgentExecutionRecord:
    """Agent 执行记录"""
    request_id: str
    timestamp: str
    latency_ms: float          # 执行耗时（毫秒）
    input_tokens: int          # 输入 Token 数
    output_tokens: int         # 输出 Token 数
    total_tokens: int          # 总 Token 数
    success: bool              # 是否成功
    error_type: Optional[str]  # 错误类型
    prompt_version: str        # 使用的 Prompt 版本
    model_name: str            # 使用的模型


class AgentMonitor:
    """Agent 运行时监控器"""
    
    def __init__(self, 
                 log_dir: str = "logs/harness",
                 max_latency_ms: float = 30000,
                 max_token_per_request: int = 4000,
                 error_rate_threshold: float = 0.05):
        """
        初始化监控器
        
        Args:
            log_dir: 日志目录
            max_latency_ms: 最大延迟阈值
            max_token_per_request: 单次最大 Token 阈值
            error_rate_threshold: 错误率阈值
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_latency_ms = max_latency_ms
        self.max_token_per_request = max_token_per_request
        self.error_rate_threshold = error_rate_threshold
        
        # 内存中的统计数据
        self.records: list = []
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "error_counts": defaultdict(int)
        }
        
        # 当前日志文件
        self.current_log_file = self.log_dir / f"agent_{datetime.now().strftime('%Y%m%d')}.jsonl"
    
    def record(self, 
               request_id: str,
               latency_ms: float,
               input_tokens: int = 0,
               output_tokens: int = 0,
               success: bool = True,
               error_type: Optional[str] = None,
               prompt_version: str = "v1",
               model_name: str = "unknown") -> AgentExecutionRecord:
        """
        记录一次执行
        
        Args:
            request_id: 请求 ID
            latency_ms: 执行耗时
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
            success: 是否成功
            error_type: 错误类型
            prompt_version: Prompt 版本
            model_name: 模型名称
        
        Returns:
            执行记录
        """
        record = AgentExecutionRecord(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            success=success,
            error_type=error_type,
            prompt_version=prompt_version,
            model_name=model_name
        )
        
        # 保存到内存
        self.records.append(record)
        
        # 更新统计
        self.stats["total_requests"] += 1
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
            if error_type:
                self.stats["error_counts"][error_type] += 1
        
        self.stats["total_tokens"] += record.total_tokens
        self.stats["total_latency_ms"] += latency_ms
        
        # 保存到文件
        self._append_to_log(record)
        
        # 检查告警
        self._check_alerts(record)
        
        return record
    
    def _append_to_log(self, record: AgentExecutionRecord):
        """追加记录到日志文件"""
        with open(self.current_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + '\n')
    
    def _check_alerts(self, record: AgentExecutionRecord):
        """检查是否需要告警"""
        alerts = []
        
        # 延迟告警
        if record.latency_ms > self.max_latency_ms:
            alerts.append(f"⚠️ 延迟过高: {record.latency_ms:.0f}ms > {self.max_latency_ms}ms")
        
        # Token 告警
        if record.total_tokens > self.max_token_per_request:
            alerts.append(f"⚠️ Token 消耗过高: {record.total_tokens} > {self.max_token_per_request}")
        
        # 错误率告警
        if self.stats["total_requests"] > 10:
            error_rate = self.stats["failed_requests"] / self.stats["total_requests"]
            if error_rate > self.error_rate_threshold:
                alerts.append(f"⚠️ 错误率过高: {error_rate*100:.1f}% > {self.error_rate_threshold*100}%")
        
        if alerts:
            for alert in alerts:
                print(f"  {alert}")
    
    def get_stats(self, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        获取统计信息
        
        Args:
            window_minutes: 时间窗口（分钟），None 表示全部
        
        Returns:
            统计信息字典
        """
        records = self.records
        
        if window_minutes:
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            records = [
                r for r in records 
                if datetime.fromisoformat(r.timestamp) > cutoff
            ]
        
        if not records:
            return {"message": "暂无数据"}
        
        total = len(records)
        successful = sum(1 for r in records if r.success)
        failed = total - successful
        
        latencies = [r.latency_ms for r in records]
        tokens = [r.total_tokens for r in records]
        
        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "success_rate": round(successful / total * 100, 2),
            "error_rate": round(failed / total * 100, 2),
            "latency": {
                "avg_ms": round(sum(latencies) / len(latencies), 2),
                "min_ms": round(min(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "p95_ms": round(sorted(latencies)[int(len(latencies)*0.95)], 2) if len(latencies) > 20 else None
            },
            "tokens": {
                "total": sum(tokens),
                "avg_per_request": round(sum(tokens) / len(tokens), 2),
                "max": max(tokens)
            },
            "error_breakdown": dict(self.stats["error_counts"])
        }
    
    def print_stats(self, window_minutes: Optional[int] = None):
        """打印统计信息"""
        stats = self.get_stats(window_minutes)
        
        if "message" in stats:
            print(f"\n{stats['message']}")
            return
        
        window_str = f" (最近 {window_minutes} 分钟)" if window_minutes else ""
        
        print(f"\n{'='*60}")
        print(f"📊 Agent 运行统计{window_str}")
        print(f"{'='*60}")
        print(f"总请求数: {stats['total_requests']}")
        print(f"成功: {stats['successful_requests']} ✅")
        print(f"失败: {stats['failed_requests']} ❌")
        print(f"成功率: {stats['success_rate']}%")
        print(f"{'-'*60}")
        print(f"延迟统计:")
        print(f"  平均: {stats['latency']['avg_ms']:.0f}ms")
        print(f"  最小: {stats['latency']['min_ms']:.0f}ms")
        print(f"  最大: {stats['latency']['max_ms']:.0f}ms")
        if stats['latency']['p95_ms']:
            print(f"  P95:  {stats['latency']['p95_ms']:.0f}ms")
        print(f"{'-'*60}")
        print(f"Token 统计:")
        print(f"  总计: {stats['tokens']['total']}")
        print(f"  平均: {stats['tokens']['avg_per_request']:.0f}/请求")
        print(f"  最大: {stats['tokens']['max']}")
        if stats['error_breakdown']:
            print(f"{'-'*60}")
            print(f"错误分布:")
            for error, count in stats['error_breakdown'].items():
                print(f"  {error}: {count}")
        print(f"{'='*60}\n")
    
    def decorator(self, func: Callable) -> Callable:
        """
        装饰器：自动追踪函数执行
        
        使用示例:
            @monitor.decorator
            def my_agent(input_text):
                # ...
                return result
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            start_time = time.time()
            
            success = True
            error_type = None
            result = None
            
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                success = False
                error_type = type(e).__name__
                raise
            finally:
                latency_ms = (time.time() - start_time) * 1000
                
                # 尝试从结果中提取 Token 信息
                input_tokens = 0
                output_tokens = 0
                if isinstance(result, dict):
                    input_tokens = result.get("input_tokens", 0)
                    output_tokens = result.get("output_tokens", 0)
                
                self.record(
                    request_id=request_id,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=success,
                    error_type=error_type
                )
            
            return result
        
        return wrapper


# 全局监控器实例
_monitor: Optional[AgentMonitor] = None


def get_monitor(log_dir: str = "logs/harness") -> AgentMonitor:
    """获取全局监控器实例"""
    global _monitor
    if _monitor is None:
        _monitor = AgentMonitor(log_dir=log_dir)
    return _monitor
