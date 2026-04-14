"""
Prometheus 指标导出器
将 Agent 监控指标暴露给 Prometheus
"""

import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

# 尝试导入 prometheus_client
# 如果未安装，提供降级方案
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("⚠️  prometheus_client 未安装，使用文件导出模式")
    print("   安装命令: pip install prometheus-client")


class MetricsExporter:
    """Prometheus 指标导出器"""
    
    def __init__(self, 
                 port: int = 8000,
                 namespace: str = "harness",
                 use_file_export: bool = False):
        """
        初始化指标导出器
        
        Args:
            port: Prometheus HTTP 服务端口
            namespace: 指标命名空间
            use_file_export: 强制使用文件导出模式（即使安装了 prometheus_client）
        """
        self.port = port
        self.namespace = namespace
        self.use_file_export = use_file_export or not PROMETHEUS_AVAILABLE
        
        # 文件导出模式配置
        self.metrics_file = Path("logs/harness/metrics.prom")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 内存中的指标数据（文件模式使用）
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = {}
        
        if not self.use_file_export:
            self._init_prometheus_metrics()
        
        self._running = False
        self._server_thread: Optional[threading.Thread] = None
    
    def _init_prometheus_metrics(self):
        """初始化 Prometheus 指标"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # 创建独立的注册表
        self.registry = CollectorRegistry()
        
        # 请求计数器
        self.request_counter = Counter(
            'agent_requests_total',
            'Total number of agent requests',
            ['prompt_version', 'model_name', 'status'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # Token 消耗计数器
        self.token_counter = Counter(
            'agent_tokens_total',
            'Total number of tokens consumed',
            ['type', 'prompt_version'],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 延迟直方图
        self.latency_histogram = Histogram(
            'agent_request_duration_seconds',
            'Request duration in seconds',
            ['prompt_version', 'model_name'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 当前活跃请求数
        self.active_requests = Gauge(
            'agent_active_requests',
            'Number of active requests',
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 错误率
        self.error_rate = Gauge(
            'agent_error_rate',
            'Current error rate (0-1)',
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 平均延迟
        self.avg_latency = Gauge(
            'agent_avg_latency_seconds',
            'Average request latency',
            namespace=self.namespace,
            registry=self.registry
        )
        
        # 应用信息
        self.app_info = Info(
            'app',
            'Application information',
            namespace=self.namespace,
            registry=self.registry
        )
        self.app_info.info({'version': '0.1.0', 'name': 'SparkNotebook'})
    
    def start(self):
        """启动指标导出服务"""
        if self._running:
            return
        
        if self.use_file_export:
            print(f"📊 指标文件导出模式: {self.metrics_file}")
            self._running = True
            # 定期写入文件
            self._start_file_writer()
        else:
            try:
                start_http_server(self.port, registry=self.registry)
                print(f"📊 Prometheus 指标服务已启动: http://localhost:{self.port}/metrics")
                self._running = True
            except Exception as e:
                print(f"❌ 启动 Prometheus 服务失败: {e}")
                print("   切换到文件导出模式")
                self.use_file_export = True
                self.start()
    
    def stop(self):
        """停止指标导出服务"""
        self._running = False
        if self._server_thread:
            self._server_thread.join(timeout=5)
    
    def _start_file_writer(self):
        """启动文件写入线程"""
        def writer():
            while self._running:
                self._write_metrics_file()
                time.sleep(15)  # 每 15 秒写入一次
        
        self._server_thread = threading.Thread(target=writer, daemon=True)
        self._server_thread.start()
    
    def _write_metrics_file(self):
        """将指标写入文件（Prometheus text format）"""
        lines = []
        timestamp = int(time.time() * 1000)
        
        # 计数器
        for name, value in self._counters.items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value} {timestamp}")
        
        # 仪表盘
        for name, value in self._gauges.items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value} {timestamp}")
        
        # 直方图
        for name, values in self._histograms.items():
            if values:
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_sum {sum(values)} {timestamp}")
                lines.append(f"{name}_count {len(values)} {timestamp}")
        
        # 写入文件
        with open(self.metrics_file, 'w') as f:
            f.write('\n'.join(lines))
    
    def record_request(self, 
                       prompt_version: str,
                       model_name: str,
                       latency_seconds: float,
                       input_tokens: int,
                       output_tokens: int,
                       success: bool):
        """
        记录一次请求
        
        Args:
            prompt_version: Prompt 版本
            model_name: 模型名称
            latency_seconds: 延迟（秒）
            input_tokens: 输入 Token 数
            output_tokens: 输出 Token 数
            success: 是否成功
        """
        status = "success" if success else "error"
        
        if self.use_file_export:
            # 文件模式
            counter_key = f'{self.namespace}_agent_requests_total{{prompt_version="{prompt_version}",model_name="{model_name}",status="{status}"}}'
            self._counters[counter_key] = self._counters.get(counter_key, 0) + 1
            
            # Token 计数
            input_key = f'{self.namespace}_agent_tokens_total{{type="input",prompt_version="{prompt_version}"}}'
            output_key = f'{self.namespace}_agent_tokens_total{{type="output",prompt_version="{prompt_version}"}}'
            self._counters[input_key] = self._counters.get(input_key, 0) + input_tokens
            self._counters[output_key] = self._counters.get(output_key, 0) + output_tokens
            
            # 延迟直方图
            hist_key = f'{self.namespace}_agent_request_duration_seconds{{prompt_version="{prompt_version}",model_name="{model_name}"}}'
            if hist_key not in self._histograms:
                self._histograms[hist_key] = []
            self._histograms[hist_key].append(latency_seconds)
            
            # 更新仪表盘
            self._update_gauges()
        else:
            # Prometheus 模式
            self.request_counter.labels(
                prompt_version=prompt_version,
                model_name=model_name,
                status=status
            ).inc()
            
            self.token_counter.labels(
                type="input",
                prompt_version=prompt_version
            ).inc(input_tokens)
            
            self.token_counter.labels(
                type="output",
                prompt_version=prompt_version
            ).inc(output_tokens)
            
            self.latency_histogram.labels(
                prompt_version=prompt_version,
                model_name=model_name
            ).observe(latency_seconds)
    
    def _update_gauges(self):
        """更新仪表盘指标（文件模式）"""
        # 计算错误率
        total = sum(1 for k in self._counters if 'status="success"' in k or 'status="error"' in k)
        errors = sum(1 for k in self._counters if 'status="error"' in k)
        if total > 0:
            self._gauges[f'{self.namespace}_agent_error_rate'] = errors / total
        
        # 计算平均延迟
        all_latencies = []
        for values in self._histograms.values():
            all_latencies.extend(values)
        if all_latencies:
            self._gauges[f'{self.namespace}_agent_avg_latency_seconds'] = sum(all_latencies) / len(all_latencies)
    
    def set_active_requests(self, count: int):
        """设置活跃请求数"""
        if self.use_file_export:
            self._gauges[f'{self.namespace}_agent_active_requests'] = count
        else:
            self.active_requests.set(count)
    
    def get_metrics_text(self) -> str:
        """获取指标文本（用于调试）"""
        if self.use_file_export:
            if self.metrics_file.exists():
                return self.metrics_file.read_text()
            return "No metrics file yet"
        else:
            from prometheus_client import generate_latest
            return generate_latest(self.registry).decode('utf-8')


# 全局实例
_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter(port: int = 8000, **kwargs) -> MetricsExporter:
    """获取全局指标导出器实例"""
    global _exporter
    if _exporter is None:
        _exporter = MetricsExporter(port=port, **kwargs)
    return _exporter
