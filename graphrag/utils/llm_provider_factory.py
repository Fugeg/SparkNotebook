"""
LLM 提供商工厂模块
实现工业级高可用双模型降级架构

设计模式:
- 工厂模式: 统一创建不同的 LLM 提供商
- 策略模式: 不同的降级策略可插拔
- 观察者模式: 监控和告警
- 断路器模式: 防止级联故障
"""
import os
import time
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta
import threading


# 配置日志
logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """提供商状态"""
    HEALTHY = "healthy"           # 健康
    DEGRADED = "degraded"         # 降级
    UNHEALTHY = "unhealthy"       # 不健康
    CIRCUIT_OPEN = "circuit_open" # 断路器打开


class FailoverStrategy(Enum):
    """故障转移策略"""
    IMMEDIATE = "immediate"       # 立即切换
    GRADUAL = "gradual"           # 渐进切换
    PREDICTIVE = "predictive"     # 预测切换


@dataclass
class ProviderMetrics:
    """提供商性能指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    total_latency: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_latency(self) -> float:
        """平均延迟"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests + self.timeout_requests) / self.total_requests


@dataclass
class CircuitBreakerConfig:
    """断路器配置"""
    failure_threshold: int = 5           # 连续失败阈值
    recovery_timeout: int = 30           # 恢复超时(秒)
    half_open_max_calls: int = 3         # 半开状态最大测试调用数
    success_threshold: int = 2           # 恢复所需连续成功数


class CircuitBreaker:
    """
    断路器模式实现
    防止级联故障，实现自动故障恢复
    """
    
    class State(Enum):
        CLOSED = "closed"         # 关闭状态 - 正常服务
        OPEN = "open"             # 打开状态 - 拒绝服务
        HALF_OPEN = "half_open"   # 半开状态 - 测试恢复
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        self._lock = threading.RLock()
    
    def can_execute(self) -> bool:
        """检查是否可以执行"""
        with self._lock:
            if self.state == self.State.CLOSED:
                return True
            
            if self.state == self.State.OPEN:
                # 检查是否超过恢复超时
                if self.last_failure_time:
                    elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                    if elapsed >= self.config.recovery_timeout:
                        logger.info(f"[{self.name}] 断路器进入半开状态，尝试恢复")
                        self.state = self.State.HALF_OPEN
                        self.half_open_calls = 0
                        return True
                return False
            
            if self.state == self.State.HALF_OPEN:
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return True
    
    def record_success(self):
        """记录成功"""
        with self._lock:
            if self.state == self.State.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"[{self.name}] 断路器关闭，服务恢复正常")
                    self._reset()
            else:
                self.failure_count = 0
    
    def record_failure(self):
        """记录失败"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == self.State.HALF_OPEN:
                logger.warning(f"[{self.name}] 半开状态测试失败，重新打开断路器")
                self.state = self.State.OPEN
                self.half_open_calls = 0
                self.success_count = 0
            elif self.failure_count >= self.config.failure_threshold:
                logger.error(f"[{self.name}] 连续失败 {self.failure_count} 次，打开断路器")
                self.state = self.State.OPEN
    
    def _reset(self):
        """重置状态"""
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = None
    
    @property
    def current_state(self) -> str:
        return self.state.value


class LLMProvider(ABC):
    """
    LLM 提供商抽象基类
    所有具体提供商必须实现此接口
    """
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority  # 优先级，数字越小优先级越高
        self.metrics = ProviderMetrics()
        self.status = ProviderStatus.HEALTHY
        self.circuit_breaker = CircuitBreaker(name, CircuitBreakerConfig())
        self._lock = threading.RLock()
    
    @abstractmethod
    def chat(self, prompt: str, system_prompt: Optional[str] = None, 
             temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """同步对话接口"""
        pass
    
    @abstractmethod
    async def chat_async(self, prompt: str, system_prompt: Optional[str] = None,
                        temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """异步对话接口"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查提供商是否可用"""
        pass
    
    def record_request(self, latency: float, success: bool, timeout: bool = False):
        """记录请求指标"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.total_latency += latency
            self.metrics.last_request_time = datetime.now()
            
            if success:
                self.metrics.successful_requests += 1
                self.metrics.consecutive_successes += 1
                self.metrics.consecutive_failures = 0
                self.circuit_breaker.record_success()
            else:
                if timeout:
                    self.metrics.timeout_requests += 1
                else:
                    self.metrics.failed_requests += 1
                self.metrics.consecutive_failures += 1
                self.metrics.consecutive_successes = 0
                self.metrics.last_error_time = datetime.now()
                self.circuit_breaker.record_failure()
            
            # 更新状态
            self._update_status()
    
    def _update_status(self):
        """更新提供商状态"""
        if self.circuit_breaker.state == CircuitBreaker.State.OPEN:
            self.status = ProviderStatus.CIRCUIT_OPEN
        elif self.metrics.error_rate > 0.5:
            self.status = ProviderStatus.UNHEALTHY
        elif self.metrics.error_rate > 0.2:
            self.status = ProviderStatus.DEGRADED
        else:
            self.status = ProviderStatus.HEALTHY
    
    def can_serve(self) -> bool:
        """检查是否可以提供服务"""
        return self.is_available() and self.circuit_breaker.can_execute()
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        return {
            "name": self.name,
            "status": self.status.value,
            "circuit_breaker_state": self.circuit_breaker.current_state,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": f"{self.metrics.success_rate:.2%}",
                "error_rate": f"{self.metrics.error_rate:.2%}",
                "average_latency": f"{self.metrics.average_latency:.2f}s",
                "consecutive_failures": self.metrics.consecutive_failures
            }
        }


class StepFunProvider(LLMProvider):
    """StepFun 提供商实现"""
    
    def __init__(self):
        super().__init__(name="StepFun", priority=1)
        self.api_key = os.getenv('STEPFUN_API_KEY')
        self.base_url = os.getenv('STEPFUN_BASE_URL', 'https://api.stepfun.com/step_plan/v1')
        self.model = os.getenv('STEPFUN_MODEL', 'step-3.5-flash')
        self._client = None
        
        if self.api_key:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def is_available(self) -> bool:
        return self._client is not None
    
    def chat(self, prompt: str, system_prompt: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """同步对话"""
        if not self._client:
            raise RuntimeError("StepFun 客户端未初始化")
        
        start_time = time.time()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.choices[0].message.content
            latency = time.time() - start_time
            
            # 检查结果是否有效
            if not result or len(result.strip()) < 2:
                self.record_request(latency, success=False)
                raise ValueError(f"StepFun 返回无效结果: '{result}'")
            
            self.record_request(latency, success=True)
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            self.record_request(latency, success=False)
            raise e
    
    async def chat_async(self, prompt: str, system_prompt: Optional[str] = None,
                        temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """异步对话"""
        # 使用线程池执行同步调用
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.chat, prompt, system_prompt, temperature, max_tokens
        )


class QwenProvider(LLMProvider):
    """Qwen 提供商实现"""
    
    def __init__(self):
        super().__init__(name="Qwen-Plus", priority=2)
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        self.model = "qwen-plus-latest"
        
        if self.api_key:
            import dashscope
            dashscope.api_key = self.api_key
    
    def is_available(self) -> bool:
        return self.api_key is not None
    
    def chat(self, prompt: str, system_prompt: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """同步对话"""
        if not self.api_key:
            raise RuntimeError("Qwen API Key 未设置")
        
        import dashscope
        
        start_time = time.time()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = dashscope.Generation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            result = response.output.text
            latency = time.time() - start_time
            
            if not result or len(result.strip()) < 2:
                self.record_request(latency, success=False)
                raise ValueError(f"Qwen 返回无效结果: '{result}'")
            
            self.record_request(latency, success=True)
            return result
            
        except Exception as e:
            latency = time.time() - start_time
            self.record_request(latency, success=False)
            raise e
    
    async def chat_async(self, prompt: str, system_prompt: Optional[str] = None,
                        temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """异步对话"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.chat, prompt, system_prompt, temperature, max_tokens
        )


class LLMProviderFactory:
    """
    LLM 提供商工厂
    统一管理所有提供商的创建和配置
    """
    
    _providers: Dict[str, Type[LLMProvider]] = {}
    _instances: Dict[str, LLMProvider] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]):
        """注册提供商"""
        cls._providers[name] = provider_class
        logger.info(f"[Factory] 注册提供商: {name}")
    
    @classmethod
    def create(cls, name: str) -> LLMProvider:
        """创建提供商实例"""
        if name not in cls._providers:
            raise ValueError(f"未知的提供商: {name}")
        
        if name not in cls._instances:
            provider = cls._providers[name]()
            cls._instances[name] = provider
            logger.info(f"[Factory] 创建提供商实例: {name}")
        
        return cls._instances[name]
    
    @classmethod
    def get_all_providers(cls) -> List[LLMProvider]:
        """获取所有已创建的提供商"""
        return list(cls._instances.values())
    
    @classmethod
    def get_healthy_providers(cls) -> List[LLMProvider]:
        """获取健康的提供商"""
        return [p for p in cls._instances.values() if p.can_serve()]


# 注册默认提供商
LLMProviderFactory.register("stepfun", StepFunProvider)
LLMProviderFactory.register("qwen", QwenProvider)


class ResilientLLMClient:
    """
    高弹性 LLM 客户端
    实现工业级故障转移和降级策略
    """
    
    def __init__(self, 
                 primary_provider: str = "stepfun",
                 fallback_providers: List[str] = None,
                 strategy: FailoverStrategy = FailoverStrategy.IMMEDIATE,
                 timeout: float = 10.0):
        """
        初始化弹性客户端
        
        Args:
            primary_provider: 主提供商名称
            fallback_providers: 备用提供商列表
            strategy: 故障转移策略
            timeout: 请求超时时间
        """
        self.primary_name = primary_provider
        self.fallback_names = fallback_providers or ["qwen"]
        self.strategy = strategy
        self.timeout = timeout
        
        # 初始化提供商
        self.primary = LLMProviderFactory.create(primary_provider)
        self.fallbacks = [
            LLMProviderFactory.create(name) 
            for name in self.fallback_names
        ]
        
        # 监控回调
        self._monitor_callbacks: List[Callable] = []
        
        logger.info(f"[ResilientClient] 初始化完成: 主={primary_provider}, 备={fallback_providers}")
    
    def chat(self, prompt: str, system_prompt: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        同步对话（带故障转移）
        """
        providers = [self.primary] + self.fallbacks
        last_error = None
        
        for provider in providers:
            if not provider.can_serve():
                logger.warning(f"[ResilientClient] {provider.name} 不可用，跳过")
                continue
            
            try:
                logger.info(f"[ResilientClient] 尝试使用 {provider.name}")
                result = provider.chat(prompt, system_prompt, temperature, max_tokens)
                
                # 记录成功
                self._notify_monitor(provider.name, True)
                return result
                
            except Exception as e:
                logger.error(f"[ResilientClient] {provider.name} 调用失败: {e}")
                last_error = e
                self._notify_monitor(provider.name, False)
                
                # 根据策略决定是否继续
                if self.strategy == FailoverStrategy.IMMEDIATE:
                    continue
                elif self.strategy == FailoverStrategy.GRADUAL:
                    time.sleep(0.5)  # 渐进延迟
        
        # 所有提供商都失败
        raise RuntimeError(f"所有 LLM 提供商均不可用: {last_error}")
    
    async def chat_async(self, prompt: str, system_prompt: Optional[str] = None,
                        temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        异步对话（带故障转移）
        """
        providers = [self.primary] + self.fallbacks
        last_error = None
        
        for provider in providers:
            if not provider.can_serve():
                logger.warning(f"[ResilientClient] {provider.name} 不可用，跳过")
                continue
            
            try:
                logger.info(f"[ResilientClient] 尝试使用 {provider.name}")
                result = await provider.chat_async(prompt, system_prompt, temperature, max_tokens)
                
                # 记录成功
                self._notify_monitor(provider.name, True)
                return result
                
            except Exception as e:
                logger.error(f"[ResilientClient] {provider.name} 调用失败: {e}")
                last_error = e
                self._notify_monitor(provider.name, False)
                
                if self.strategy == FailoverStrategy.IMMEDIATE:
                    continue
                elif self.strategy == FailoverStrategy.GRADUAL:
                    await asyncio.sleep(0.5)
        
        raise RuntimeError(f"所有 LLM 提供商均不可用: {last_error}")
    
    def register_monitor(self, callback: Callable[[str, bool], None]):
        """注册监控回调"""
        self._monitor_callbacks.append(callback)
    
    def _notify_monitor(self, provider_name: str, success: bool):
        """通知监控器"""
        for callback in self._monitor_callbacks:
            try:
                callback(provider_name, success)
            except Exception as e:
                logger.error(f"[ResilientClient] 监控回调错误: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态报告"""
        return {
            "primary": self.primary.get_health_report(),
            "fallbacks": [p.get_health_report() for p in self.fallbacks],
            "strategy": self.strategy.value,
            "timeout": self.timeout
        }


# 全局弹性客户端实例
_resilient_client: Optional[ResilientLLMClient] = None

def get_resilient_client() -> ResilientLLMClient:
    """获取全局弹性客户端实例"""
    global _resilient_client
    if _resilient_client is None:
        _resilient_client = ResilientLLMClient(
            primary_provider="stepfun",
            fallback_providers=["qwen"],
            strategy=FailoverStrategy.IMMEDIATE,
            timeout=10.0
        )
    return _resilient_client


if __name__ == "__main__":
    # 测试
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    client = get_resilient_client()
    
    # 测试同步调用
    try:
        result = client.chat("Hello, how are you?", max_tokens=50)
        print(f"结果: {result}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 打印健康状态
    import json
    print("\n健康状态:")
    print(json.dumps(client.get_health_status(), indent=2, ensure_ascii=False))
