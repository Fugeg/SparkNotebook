#!/usr/bin/env python3
"""
演示 MetricsExporter 的使用
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.runtime.metrics_exporter import get_metrics_exporter


def simulate_agent_requests(exporter, num_requests=20):
    """模拟 Agent 请求并记录指标"""
    print(f"\n模拟 {num_requests} 次 Agent 请求...")
    
    models = ["qwen-plus", "stepfun-3.5-flash"]
    versions = ["v1", "v2"]
    
    for i in range(num_requests):
        model = random.choice(models)
        version = random.choice(versions)
        
        # 模拟延迟 (0.5s - 5s)
        latency = random.uniform(0.5, 5.0)
        
        # 模拟 Token 消耗
        input_tokens = random.randint(500, 1500)
        output_tokens = random.randint(200, 800)
        
        # 90% 成功率
        success = random.random() > 0.1
        
        # 记录指标
        exporter.record_request(
            prompt_version=version,
            model_name=model,
            latency_seconds=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success
        )
        
        print(f"  [{i+1}/{num_requests}] {model} ({version}): "
              f"{latency:.2f}s, {input_tokens}+{output_tokens} tokens, "
              f"{'✅' if success else '❌'}")
        
        # 模拟请求间隔
        time.sleep(0.1)


def main():
    print("="*60)
    print("📊 MetricsExporter 演示")
    print("="*60)
    
    # 获取导出器
    exporter = get_metrics_exporter(port=8000)
    
    # 启动服务
    print("\n启动 Prometheus 指标服务...")
    exporter.start()
    
    print("\n" + "="*60)
    print("指标端点: http://localhost:8000/metrics")
    print("="*60)
    
    # 模拟请求
    simulate_agent_requests(exporter, num_requests=20)
    
    # 显示指标
    print("\n" + "="*60)
    print("当前指标预览（前 30 行）:")
    print("="*60)
    metrics_text = exporter.get_metrics_text()
    lines = metrics_text.split('\n')
    for line in lines[:30]:
        if line.strip():
            print(line)
    if len(lines) > 30:
        print(f"... ({len(lines) - 30} more lines)")
    
    print("\n" + "="*60)
    print("✅ 演示完成！")
    print("="*60)
    print("\n你可以:")
    print("  1. 访问 http://localhost:8000/metrics 查看完整指标")
    print("  2. 启动 Prometheus: docker-compose -f docker-compose.monitoring.yml up -d")
    print("  3. 访问 Grafana: http://localhost:3000 (admin/admin123)")
    
    # 保持运行
    print("\n按 Ctrl+C 停止...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n停止服务...")
        exporter.stop()


if __name__ == "__main__":
    main()
