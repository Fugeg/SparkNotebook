#!/usr/bin/env python3
"""
Harness Engineering MVP 演示脚本
展示核心功能的使用方法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.deployment.prompt_version import PromptVersionManager, get_prompt_manager
from harness.runtime.monitor import AgentMonitor, get_monitor
from harness.runtime.checkpointer import AgentStateManager
from harness.evaluation.golden_dataset import GoldenDataset


def demo_prompt_version():
    """演示 Prompt 版本管理"""
    print("\n" + "="*60)
    print("📝 Demo 1: Prompt 版本管理")
    print("="*60)
    
    # 初始化版本管理器
    manager = get_prompt_manager()
    
    # 显示当前状态
    manager.print_status()
    
    # 创建示例 Prompt 文件
    os.makedirs("prompts", exist_ok=True)
    
    # v1 基线版本
    with open("prompts/v1_analysis.txt", "w") as f:
        f.write("""你是一个 GitHub 项目分析助手。
请分析给定的开源项目，提供以下信息：
1. 项目基本信息（Stars, Forks）
2. 社区活跃度
3. 最近更新情况

请用简洁的表格形式输出。""")
    
    # v2 改进版本
    with open("prompts/v2_analysis.txt", "w") as f:
        f.write("""你是一位资深的开源项目分析师。
请深入分析给定的 GitHub 项目：

## 分析维度
1. **项目健康度**: Stars/Forks 趋势、Issue 响应速度
2. **社区活跃度**: 贡献者数量、PR 合并率
3. **技术栈分析**: 主要编程语言、依赖关系
4. **竞争对比**: 与同类项目的横向比较

## 输出格式
使用 Markdown 表格，包含具体数据和百分比。""")
    
    # 重新加载
    manager._load_prompts()
    
    # 获取当前 Prompt
    print("\n当前 Prompt (v1):")
    print("-" * 40)
    print(manager.get_prompt("v1")[:200] + "...")
    
    # 切换到 v2
    print("\n切换到 v2...")
    manager.switch_version("v2")
    
    print("\n当前 Prompt (v2):")
    print("-" * 40)
    print(manager.get_prompt("v2")[:200] + "...")
    
    # 切换回 v1
    manager.switch_version("v1")


def demo_monitor():
    """演示监控功能"""
    print("\n" + "="*60)
    print("📊 Demo 2: 运行时监控")
    print("="*60)
    
    # 获取监控器
    monitor = get_monitor()
    
    # 模拟一些执行记录
    import random
    
    print("\n模拟 10 次 Agent 执行...")
    for i in range(10):
        latency = random.uniform(1000, 5000)  # 1-5秒
        input_tokens = random.randint(500, 1500)
        output_tokens = random.randint(200, 800)
        success = random.random() > 0.1  # 90% 成功率
        
        monitor.record(
            request_id=f"req_{i:03d}",
            latency_ms=latency,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            error_type="APIError" if not success else None,
            prompt_version="v1",
            model_name="qwen-plus"
        )
    
    # 打印统计
    monitor.print_stats()


def demo_checkpointer():
    """演示 Checkpointer 功能"""
    print("\n" + "="*60)
    print("💾 Demo 3: 状态检查点")
    print("="*60)
    
    # 创建状态管理器
    state_manager = AgentStateManager()
    
    # 模拟一个多步骤任务
    task_id = "analysis_task_001"
    
    print(f"\n模拟任务: {task_id}")
    print("步骤 1: 获取 GitHub 数据...")
    state_manager.checkpoint(task_id, 1, {
        "repos": ["react", "vue"],
        "data_fetched": True,
        "github_data": {"react": {"stars": 220000}, "vue": {"stars": 207000}}
    })
    
    print("步骤 2: 分析数据...")
    state_manager.checkpoint(task_id, 2, {
        "repos": ["react", "vue"],
        "data_fetched": True,
        "github_data": {"react": {"stars": 220000}, "vue": {"stars": 207000}},
        "analysis_done": True,
        "comparison": "React 领先 6%"
    })
    
    print("步骤 3: 生成报告...")
    state_manager.checkpoint(task_id, 3, {
        "repos": ["react", "vue"],
        "data_fetched": True,
        "github_data": {"react": {"stars": 220000}, "vue": {"stars": 207000}},
        "analysis_done": True,
        "comparison": "React 领先 6%",
        "report_generated": True,
        "final_report": "# React vs Vue 分析报告"
    })
    
    # 模拟崩溃后恢复
    print("\n模拟任务崩溃，尝试恢复...")
    restored_state = state_manager.resume(task_id)
    
    if restored_state:
        print(f"✅ 恢复成功！当前步骤: {restored_state['step']}")
        print(f"   数据: {restored_state['data']}")
    
    # 列出所有检查点
    print("\n所有检查点:")
    checkpoints = state_manager.checkpointer.list_checkpoints()
    for cp in checkpoints[:5]:  # 只显示前5个
        print(f"  - {cp['checkpoint_id']} ({cp['timestamp']})")


def demo_golden_dataset():
    """演示黄金数据集"""
    print("\n" + "="*60)
    print("🧪 Demo 4: 黄金数据集")
    print("="*60)
    
    # 创建示例数据集
    dataset_path = "tests/evaluation/demo_dataset.jsonl"
    GoldenDataset.create_sample_dataset(dataset_path, num_cases=5)
    
    # 加载数据集
    dataset = GoldenDataset(dataset_path)
    
    print(f"\n数据集统计:")
    print(f"  总用例数: {len(dataset)}")
    print(f"  对比类: {len(dataset.get_by_category('comparison'))}")
    print(f"  分析类: {len(dataset.get_by_category('analysis'))}")
    print(f"  关键用例: {len(dataset.get_critical_cases())}")
    
    print("\n前 3 个测试用例:")
    for i, case in enumerate(dataset):
        if i >= 3:
            break
        print(f"\n  [{case.id}] {case.category} ({case.difficulty})")
        print(f"    输入: {case.input[:50]}...")
        print(f"    期望格式: {case.expected_output.get('format', 'N/A')}")


def main():
    """运行所有演示"""
    print("\n" + "🚀"*30)
    print("  Harness Engineering MVP 演示")
    print("🚀"*30)
    
    try:
        demo_prompt_version()
    except Exception as e:
        print(f"⚠️  Prompt 版本演示出错: {e}")
    
    try:
        demo_monitor()
    except Exception as e:
        print(f"⚠️  监控演示出错: {e}")
    
    try:
        demo_checkpointer()
    except Exception as e:
        print(f"⚠️  Checkpointer 演示出错: {e}")
    
    try:
        demo_golden_dataset()
    except Exception as e:
        print(f"⚠️  数据集演示出错: {e}")
    
    print("\n" + "="*60)
    print("✅ 演示完成！")
    print("="*60)
    print("\n下一步:")
    print("  1. 创建真实数据集: python scripts/run_evaluation.py --create-sample")
    print("  2. 运行评测: python scripts/run_evaluation.py --version v1")
    print("  3. 集成到你的 ChatAgent 中")


if __name__ == "__main__":
    main()
