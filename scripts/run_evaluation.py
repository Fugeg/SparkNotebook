#!/usr/bin/env python3
"""
评测脚本 - 运行 Harness Engineering CI 评测

使用方法:
    python scripts/run_evaluation.py --version v1 --dataset tests/evaluation/golden_dataset.jsonl
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

from harness.evaluation.ci_runner import CIRunner
from harness.evaluation.golden_dataset import GoldenDataset


def mock_agent(input_text: str) -> str:
    """
    模拟 Agent 执行
    实际使用时替换为真实的 Agent 调用
    """
    # 这里只是一个示例，实际应该调用你的 ChatAgent
    return f"[模拟输出] 收到输入: {input_text[:50]}..."


def main():
    parser = argparse.ArgumentParser(description="运行 Harness Engineering 评测")
    parser.add_argument(
        "--version", 
        default="v1",
        help="评测版本标识 (默认: v1)"
    )
    parser.add_argument(
        "--dataset",
        default="tests/evaluation/golden_dataset.jsonl",
        help="黄金数据集路径"
    )
    parser.add_argument(
        "--judge-model",
        default="qwen-plus",
        help="裁判模型 (默认: qwen-plus)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=75.0,
        help="通过阈值 (默认: 75)"
    )
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="创建示例数据集"
    )
    
    args = parser.parse_args()
    
    # 创建示例数据集
    if args.create_sample:
        print("📝 创建示例数据集...")
        GoldenDataset.create_sample_dataset(args.dataset, num_cases=10)
        return
    
    # 检查数据集是否存在
    if not Path(args.dataset).exists():
        print(f"❌ 数据集不存在: {args.dataset}")
        print("💡 使用 --create-sample 创建示例数据集")
        return
    
    # 初始化 CI Runner
    print("🚀 初始化 CI Runner...")
    runner = CIRunner(
        dataset_path=args.dataset,
        judge_model=args.judge_model,
        pass_threshold=args.threshold
    )
    
    # 运行评测
    report = runner.run_evaluation(
        agent_fn=mock_agent,
        version=args.version
    )
    
    # 返回退出码（用于 CI 门禁）
    if report.get("gate_passed", False):
        print("✅ 评测通过！")
        sys.exit(0)
    else:
        print("❌ 评测未通过！")
        sys.exit(1)


if __name__ == "__main__":
    main()
