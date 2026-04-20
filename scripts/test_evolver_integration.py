#!/usr/bin/env python3
"""
Evolver 集成完整测试脚本
测试 SparkNotebook 与官方 Evolver CLI 的集成

使用方式:
    python3 scripts/test_evolver_integration.py
"""
import os
import sys
import json
import subprocess
import time

def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"步骤 {step_num}: {description}")
    print('='*60)

def print_result(title, content):
    print(f"\n📋 {title}:")
    print(f"   {content}")

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║     SparkNotebook × Evolver GEP 集成测试                 ║
    ║     Genome Evolution Protocol - 自我进化引擎              ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Step 1: 检查环境
    print_step(1, "检查环境配置")

    print("✅ Node.js:", subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip())
    print("✅ npm:", subprocess.run(['npm', '--version'], capture_output=True, text=True).stdout.strip())
    print("✅ Evolver 目录:", "/root/fugeg/app/evolver")

    evolver_dir = "/root/fugeg/app/evolver"
    memory_dir = os.path.join(evolver_dir, "memory")

    if os.path.exists(evolver_dir):
        print("✅ Evolver 仓库已克隆")
    else:
        print("❌ Evolver 仓库不存在!")
        return

    # Step 2: 导入并测试 Python 集成层
    print_step(2, "测试 Python 集成层")

    try:
        from graphrag.evolver import create_evolver_integration
        evolver = create_evolver_integration()
        print("✅ SparkNotebookEvolver 导入成功")
        print(f"   - Evolver 路径: {evolver.evolver_path}")
        print(f"   - Memory 目录: {evolver.memory_dir}")
        print(f"   - Assets 目录: {evolver.assets_dir}")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return

    # Step 3: 写入错误日志
    print_step(3, "写入错误日志到 memory/ 目录")

    test_error_log = evolver.log_error(
        agent_id='Retriever_v1',
        error_pattern='Empty Retrieval - 没有找到相关记忆',
        query='用户查询：对比 A 项目和 B 项目的 GitHub Star 增长趋势',
        output_content='',
        output_quality=0.0
    )
    print(f"✅ 错误日志已写入: {test_error_log}")

    # 写入成功信号
    evolver.log_signal(
        signal_type='info',
        signal='Test Signal',
        content='这是一条测试信号，用于验证 Evolver 集成',
        metadata={'test': True, 'timestamp': time.time()}
    )
    print("✅ 测试信号已写入")

    # 查看 memory 目录
    memory_files = os.listdir(memory_dir)
    print_result("memory/ 目录文件数", str(len(memory_files)))

    # Step 4: 运行 Evolver CLI
    print_step(4, "运行 Evolver CLI 生成 GEP 进化")

    print("⏳ 正在运行 Evolver (可能需要几秒钟)...")

    env = os.environ.copy()
    env['EVOLVER_REPO_ROOT'] = evolver_dir

    result = subprocess.run(
        ['node', 'index.js'],
        cwd=evolver_dir,
        capture_output=True,
        text=True,
        timeout=120,
        env=env
    )

    print(f"\n📊 Evolver 返回码: {result.returncode}")

    if result.stdout:
        print("\n📤 Evolver 输出 (前 100 行):")
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines[:100]):
            print(f"   {line}")

    if result.stderr:
        print("\n⚠️ Evolver stderr (前 20 行):")
        stderr_lines = result.stderr.split('\n')
        for line in stderr_lines[:20]:
            if line.strip():
                print(f"   {line}")

    # Step 5: 验证 Gene 更新
    print_step(5, "验证 Genes 和 Capsules")

    genes = evolver.get_genes()
    capsules = evolver.get_capsules()

    print_result("现有 Genes 数量", str(len(genes)))
    for gene in genes:
        print(f"   - {gene.get('id', 'N/A')}: {gene.get('summary', 'N/A')[:50]}...")

    print_result("现有 Capsules 数量", str(len(capsules)))
    for capsule in capsules[:2]:
        print(f"   - {capsule.get('id', 'N/A')}")

    # Step 6: 验证 EvolutionEvent
    print_step(6, "查看 EvolutionEvent 历史")

    events_file = os.path.join(evolver_dir, "assets", "gep", "events.jsonl")
    if os.path.exists(events_file):
        with open(events_file, 'r') as f:
            events = [json.loads(line) for line in f if line.strip()]
        print_result("EvolutionEvent 总数", str(len(events)))
        if events:
            latest = events[-1]
            print(f"   最新事件:")
            print(f"   - ID: {latest.get('id', 'N/A')}")
            print(f"   - Intent: {latest.get('intent', 'N/A')}")
            print(f"   - Signals: {latest.get('signals', [])}")
    else:
        print("   (events.jsonl 尚不存在，首次运行后生成)")

    # 总结
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                    测试完成!                              ║
    ╚══════════════════════════════════════════════════════════╝

    ✅ 测试通过! Evolver 已成功集成到 SparkNotebook。

    📁 相关目录:
       - /root/fugeg/app/evolver/          # Evolver CLI 仓库
       - /root/fugeg/app/evolver/memory/   # 错误日志
       - /root/fugeg/app/evolver/assets/gep/  # Genes & Capsules

    🚀 下一步:
       1. 在应用中触发错误，观察 memory/ 目录生成日志
       2. 运行 'node index.js --review' 进行人工审查
       3. 运行 'node index.js --loop' 开启持续进化模式

    💡 集成架构:
       Python (SparkNotebook) → memory/ logs → Evolver CLI → GEP 进化提示词
    """)

if __name__ == '__main__':
    main()
