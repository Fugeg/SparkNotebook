"""
批量测试运行脚本 - 完整的评估工具链
整合数据生成、记忆提取和自动评估
"""
import json
import os
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphrag.tests.test_data_generator import TestDataGenerator
from graphrag.tests.evaluator import EvaluatorAgent
from graphrag.agents.memory_generator_agent import MemoryGeneratorAgent
from graphrag.utils.logger import Logger


class EvaluationPipeline:
    """
    评估流水线 - 完整的自动化测试工具链
    
    流程:
    1. 生成模拟数据
    2. 使用 MemoryGeneratorAgent 提取信息
    3. 使用 EvaluatorAgent 自动评分
    4. 生成评估报告
    """
    
    def __init__(self, db=None):
        self.logger = Logger()
        self.data_generator = TestDataGenerator()
        self.evaluator = EvaluatorAgent()
        
        # 创建 mock db 用于 MemoryGeneratorAgent
        if db is None:
            db = MockDB()
        self.memory_generator = MemoryGeneratorAgent(db, self.logger)
    
    def run_full_evaluation(self, sample_count=100, output_dir=None):
        """
        运行完整评估流程
        
        Args:
            sample_count: 测试样本数量
            output_dir: 输出目录
            
        Returns:
            dict: 评估结果汇总
        """
        if output_dir is None:
            output_dir = os.path.dirname(__file__)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("=" * 60)
        print("GraphRAG 记忆提取质量评估工具链")
        print("=" * 60)
        print(f"测试样本数: {sample_count}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # 步骤 1: 生成测试数据
        print("\n[步骤 1/4] 生成测试数据...")
        test_data_file = os.path.join(output_dir, f"test_data_{timestamp}.json")
        test_data = self.data_generator.generate_batch(
            count=sample_count,
            output_file=test_data_file
        )
        
        # 步骤 2: 运行记忆提取
        print("\n[步骤 2/4] 运行记忆提取...")
        extraction_results = self._run_extraction(test_data)
        
        # 步骤 3: 自动评估
        print("\n[步骤 3/4] 自动评估...")
        evaluation_file = os.path.join(output_dir, f"evaluation_{timestamp}.json")
        summary = self.evaluator.batch_evaluate(extraction_results, output_file=evaluation_file)
        
        # 步骤 4: 生成报告
        print("\n[步骤 4/4] 生成评估报告...")
        report_file = os.path.join(output_dir, f"report_{timestamp}.md")
        self._generate_report(summary, extraction_results, report_file)
        
        print("\n" + "=" * 60)
        print("评估完成!")
        print(f"结果文件:")
        print(f"  - 测试数据: {test_data_file}")
        print(f"  - 评估详情: {evaluation_file}")
        print(f"  - 评估报告: {report_file}")
        print("=" * 60)
        
        return summary
    
    def _run_extraction(self, test_data):
        """对测试数据运行记忆提取"""
        results = []
        
        for i, data in enumerate(test_data):
            print(f"  处理 {i+1}/{len(test_data)}: {data['id']}")
            
            try:
                # 使用 MemoryGeneratorAgent 提取信息
                extracted_json = self.memory_generator.process_input(data['raw_text'])
                
                result = {
                    "id": data['id'],
                    "scenario_type": data['scenario_type'],
                    "raw_text": data['raw_text'],
                    "preset_entities": data['preset_entities'],
                    "extracted_json": extracted_json if extracted_json else [],
                    "extraction_success": extracted_json is not None and len(extracted_json) > 0
                }
            except Exception as e:
                print(f"    提取失败: {e}")
                result = {
                    "id": data['id'],
                    "scenario_type": data['scenario_type'],
                    "raw_text": data['raw_text'],
                    "preset_entities": data['preset_entities'],
                    "extracted_json": [],
                    "extraction_success": False,
                    "error": str(e)
                }
            
            results.append(result)
            
            # 添加延迟避免 API 限流
            time.sleep(0.5)
        
        return results
    
    def _generate_report(self, summary, detailed_results, output_file):
        """生成 Markdown 格式的评估报告"""
        
        # 计算成功率
        success_count = sum(1 for r in detailed_results if r.get('extraction_success', False))
        success_rate = success_count / len(detailed_results) if detailed_results else 0
        
        # 按场景类型分组统计
        scenario_stats = {}
        for result in detailed_results:
            scenario = result.get('scenario_type', 'unknown')
            if scenario not in scenario_stats:
                scenario_stats[scenario] = {
                    'count': 0,
                    'total_score': 0,
                    'success_count': 0
                }
            scenario_stats[scenario]['count'] += 1
            if result.get('extraction_success'):
                scenario_stats[scenario]['success_count'] += 1
            if 'evaluation' in result:
                scenario_stats[scenario]['total_score'] += result['evaluation'].get('overall_score', 0)
        
        report = f"""# GraphRAG 记忆提取质量评估报告

## 1. 评估概览

| 指标 | 数值 |
|------|------|
| 测试样本总数 | {summary['total_samples']} |
| 提取成功率 | {success_rate:.1%} ({success_count}/{len(detailed_results)}) |
| 平均综合评分 | {summary['average_score']:.3f} |
| 平均精确率 (P) | {summary['average_precision']:.3f} |
| 平均召回率 (R) | {summary['average_recall']:.3f} |
| 平均 F1 分数 | {summary['average_f1']:.3f} |

## 2. 评分分布

| 评分区间 | 样本数 | 占比 |
|----------|--------|------|
| 优秀 (>=0.9) | {summary['score_distribution']['excellent (>=0.9)']} | {summary['score_distribution']['excellent (>=0.9)']/summary['total_samples']:.1%} |
| 良好 (0.8-0.9) | {summary['score_distribution']['good (0.8-0.9)']} | {summary['score_distribution']['good (0.8-0.9)']/summary['total_samples']:.1%} |
| 一般 (0.6-0.8) | {summary['score_distribution']['fair (0.6-0.8)']} | {summary['score_distribution']['fair (0.6-0.8)']/summary['total_samples']:.1%} |
| 较差 (<0.6) | {summary['score_distribution']['poor (<0.6)']} | {summary['score_distribution']['poor (<0.6)']/summary['total_samples']:.1%} |

## 3. 场景类型分析

| 场景类型 | 样本数 | 成功率 | 平均评分 |
|----------|--------|--------|----------|
"""
        
        for scenario, stats in scenario_stats.items():
            avg_score = stats['total_score'] / stats['count'] if stats['count'] > 0 else 0
            scenario_success_rate = stats['success_count'] / stats['count'] if stats['count'] > 0 else 0
            report += f"| {scenario} | {stats['count']} | {scenario_success_rate:.1%} | {avg_score:.3f} |\n"
        
        report += f"""
## 4. 论文表 5-1 数据

根据评估结果，论文中表 5-1 的数据如下：

| 评估指标 | 数值 |
|----------|------|
| 精确率 (Precision) | {summary['average_precision']:.3f} |
| 召回率 (Recall) | {summary['average_recall']:.3f} |
| F1-Score | {summary['average_f1']:.3f} |

### 4.1 混淆矩阵统计

"""
        
        # 统计 TP, FP, FN
        total_tp = sum(r.get('evaluation', {}).get('entity_metrics', {}).get('tp', 0) for r in detailed_results)
        total_fp = sum(r.get('evaluation', {}).get('entity_metrics', {}).get('fp', 0) for r in detailed_results)
        total_fn = sum(r.get('evaluation', {}).get('entity_metrics', {}).get('fn', 0) for r in detailed_results)
        
        report += f"""| 类型 | 数量 |
|------|------|
| 真正例 (TP) | {total_tp} |
| 假正例 (FP) | {total_fp} |
| 假反例 (FN) | {total_fn} |

## 5. 详细案例分析

### 5.1 优秀案例 (评分 >= 0.9)

"""
        
        # 找出优秀案例
        excellent_cases = [r for r in detailed_results 
                          if r.get('evaluation', {}).get('overall_score', 0) >= 0.9]
        
        for i, case in enumerate(excellent_cases[:3]):  # 只展示前3个
            report += f"""**案例 {i+1}** (ID: {case['id']})
- 场景: {case['scenario_type']}
- 评分: {case['evaluation']['overall_score']}
- 原文片段: {case['raw_text'][:100]}...

"""
        
        report += """### 5.2 待改进案例 (评分 < 0.6)

"""
        
        # 找出待改进案例
        poor_cases = [r for r in detailed_results 
                     if r.get('evaluation', {}).get('overall_score', 0) < 0.6]
        
        for i, case in enumerate(poor_cases[:3]):  # 只展示前3个
            report += f"""**案例 {i+1}** (ID: {case['id']})
- 场景: {case['scenario_type']}
- 评分: {case['evaluation'].get('overall_score', 'N/A')}
- 原文片段: {case['raw_text'][:100]}...
- 问题: {case['evaluation'].get('detailed_analysis', {}).get('weaknesses', ['未记录'])}

"""
        
        report += f"""## 6. 结论与建议

基于本次评估 ({summary['total_samples']} 个样本)，得出以下结论：

1. **整体表现**: 系统平均综合评分为 {summary['average_score']:.3f}，提取成功率为 {success_rate:.1%}。

2. **实体识别**: 平均精确率为 {summary['average_precision']:.3f}，召回率为 {summary['average_recall']:.3f}。

3. **改进方向**:
   - 优化实体识别 Prompt，提高召回率
   - 增强关系抽取能力
   - 改进复杂场景的处理

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"  报告已保存到: {output_file}")


class MockDB:
    """Mock 数据库，用于测试"""
    def insert_node(self, *args, **kwargs):
        return 1
    
    def insert_edge(self, *args, **kwargs):
        pass
    
    def search_similar_nodes(self, *args, **kwargs):
        return []


def quick_test(sample_count=5):
    """快速测试 - 使用少量样本验证工具链"""
    print("\n" + "=" * 60)
    print("快速测试模式")
    print("=" * 60)
    
    pipeline = EvaluationPipeline()
    summary = pipeline.run_full_evaluation(sample_count=sample_count)
    
    return summary


def full_evaluation(sample_count=100):
    """完整评估 - 使用指定数量的样本进行全面评估"""
    print("\n" + "=" * 60)
    print("完整评估模式")
    print("=" * 60)
    
    pipeline = EvaluationPipeline()
    summary = pipeline.run_full_evaluation(sample_count=sample_count)
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GraphRAG 记忆提取质量评估工具')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='运行模式: quick(快速测试5条) 或 full(完整评估100条)')
    parser.add_argument('--count', type=int, default=None,
                       help='自定义样本数量')
    
    args = parser.parse_args()
    
    if args.count:
        full_evaluation(args.count)
    elif args.mode == 'quick':
        quick_test(5)
    else:
        full_evaluation(100)
