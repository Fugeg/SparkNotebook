"""
可断点续跑的评估工具链
支持中断后从上次进度继续运行
"""
import json
import os
import sys
import time
import signal
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphrag.tests.test_data_generator import TestDataGenerator
from graphrag.tests.evaluator import EvaluatorAgent
from graphrag.agents.memory_generator_agent import MemoryGeneratorAgent
from graphrag.utils.logger import Logger


class ResumableEvaluationPipeline:
    """
    可断点续跑的评估流水线
    
    特性:
    - 每处理一条样本自动保存进度
    - 支持从上次中断处继续运行
    - 自动检测并恢复未完成的评估任务
    """
    
    def __init__(self, session_id=None, output_dir=None):
        self.logger = Logger()
        self.data_generator = TestDataGenerator()
        self.evaluator = EvaluatorAgent()
        
        # 创建 mock db
        self.db = MockDB()
        self.memory_generator = MemoryGeneratorAgent(self.db, self.logger)
        
        # 设置输出目录
        self.output_dir = output_dir or os.path.dirname(__file__)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 会话管理
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_file = os.path.join(self.output_dir, f"checkpoint_{self.session_id}.json")
        self.results_file = os.path.join(self.output_dir, f"extraction_results_{self.session_id}.json")
        
        # 状态跟踪
        self.processed_count = 0
        self.total_count = 0
        self.is_running = False
        self.current_test_data = []
        self.extraction_results = []
        
        # 注册信号处理
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """注册信号处理器以实现优雅退出"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print(f"\n\n收到中断信号 ({signum})，正在保存进度...")
        self.is_running = False
        self._save_checkpoint()
        print(f"进度已保存到: {self.checkpoint_file}")
        print(f"已处理: {self.processed_count}/{self.total_count}")
        print("可以使用 --resume 参数恢复运行")
        sys.exit(0)
    
    def run_evaluation(self, sample_count=100, resume=False):
        """
        运行评估（支持断点续跑）
        
        Args:
            sample_count: 测试样本数量
            resume: 是否从上次中断处恢复
            
        Returns:
            dict: 评估结果汇总
        """
        print("=" * 70)
        print("GraphRAG 记忆提取质量评估工具链 (支持断点续跑)")
        print("=" * 70)
        print(f"会话ID: {self.session_id}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # 检查是否有可恢复的进度
        if resume and self._can_resume():
            self._load_checkpoint()
            print(f"\n✓ 已恢复上次进度: {self.processed_count}/{self.total_count}")
        else:
            # 生成新的测试数据
            print(f"\n[步骤 1/4] 生成 {sample_count} 条测试数据...")
            self.total_count = sample_count
            self.current_test_data = self.data_generator.generate_batch(
                count=sample_count,
                output_file=os.path.join(self.output_dir, f"test_data_{self.session_id}.json")
            )
            self.extraction_results = []
            self.processed_count = 0
            self._save_checkpoint()
        
        # 运行记忆提取（支持断点续跑）
        print(f"\n[步骤 2/4] 运行记忆提取 ({self.processed_count}/{self.total_count})...")
        self._run_extraction_resumable()
        
        # 自动评估
        print("\n[步骤 3/4] 自动评估...")
        evaluation_file = os.path.join(self.output_dir, f"evaluation_{self.session_id}.json")
        summary = self.evaluator.batch_evaluate(self.extraction_results, output_file=evaluation_file)
        
        # 生成报告
        print("\n[步骤 4/4] 生成评估报告...")
        report_file = os.path.join(self.output_dir, f"report_{self.session_id}.md")
        self._generate_report(summary, report_file)
        
        # 清理检查点文件
        self._cleanup_checkpoint()
        
        print("\n" + "=" * 70)
        print("评估完成!")
        print(f"结果文件:")
        print(f"  - 测试数据: {os.path.join(self.output_dir, f'test_data_{self.session_id}.json')}")
        print(f"  - 提取结果: {self.results_file}")
        print(f"  - 评估详情: {evaluation_file}")
        print(f"  - 评估报告: {report_file}")
        print("=" * 70)
        
        return summary
    
    def _run_extraction_resumable(self):
        """可断点续跑的记忆提取"""
        self.is_running = True
        
        # 从上次中断处继续
        start_index = self.processed_count
        remaining_data = self.current_test_data[start_index:]
        
        for i, data in enumerate(remaining_data):
            actual_index = start_index + i
            
            if not self.is_running:
                break
            
            print(f"  处理 {actual_index + 1}/{self.total_count}: {data['id']}")
            
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
            
            self.extraction_results.append(result)
            self.processed_count += 1
            
            # 每处理一条保存一次检查点
            self._save_checkpoint()
            
            # 添加延迟避免 API 限流
            time.sleep(0.5)
        
        self.is_running = False
        
        # 保存最终结果
        self._save_results()
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "processed_count": self.processed_count,
            "total_count": self.total_count,
            "current_test_data": self.current_test_data,
            "extraction_results": self.extraction_results
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def _load_checkpoint(self):
        """加载检查点"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            self.session_id = checkpoint.get("session_id", self.session_id)
            self.processed_count = checkpoint.get("processed_count", 0)
            self.total_count = checkpoint.get("total_count", 0)
            self.current_test_data = checkpoint.get("current_test_data", [])
            self.extraction_results = checkpoint.get("extraction_results", [])
    
    def _can_resume(self):
        """检查是否可以恢复"""
        return os.path.exists(self.checkpoint_file)
    
    def _save_results(self):
        """保存提取结果"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_results, f, ensure_ascii=False, indent=2)
    
    def _cleanup_checkpoint(self):
        """清理检查点文件"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
            print(f"  已清理检查点文件")
    
    def _generate_report(self, summary, output_file):
        """生成 Markdown 格式的评估报告"""
        
        # 计算成功率
        success_count = sum(1 for r in self.extraction_results if r.get('extraction_success', False))
        success_rate = success_count / len(self.extraction_results) if self.extraction_results else 0
        
        # 按场景类型分组统计
        scenario_stats = {}
        for result in self.extraction_results:
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
| 提取成功率 | {success_rate:.1%} ({success_count}/{len(self.extraction_results)}) |
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
        total_tp = sum(r.get('evaluation', {}).get('entity_metrics', {}).get('tp', 0) for r in self.extraction_results)
        total_fp = sum(r.get('evaluation', {}).get('entity_metrics', {}).get('fp', 0) for r in self.extraction_results)
        total_fn = sum(r.get('evaluation', {}).get('entity_metrics', {}).get('fn', 0) for r in self.extraction_results)
        
        report += f"""| 类型 | 数量 |
|------|------|
| 真正例 (TP) | {total_tp} |
| 假正例 (FP) | {total_fp} |
| 假反例 (FN) | {total_fn} |

## 5. 详细案例分析

### 5.1 优秀案例 (评分 >= 0.9)

"""
        
        # 找出优秀案例
        excellent_cases = [r for r in self.extraction_results 
                          if r.get('evaluation', {}).get('overall_score', 0) >= 0.9]
        
        for i, case in enumerate(excellent_cases[:3]):
            report += f"""**案例 {i+1}** (ID: {case['id']})
- 场景: {case['scenario_type']}
- 评分: {case['evaluation']['overall_score']}
- 原文片段: {case['raw_text'][:100]}...

"""
        
        report += """### 5.2 待改进案例 (评分 < 0.6)

"""
        
        # 找出待改进案例
        poor_cases = [r for r in self.extraction_results 
                     if r.get('evaluation', {}).get('overall_score', 0) < 0.6]
        
        for i, case in enumerate(poor_cases[:3]):
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


def list_checkpoints(output_dir=None):
    """列出所有可恢复的检查点"""
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    checkpoints = []
    for file in os.listdir(output_dir):
        if file.startswith("checkpoint_") and file.endswith(".json"):
            checkpoint_path = os.path.join(output_dir, file)
            try:
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                checkpoints.append({
                    "session_id": data.get("session_id"),
                    "timestamp": data.get("timestamp"),
                    "progress": f"{data.get('processed_count', 0)}/{data.get('total_count', 0)}",
                    "file": checkpoint_path
                })
            except:
                pass
    
    return checkpoints


def resume_evaluation(session_id=None, output_dir=None):
    """
    恢复之前的评估任务
    
    Args:
        session_id: 指定会话ID，如果为None则自动查找最新的检查点
        output_dir: 输出目录
    """
    if output_dir is None:
        output_dir = os.path.dirname(__file__)
    
    # 如果没有指定session_id，查找最新的检查点
    if session_id is None:
        checkpoints = list_checkpoints(output_dir)
        if not checkpoints:
            print("没有找到可恢复的检查点")
            return None
        
        # 按时间排序，取最新的
        checkpoints.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        session_id = checkpoints[0]["session_id"]
        print(f"找到最新的检查点: {session_id}")
    
    # 创建pipeline并恢复
    pipeline = ResumableEvaluationPipeline(session_id=session_id, output_dir=output_dir)
    
    if not pipeline._can_resume():
        print(f"找不到会话 {session_id} 的检查点")
        return None
    
    summary = pipeline.run_evaluation(resume=True)
    return summary


def start_new_evaluation(sample_count=100, output_dir=None):
    """开始新的评估任务"""
    pipeline = ResumableEvaluationPipeline(output_dir=output_dir)
    summary = pipeline.run_evaluation(sample_count=sample_count, resume=False)
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GraphRAG 记忆提取质量评估工具 (支持断点续跑)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 开始新的评估（100条样本）
  python resumable_evaluation.py --new --count 100
  
  # 恢复上次的评估
  python resumable_evaluation.py --resume
  
  # 恢复指定的会话
  python resumable_evaluation.py --resume --session 20260409_143022
  
  # 列出所有可恢复的检查点
  python resumable_evaluation.py --list
        """
    )
    
    parser.add_argument('--new', action='store_true',
                       help='开始新的评估任务')
    parser.add_argument('--resume', action='store_true',
                       help='恢复上次的评估任务')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可恢复的检查点')
    parser.add_argument('--count', type=int, default=100,
                       help='样本数量（默认100）')
    parser.add_argument('--session', type=str, default=None,
                       help='指定会话ID进行恢复')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    
    args = parser.parse_args()
    
    if args.list:
        checkpoints = list_checkpoints(args.output_dir)
        if checkpoints:
            print("\n可恢复的检查点:")
            print("-" * 70)
            for cp in checkpoints:
                print(f"会话ID: {cp['session_id']}")
                print(f"  时间: {cp['timestamp']}")
                print(f"  进度: {cp['progress']}")
                print()
        else:
            print("没有找到可恢复的检查点")
    
    elif args.resume:
        resume_evaluation(args.session, args.output_dir)
    
    elif args.new:
        start_new_evaluation(args.count, args.output_dir)
    
    else:
        # 默认行为：如果有检查点则提示恢复，否则开始新任务
        checkpoints = list_checkpoints(args.output_dir)
        if checkpoints:
            print("\n发现未完成的评估任务:")
            for i, cp in enumerate(checkpoints[:3], 1):
                print(f"  {i}. 会话 {cp['session_id']} - 进度 {cp['progress']}")
            
            choice = input("\n是否恢复上次任务? (y/n): ").strip().lower()
            if choice == 'y':
                resume_evaluation(checkpoints[0]["session_id"], args.output_dir)
            else:
                start_new_evaluation(args.count, args.output_dir)
        else:
            start_new_evaluation(args.count, args.output_dir)
