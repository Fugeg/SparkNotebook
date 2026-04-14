"""
CI 执行器
自动化运行评测流程，生成报告，门禁检查
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import asdict

from harness.evaluation.golden_dataset import GoldenDataset
from harness.evaluation.llm_judge import LLMJudge, EvaluationResult


class CIRunner:
    """CI 评测执行器"""
    
    def __init__(self, 
                 dataset_path: str,
                 output_dir: str = "tests/evaluation/results",
                 judge_model: str = "qwen-plus",
                 pass_threshold: float = 75.0):
        """
        初始化 CI 执行器
        
        Args:
            dataset_path: 黄金数据集路径
            output_dir: 评测结果输出目录
            judge_model: 裁判模型
            pass_threshold: 通过阈值
        """
        self.dataset = GoldenDataset(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.judge = LLMJudge(
            model=judge_model,
            pass_threshold=pass_threshold
        )
        
        self.results: List[EvaluationResult] = []
    
    def run_evaluation(self, 
                       agent_fn: Callable[[str], str],
                       version: str = "v1",
                       save_results: bool = True) -> Dict[str, Any]:
        """
        运行完整评测流程
        
        Args:
            agent_fn: Agent 执行函数，接收 input 返回 output
            version: 当前评测的版本标识
            save_results: 是否保存结果
        
        Returns:
            评测报告字典
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始 CI 评测 - 版本: {version}")
        print(f"{'='*60}")
        print(f"📊 测试用例数: {len(self.dataset)}")
        print(f"🎯 通过阈值: {self.judge.pass_threshold}")
        print(f"⚖️  裁判模型: {self.judge.model}")
        print(f"{'='*60}\n")
        
        self.results = []
        agent_outputs = []
        
        # 1. 执行所有测试用例
        print("📝 步骤 1: 执行测试用例...")
        for i, test_case in enumerate(self.dataset):
            print(f"  [{i+1}/{len(self.dataset)}] {test_case.id}: {test_case.input[:50]}...")
            try:
                output = agent_fn(test_case.input)
                agent_outputs.append(output)
            except Exception as e:
                print(f"    ❌ 执行失败: {e}")
                agent_outputs.append(f"[ERROR] {str(e)}")
        
        # 2. LLM-as-a-Judge 评分
        print("\n⚖️  步骤 2: LLM 评分...")
        for i, (test_case, output) in enumerate(zip(self.dataset, agent_outputs)):
            result = self.judge.evaluate(
                test_case.__dict__,
                output
            )
            self.results.append(result)
            status = "✅" if result.passed else "❌"
            print(f"  {status} {result.test_case_id}: {result.total_score:.1f}分")
        
        # 3. 生成报告
        print("\n📊 步骤 3: 生成评测报告...")
        report = self._generate_report(version)
        
        # 4. 门禁检查
        print("\n🚪 步骤 4: 门禁检查...")
        gate_passed = self._gatekeeper_check(report)
        report["gate_passed"] = gate_passed
        
        # 5. 保存结果
        if save_results:
            self._save_results(report, version)
        
        # 6. 打印摘要
        self._print_summary(report)
        
        return report
    
    def _generate_report(self, version: str) -> Dict[str, Any]:
        """生成评测报告"""
        total_cases = len(self.results)
        passed_cases = sum(1 for r in self.results if r.passed)
        failed_cases = total_cases - passed_cases
        
        # 计算各维度平均分
        avg_scores = {}
        for dim in ["accuracy", "format", "logic", "efficiency"]:
            scores = [r.scores.get(dim, 0) for r in self.results if dim in r.scores]
            avg_scores[dim] = round(sum(scores) / len(scores), 2) if scores else 0
        
        # 综合得分
        total_scores = [r.total_score for r in self.results]
        avg_total = round(sum(total_scores) / len(total_scores), 2) if total_scores else 0
        
        # Token 消耗统计
        total_tokens = sum(
            r.token_usage.get("total_tokens", 0) 
            for r in self.results
        )
        
        # 关键用例检查
        critical_cases = self.dataset.get_critical_cases()
        critical_ids = {c.id for c in critical_cases}
        critical_results = [r for r in self.results if r.test_case_id in critical_ids]
        critical_passed = all(r.passed for r in critical_results)
        
        report = {
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "failed_cases": failed_cases,
                "pass_rate": round(passed_cases / total_cases * 100, 2),
                "avg_total_score": avg_total,
                "avg_scores": avg_scores,
                "total_tokens": total_tokens,
                "critical_cases_passed": critical_passed
            },
            "details": [asdict(r) for r in self.results]
        }
        
        return report
    
    def _gatekeeper_check(self, report: Dict[str, Any]) -> bool:
        """
        门禁检查
        
        通过条件：
        1. 综合得分 >= 阈值
        2. 关键用例全部通过
        3. 通过率 >= 80%
        """
        summary = report["summary"]
        
        checks = []
        
        # 检查 1: 综合得分
        score_pass = summary["avg_total_score"] >= self.judge.pass_threshold
        checks.append(("综合得分达标", score_pass, f"{summary['avg_total_score']:.1f} >= {self.judge.pass_threshold}"))
        
        # 检查 2: 关键用例
        critical_pass = summary["critical_cases_passed"]
        checks.append(("关键用例通过", critical_pass, "全部通过" if critical_pass else "有失败"))
        
        # 检查 3: 通过率
        pass_rate_pass = summary["pass_rate"] >= 80
        checks.append(("通过率达标", pass_rate_pass, f"{summary['pass_rate']:.1f}% >= 80%"))
        
        print("\n  门禁检查项:")
        for name, passed, detail in checks:
            status = "✅" if passed else "❌"
            print(f"    {status} {name}: {detail}")
        
        all_passed = all(passed for _, passed, _ in checks)
        return all_passed
    
    def _save_results(self, report: Dict[str, Any], version: str):
        """保存评测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{version}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存: {filepath}")
        
        # 同时保存最新结果
        latest_path = self.output_dir / "latest_result.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    
    def _print_summary(self, report: Dict[str, Any]):
        """打印评测摘要"""
        s = report["summary"]
        
        print(f"\n{'='*60}")
        print("📊 评测摘要")
        print(f"{'='*60}")
        print(f"版本: {report['version']}")
        print(f"时间: {report['timestamp']}")
        print(f"{'-'*60}")
        print(f"总用例数: {s['total_cases']}")
        print(f"通过: {s['passed_cases']} ✅")
        print(f"失败: {s['failed_cases']} ❌")
        print(f"通过率: {s['pass_rate']}%")
        print(f"{'-'*60}")
        print(f"综合得分: {s['avg_total_score']:.1f}")
        print(f"  - 准确性: {s['avg_scores'].get('accuracy', 0):.1f}")
        print(f"  - 格式对齐: {s['avg_scores'].get('format', 0):.1f}")
        print(f"  - 逻辑完整: {s['avg_scores'].get('logic', 0):.1f}")
        print(f"  - Token效率: {s['avg_scores'].get('efficiency', 0):.1f}")
        print(f"{'-'*60}")
        print(f"总 Token 消耗: {s['total_tokens']}")
        print(f"关键用例: {'✅ 全部通过' if s['critical_cases_passed'] else '❌ 有失败'}")
        print(f"{'-'*60}")
        print(f"门禁结果: {'✅ 通过' if report['gate_passed'] else '❌ 未通过'}")
        print(f"{'='*60}\n")
    
    def compare_versions(self, 
                         old_report_path: str, 
                         new_report_path: str) -> Dict[str, Any]:
        """
        对比两个版本的评测结果
        
        Args:
            old_report_path: 旧版本报告路径
            new_report_path: 新版本报告路径
        
        Returns:
            对比报告
        """
        with open(old_report_path, 'r') as f:
            old_report = json.load(f)
        with open(new_report_path, 'r') as f:
            new_report = json.load(f)
        
        old_score = old_report["summary"]["avg_total_score"]
        new_score = new_report["summary"]["avg_total_score"]
        
        comparison = {
            "old_version": old_report["version"],
            "new_version": new_report["version"],
            "old_score": old_score,
            "new_score": new_score,
            "improvement": round(new_score - old_score, 2),
            "improvement_percent": round((new_score - old_score) / old_score * 100, 2),
            "recommendation": "升级" if new_score > old_score else "保持现状"
        }
        
        print(f"\n{'='*60}")
        print("📊 版本对比")
        print(f"{'='*60}")
        print(f"旧版本 ({comparison['old_version']}): {comparison['old_score']:.1f}分")
        print(f"新版本 ({comparison['new_version']}): {comparison['new_score']:.1f}分")
        print(f"提升: {comparison['improvement']:+.1f}分 ({comparison['improvement_percent']:+.1f}%)")
        print(f"建议: {comparison['recommendation']}")
        print(f"{'='*60}\n")
        
        return comparison
