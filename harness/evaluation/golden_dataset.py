"""
黄金数据集管理器
用于管理测试用例的加载、保存和迭代
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestCase:
    """单个测试用例"""
    id: str
    input: str                    # 用户输入
    expected_output: Dict[str, Any]  # 期望输出格式/内容
    category: str = "general"     # 分类：comparison, analysis, summary
    difficulty: str = "medium"    # 难度：easy, medium, hard
    tags: List[str] = None        # 标签
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class GoldenDataset:
    """黄金数据集管理器"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.test_cases: List[TestCase] = []
        self._load_dataset()
    
    def _load_dataset(self):
        """从 JSONL 文件加载数据集"""
        if not self.dataset_path.exists():
            print(f"⚠️  数据集不存在: {self.dataset_path}")
            return
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    test_case = TestCase(**data)
                    self.test_cases.append(test_case)
                except Exception as e:
                    print(f"⚠️  解析测试用例失败: {e}")
        
        print(f"✅ 已加载 {len(self.test_cases)} 个测试用例")
    
    def save_dataset(self, output_path: Optional[str] = None):
        """保存数据集到 JSONL 文件"""
        path = Path(output_path) if output_path else self.dataset_path
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            for case in self.test_cases:
                f.write(json.dumps(asdict(case), ensure_ascii=False) + '\n')
        
        print(f"✅ 已保存 {len(self.test_cases)} 个测试用例到 {path}")
    
    def add_test_case(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)
    
    def get_by_category(self, category: str) -> List[TestCase]:
        """按分类获取测试用例"""
        return [case for case in self.test_cases if case.category == category]
    
    def get_by_difficulty(self, difficulty: str) -> List[TestCase]:
        """按难度获取测试用例"""
        return [case for case in self.test_cases if case.difficulty == difficulty]
    
    def get_critical_cases(self) -> List[TestCase]:
        """获取关键测试用例（用于门禁检查）"""
        return [case for case in self.test_cases if 'critical' in case.tags]
    
    def __len__(self) -> int:
        return len(self.test_cases)
    
    def __iter__(self):
        return iter(self.test_cases)
    
    @staticmethod
    def create_sample_dataset(output_path: str, num_cases: int = 10):
        """创建示例数据集"""
        sample_cases = [
            {
                "id": "case_001",
                "input": "对比 React 和 Vue 的 GitHub Star 增长趋势",
                "expected_output": {
                    "format": "comparison_table",
                    "required_fields": ["项目名", "当前Stars", "近30天增长", "增长率"],
                    "metrics": ["star_count", "growth_rate"]
                },
                "category": "comparison",
                "difficulty": "easy",
                "tags": ["critical", "frontend"]
            },
            {
                "id": "case_002",
                "input": "分析 langchain 项目的社区活跃度",
                "expected_output": {
                    "format": "analysis_report",
                    "required_fields": ["贡献者数量", "Issue响应时间", "PR合并率", "最近提交频率"],
                    "metrics": ["contributors", "issue_resolution_time", "pr_merge_rate"]
                },
                "category": "analysis",
                "difficulty": "medium",
                "tags": ["critical", "ai"]
            },
            {
                "id": "case_003",
                "input": "总结 kubernetes/kubernetes 项目最近一个月的更新亮点",
                "expected_output": {
                    "format": "summary",
                    "required_fields": ["主要更新", "影响范围", "版本号"],
                    "metrics": ["release_notes", "commit_count"]
                },
                "category": "summary",
                "difficulty": "medium",
                "tags": ["devops"]
            },
            {
                "id": "case_004",
                "input": "对比 TensorFlow 和 PyTorch 的企业采用情况",
                "expected_output": {
                    "format": "comparison_table",
                    "required_fields": ["框架", "企业采用率", "主要用户", "行业分布"],
                    "metrics": ["adoption_rate", "enterprise_users"]
                },
                "category": "comparison",
                "difficulty": "hard",
                "tags": ["ai", "enterprise"]
            },
            {
                "id": "case_005",
                "input": "评估 fastapi/fastapi 项目的代码质量和维护状态",
                "expected_output": {
                    "format": "analysis_report",
                    "required_fields": ["代码覆盖率", "文档完整度", "维护活跃度", "Issue处理速度"],
                    "metrics": ["code_quality", "maintenance_status"]
                },
                "category": "analysis",
                "difficulty": "medium",
                "tags": ["backend", "quality"]
            }
        ]
        
        # 如果要求更多用例，复制并修改
        cases = sample_cases.copy()
        while len(cases) < num_cases:
            for case in sample_cases:
                if len(cases) >= num_cases:
                    break
                new_case = case.copy()
                new_case["id"] = f"case_{len(cases)+1:03d}"
                cases.append(new_case)
        
        # 保存
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for case in cases[:num_cases]:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        print(f"✅ 已创建示例数据集: {output_path} ({num_cases} 个用例)")
        return output_path
