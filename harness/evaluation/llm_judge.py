"""
LLM-as-a-Judge 评分器
使用更强的模型作为裁判，评估 Agent 输出质量
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class EvaluationResult:
    """评测结果"""
    test_case_id: str
    agent_output: str
    scores: Dict[str, float]  # 各维度得分
    total_score: float        # 综合得分
    passed: bool              # 是否通过
    feedback: str             # 评判反馈
    timestamp: str
    token_usage: Dict[str, int]  # Token 消耗


class LLMJudge:
    """LLM 裁判评分器"""
    
    def __init__(self, 
                 model: str = "qwen-plus",
                 weights: Optional[Dict[str, float]] = None,
                 pass_threshold: float = 75.0):
        """
        初始化裁判
        
        Args:
            model: 裁判模型名称 (qwen-plus, gpt-4o, claude-3-sonnet)
            weights: 评分维度权重
            pass_threshold: 通过阈值
        """
        self.model = model
        self.weights = weights or {
            "accuracy": 0.4,      # 准确性
            "format": 0.3,        # 格式对齐
            "logic": 0.2,         # 逻辑完整性
            "efficiency": 0.1     # Token 效率
        }
        self.pass_threshold = pass_threshold
        
        # 初始化模型客户端
        self._init_client()
    
    def _init_client(self):
        """初始化模型客户端"""
        if "qwen" in self.model.lower():
            try:
                import dashscope
                dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
                self.client = dashscope
                self.client_type = "dashscope"
            except ImportError:
                raise ImportError("请安装 dashscope: pip install dashscope")
        
        elif "gpt" in self.model.lower():
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.client_type = "openai"
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        
        else:
            raise ValueError(f"不支持的裁判模型: {self.model}")
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> tuple:
        """
        调用 LLM
        
        Returns:
            (输出文本, token_usage)
        """
        if self.client_type == "dashscope":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.Generation.call(
                model=self.model,
                messages=messages,
                temperature=0.1,  # 低温度确保评分一致性
                max_tokens=2000,
                result_format="message"
            )
            
            if response.status_code == 200:
                content = response.output.choices[0].message.content
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                return content, usage
            else:
                raise Exception(f"API 调用失败: {response.message}")
        
        elif self.client_type == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            return content, usage
        
        else:
            raise ValueError(f"未知的客户端类型: {self.client_type}")
    
    def evaluate(self, 
                 test_case: Dict[str, Any], 
                 agent_output: str) -> EvaluationResult:
        """
        评估单个测试用例
        
        Args:
            test_case: 测试用例
            agent_output: Agent 的输出
        
        Returns:
            EvaluationResult 评估结果
        """
        # 构建评判 Prompt
        judge_prompt = self._build_judge_prompt(test_case, agent_output)
        
        system_prompt = """你是一个严格的 AI 输出质量评判专家。
你的任务是根据给定的评分标准，客观、公正地评估 AI Agent 的输出质量。
请以 JSON 格式返回评分结果，确保评分的一致性和可重复性。"""
        
        try:
            # 调用裁判模型
            judgment, token_usage = self._call_llm(judge_prompt, system_prompt)
            
            # 解析评分结果
            scores = self._parse_judgment(judgment)
            
            # 计算综合得分
            total_score = sum(
                scores.get(dim, 0) * weight 
                for dim, weight in self.weights.items()
            )
            
            # 判断是否通过
            passed = total_score >= self.pass_threshold
            
            return EvaluationResult(
                test_case_id=test_case["id"],
                agent_output=agent_output[:500] + "..." if len(agent_output) > 500 else agent_output,
                scores=scores,
                total_score=round(total_score, 2),
                passed=passed,
                feedback=scores.get("feedback", "无反馈"),
                timestamp=datetime.now().isoformat(),
                token_usage=token_usage
            )
            
        except Exception as e:
            # 评分失败，返回失败结果
            return EvaluationResult(
                test_case_id=test_case["id"],
                agent_output=agent_output[:500] if len(agent_output) > 500 else agent_output,
                scores={},
                total_score=0.0,
                passed=False,
                feedback=f"评判失败: {str(e)}",
                timestamp=datetime.now().isoformat(),
                token_usage={}
            )
    
    def _build_judge_prompt(self, test_case: Dict[str, Any], agent_output: str) -> str:
        """构建评判 Prompt"""
        prompt = f"""请评估以下 AI Agent 的输出质量。

## 测试用例
- ID: {test_case['id']}
- 输入: {test_case['input']}
- 期望输出格式: {json.dumps(test_case['expected_output'], ensure_ascii=False)}
- 分类: {test_case.get('category', 'general')}
- 难度: {test_case.get('difficulty', 'medium')}

## Agent 实际输出
```
{agent_output}
```

## 评分标准 (0-100 分)
请从以下维度评分，并以 JSON 格式返回：

1. **accuracy** (准确性): 输出内容是否准确、事实是否正确
2. **format** (格式对齐): 是否符合期望的输出格式
3. **logic** (逻辑完整性): 推理过程是否完整、逻辑是否清晰
4. **efficiency** (Token 效率): 输出是否简洁、无冗余

## 返回格式
```json
{{
    "accuracy": 85,
    "format": 90,
    "logic": 80,
    "efficiency": 75,
    "feedback": "具体评价和建议..."
}}
```

请只返回 JSON，不要其他内容。"""
        
        return prompt
    
    def _parse_judgment(self, judgment: str) -> Dict[str, Any]:
        """解析评判结果"""
        try:
            # 尝试直接解析 JSON
            return json.loads(judgment)
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', judgment)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            
            # 解析失败，返回默认分数
            return {
                "accuracy": 50,
                "format": 50,
                "logic": 50,
                "efficiency": 50,
                "feedback": f"无法解析评判结果: {judgment[:200]}"
            }
    
    def evaluate_batch(self, 
                       test_cases: list, 
                       agent_outputs: list) -> list:
        """
        批量评估
        
        Args:
            test_cases: 测试用例列表
            agent_outputs: Agent 输出列表
        
        Returns:
            EvaluationResult 列表
        """
        results = []
        for i, (case, output) in enumerate(zip(test_cases, agent_outputs)):
            print(f"  评估进度: {i+1}/{len(test_cases)} - {case['id']}")
            result = self.evaluate(case, output)
            results.append(result)
        return results
