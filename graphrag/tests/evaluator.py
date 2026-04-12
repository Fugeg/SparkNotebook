"""
判别 Agent - 用于自动评估记忆提取质量
基于论文 5.1 节描述的判别 Agent 体系
"""
import json
import re
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphrag.models.llm import LLMModel


class EvaluatorAgent:
    """
    判别 Agent - 评估 MemoryGeneratorAgent 的提取质量
    采用"提示词 + 检索器"的判别方案
    """
    
    def __init__(self):
        self.llm = LLMModel()
    
    def evaluate(self, raw_text, extracted_json, preset_entities=None):
        """
        评估提取结果的质量
        
        Args:
            raw_text: 原始模拟数据文本
            extracted_json: MemoryGeneratorAgent 解析出的 JSON 结果
            preset_entities: 生成时预设的实体（用于对比）
            
        Returns:
            dict: 包含各项评分和详细分析
        """
        try:
            # 1. 实体级别评估
            entity_metrics = self._evaluate_entities(raw_text, extracted_json, preset_entities)
            
            # 2. 关系级别评估
            relation_metrics = self._evaluate_relations(raw_text, extracted_json)
            
            # 3. 完整性评估
            completeness_metrics = self._evaluate_completeness(raw_text, extracted_json)
            
            # 4. 逻辑一致性评估
            consistency_metrics = self._evaluate_consistency(raw_text, extracted_json)
            
            # 5. 综合评分
            overall_score = self._calculate_overall_score(
                entity_metrics, 
                relation_metrics, 
                completeness_metrics,
                consistency_metrics
            )
            
            return {
                "overall_score": overall_score,
                "entity_metrics": entity_metrics,
                "relation_metrics": relation_metrics,
                "completeness_metrics": completeness_metrics,
                "consistency_metrics": consistency_metrics,
                "detailed_analysis": self._generate_detailed_analysis(
                    raw_text, extracted_json, entity_metrics
                )
            }
        except Exception as e:
            print(f"评估失败: {e}")
            return {
                "overall_score": 0,
                "error": str(e)
            }
    
    def _evaluate_entities(self, raw_text, extracted_json, preset_entities=None):
        """
        实体级别评估 - 计算精确率(P)和召回率(R)
        
        TP: 正确提取的实体
        FP: 错误提取的实体（不存在于原文）
        FN: 遗漏的实体（存在于原文但未被提取）
        """
        # 从提取结果中获取实体
        extracted_entities = {
            "人物": [],
            "地点": [],
            "事件": []
        }
        
        for item in extracted_json:
            item_type = item.get("type", "")
            content = item.get("content", "")
            
            if item_type == "人物":
                extracted_entities["人物"].append(content)
            elif item_type == "地点":
                extracted_entities["地点"].append(content)
            elif item_type == "事件":
                extracted_entities["事件"].append(content)
        
        # 使用 LLM 判断实体是否在原文中存在
        tp, fp, fn = 0, 0, 0
        entity_judgments = []
        
        # 评估提取的实体
        for entity_type, entities in extracted_entities.items():
            for entity in entities:
                is_valid = self._check_entity_in_text(entity, raw_text)
                if is_valid:
                    tp += 1
                else:
                    fp += 1
                entity_judgments.append({
                    "entity": entity,
                    "type": entity_type,
                    "valid": is_valid
                })
        
        # 如果有预设实体，计算遗漏
        if preset_entities:
            for entity_type, entities in preset_entities.items():
                for entity in entities:
                    # 检查是否被提取
                    extracted_list = extracted_entities.get(entity_type, [])
                    if not any(entity in ext or ext in entity for ext in extracted_list):
                        fn += 1
        
        # 计算精确率和召回率
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "entity_judgments": entity_judgments
        }
    
    def _check_entity_in_text(self, entity, text):
        """使用 LLM 判断实体是否在原文中真实存在"""
        prompt = f"""请判断以下实体是否存在于原文中。

原文：{text[:500]}

实体："{entity}"

请回答：该实体是否真实存在于原文中？
只回答"是"或"否"。"""
        
        try:
            response = self.llm.chat(prompt)
            return "是" in response or "存在" in response
        except:
            # 降级方案：简单字符串匹配
            return entity in text
    
    def _evaluate_relations(self, raw_text, extracted_json):
        """
        关系级别评估 - 评估线索(Clue)和关系(Connection)的构建质量
        """
        relation_items = [item for item in extracted_json 
                         if item.get("type") in ["关系", "线索"]]
        
        if not relation_items:
            return {
                "relation_count": 0,
                "valid_relations": 0,
                "relation_accuracy": 0
            }
        
        valid_count = 0
        relation_judgments = []
        
        for item in relation_items:
            content = item.get("content", "")
            related_ids = item.get("related_ids", [])
            
            # 使用 LLM 判断关系是否合理
            is_valid = self._check_relation_validity(content, raw_text)
            if is_valid:
                valid_count += 1
            
            relation_judgments.append({
                "content": content,
                "valid": is_valid,
                "related_count": len(related_ids)
            })
        
        accuracy = valid_count / len(relation_items) if relation_items else 0
        
        return {
            "relation_count": len(relation_items),
            "valid_relations": valid_count,
            "relation_accuracy": round(accuracy, 3),
            "relation_judgments": relation_judgments
        }
    
    def _check_relation_validity(self, relation_content, text):
        """使用 LLM 判断关系描述是否符合原文"""
        prompt = f"""请判断以下关系描述是否符合原文内容。

原文：{text[:500]}

关系描述："{relation_content}"

请回答：该关系描述是否符合原文的因果关系或逻辑关联？
只回答"是"或"否"。"""
        
        try:
            response = self.llm.chat(prompt)
            return "是" in response
        except:
            return True  # 默认通过
    
    def _evaluate_completeness(self, raw_text, extracted_json):
        """
        完整性评估 - 检查是否遗漏关键信息
        """
        # 统计各类信息单元的数量
        type_counts = {}
        for item in extracted_json:
            item_type = item.get("type", "unknown")
            type_counts[item_type] = type_counts.get(item_type, 0) + 1
        
        # 使用 LLM 评估完整性
        prompt = f"""请评估以下信息提取的完整性。

原文：{raw_text[:500]}

提取的信息单元类型和数量：
{json.dumps(type_counts, ensure_ascii=False)}

请回答以下问题：
1. 原文中的关键人物是否都被提取？
2. 原文中的关键地点是否都被提取？
3. 原文中的关键事件是否都被提取？
4. 是否有重要的时间信息被遗漏？

请给出 0-1 之间的完整性评分（1 表示完全完整，0 表示非常不完整）："""
        
        try:
            response = self.llm.chat(prompt)
            # 尝试提取分数
            score_match = re.search(r'(0\.\d+|1\.0|1)', response)
            if score_match:
                score = float(score_match.group(1))
            else:
                score = 0.5
        except:
            score = 0.5
        
        return {
            "completeness_score": round(score, 3),
            "type_distribution": type_counts,
            "total_units": len(extracted_json)
        }
    
    def _evaluate_consistency(self, raw_text, extracted_json):
        """
        逻辑一致性评估 - 检查提取的信息之间是否存在矛盾
        """
        # 使用 LLM 评估一致性
        extracted_summary = json.dumps(extracted_json, ensure_ascii=False, indent=2)[:1000]
        
        prompt = f"""请评估以下提取结果的逻辑一致性。

原文：{raw_text[:500]}

提取的 JSON：
{extracted_summary}

请检查：
1. 提取的实体信息是否与原文一致？
2. 提取的关系是否合理？
3. 是否存在自相矛盾的信息？

请给出 0-1 之间的一致性评分（1 表示完全一致，0 表示存在严重矛盾）："""
        
        try:
            response = self.llm.chat(prompt)
            score_match = re.search(r'(0\.\d+|1\.0|1)', response)
            if score_match:
                score = float(score_match.group(1))
            else:
                score = 0.8
        except:
            score = 0.8
        
        return {
            "consistency_score": round(score, 3)
        }
    
    def _calculate_overall_score(self, entity_metrics, relation_metrics, 
                                  completeness_metrics, consistency_metrics):
        """计算综合评分"""
        # 权重配置
        weights = {
            "entity_f1": 0.35,
            "relation_accuracy": 0.25,
            "completeness": 0.25,
            "consistency": 0.15
        }
        
        overall = (
            weights["entity_f1"] * entity_metrics.get("f1_score", 0) +
            weights["relation_accuracy"] * relation_metrics.get("relation_accuracy", 0) +
            weights["completeness"] * completeness_metrics.get("completeness_score", 0) +
            weights["consistency"] * consistency_metrics.get("consistency_score", 0)
        )
        
        return round(overall, 3)
    
    def _generate_detailed_analysis(self, raw_text, extracted_json, entity_metrics):
        """生成详细分析报告"""
        analysis = {
            "strengths": [],
            "weaknesses": [],
            "suggestions": []
        }
        
        # 基于实体评估结果生成分析
        if entity_metrics.get("precision", 0) > 0.9:
            analysis["strengths"].append("实体提取精确率高，很少产生幻觉")
        if entity_metrics.get("recall", 0) > 0.8:
            analysis["strengths"].append("实体提取召回率高，很少遗漏关键信息")
        
        if entity_metrics.get("precision", 0) < 0.7:
            analysis["weaknesses"].append("存在较多错误提取的实体（幻觉）")
            analysis["suggestions"].append("建议增强实体验证机制")
        if entity_metrics.get("recall", 0) < 0.6:
            analysis["weaknesses"].append("遗漏了较多原文中的实体")
            analysis["suggestions"].append("建议改进实体识别Prompt")
        
        return analysis
    
    def batch_evaluate(self, test_results, output_file=None):
        """
        批量评估
        
        Args:
            test_results: 包含原始文本和提取结果的列表
            output_file: 输出文件路径
            
        Returns:
            dict: 汇总统计结果
        """
        print(f"开始批量评估 {len(test_results)} 条数据...")
        
        all_scores = []
        entity_precisions = []
        entity_recalls = []
        relation_accuracies = []
        
        for i, result in enumerate(test_results):
            print(f"  评估中 {i+1}/{len(test_results)}...")
            
            evaluation = self.evaluate(
                result.get("raw_text", ""),
                result.get("extracted_json", []),
                result.get("preset_entities")
            )
            
            all_scores.append(evaluation.get("overall_score", 0))
            entity_precisions.append(evaluation.get("entity_metrics", {}).get("precision", 0))
            entity_recalls.append(evaluation.get("entity_metrics", {}).get("recall", 0))
            relation_accuracies.append(evaluation.get("relation_metrics", {}).get("relation_accuracy", 0))
            
            # 将评估结果添加到原始结果中
            result["evaluation"] = evaluation
        
        # 计算统计指标
        summary = {
            "total_samples": len(test_results),
            "average_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0,
            "average_precision": round(sum(entity_precisions) / len(entity_precisions), 3) if entity_precisions else 0,
            "average_recall": round(sum(entity_recalls) / len(entity_recalls), 3) if entity_recalls else 0,
            "average_f1": round(2 * sum(entity_precisions) * sum(entity_recalls) / 
                               (sum(entity_precisions) + sum(entity_recalls)), 3) 
                          if (sum(entity_precisions) + sum(entity_recalls)) > 0 else 0,
            "score_distribution": {
                "excellent (>=0.9)": len([s for s in all_scores if s >= 0.9]),
                "good (0.8-0.9)": len([s for s in all_scores if 0.8 <= s < 0.9]),
                "fair (0.6-0.8)": len([s for s in all_scores if 0.6 <= s < 0.8]),
                "poor (<0.6)": len([s for s in all_scores if s < 0.6])
            }
        }
        
        # 保存结果
        if output_file:
            output_data = {
                "summary": summary,
                "detailed_results": test_results
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print(f"\n评估结果已保存到: {output_file}")
        
        print("\n评估汇总:")
        print(f"  平均综合评分: {summary['average_score']}")
        print(f"  平均精确率(P): {summary['average_precision']}")
        print(f"  平均召回率(R): {summary['average_recall']}")
        print(f"  平均F1分数: {summary['average_f1']}")
        
        return summary


if __name__ == "__main__":
    # 测试评估器
    evaluator = EvaluatorAgent()
    
    # 示例数据
    raw_text = "今天在深圳参加了未来科技论坛，遇到了张悦和王伟。张悦分享了关于大模型的技术见解。"
    
    extracted_json = [
        {"temp_id": "1", "type": "地点", "content": "深圳"},
        {"temp_id": "2", "type": "事件", "content": "未来科技论坛"},
        {"temp_id": "3", "type": "人物", "content": "张悦"},
        {"temp_id": "4", "type": "人物", "content": "王伟"},
        {"temp_id": "5", "type": "灵感", "content": "大模型技术很有前景", "related_ids": ["3"]},
        {"temp_id": "6", "type": "线索", "content": "张悦在未来科技论坛上分享技术见解", "related_ids": ["2", "3"]}
    ]
    
    preset_entities = {
        "人物": ["张悦", "王伟"],
        "地点": ["深圳"],
        "事件": ["未来科技论坛"]
    }
    
    print("=" * 50)
    print("测试判别 Agent:")
    print("=" * 50)
    
    result = evaluator.evaluate(raw_text, extracted_json, preset_entities)
    print(f"\n综合评分: {result['overall_score']}")
    print(f"实体精确率: {result['entity_metrics']['precision']}")
    print(f"实体召回率: {result['entity_metrics']['recall']}")
    print(f"关系准确率: {result['relation_metrics']['relation_accuracy']}")
    print(f"完整性评分: {result['completeness_metrics']['completeness_score']}")
