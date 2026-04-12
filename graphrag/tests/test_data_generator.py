"""
模拟数据集生成器 - 用于生成测试数据
基于论文 5.1 节描述的测试数据集构建方法
"""
import json
import random
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from graphrag.models.llm import LLMModel


class TestDataGenerator:
    """测试数据生成器 - 模拟不同生活场景的记忆数据"""
    
    def __init__(self):
        self.llm = LLMModel()
        
        # 场景池 - 多样性控制
        self.scenarios = {
            "工作会议": {
                "roles": ["产品经理", "技术负责人", "项目经理", "设计师", "数据分析师"],
                "locations": ["公司会议室", "线上会议室", "咖啡厅", "共享办公空间"],
                "requirements": "必须提及至少2个同事姓名、1个会议地点，包含一个项目决策和一个后续待办事项"
            },
            "家庭聚餐": {
                "roles": ["父亲", "母亲", "哥哥", "姐姐", "表弟", "姨妈"],
                "locations": ["家里", "餐厅", "酒店", "公园", "老家"],
                "requirements": "必须提及至少2个家庭成员姓名、1个聚餐地点，包含一段家庭回忆和一个未来计划"
            },
            "技术灵感": {
                "roles": ["导师", "同学", "技术博主", "开源社区成员"],
                "locations": ["实验室", "图书馆", "技术沙龙", "线上论坛", "创业孵化器"],
                "requirements": "必须提及至少2个人物姓名、1个地点，包含一个技术创新想法和一个实践计划"
            },
            "情绪碎片": {
                "roles": ["心理咨询师", "好友", "室友", "前任"],
                "locations": ["宿舍", "操场", "天台", "深夜食堂", "地铁站"],
                "requirements": "必须提及至少2个人物姓名、1个地点，包含一段情绪表达和一个自我反思"
            },
            "学习笔记": {
                "roles": ["教授", "助教", "学霸同学", "图书馆管理员"],
                "locations": ["教室", "图书馆", "自习室", "实验室", "线上课堂"],
                "requirements": "必须提及至少2个人物姓名、1个学习地点，包含一个知识要点和一个复习提醒"
            }
        }
        
        # 人物姓名池
        self.names_pool = [
            "张悦", "王伟", "李明", "刘洋", "陈静", "赵强", "孙丽", "周杰",
            "吴敏", "郑华", "黄磊", "林峰", "徐雪", "马云", "朱迪", "胡军",
            "郭芳", "何平", "高峰", "罗丹", "梁宇", "宋佳", "曹阳", "彭丽",
            "于洋", "董洁", "袁弘", "蒋欣", "蔡明", "贾玲"
        ]
        
        # 地点池
        self.locations_pool = [
            "深圳", "北京", "上海", "杭州", "广州", "成都", "西安", "武汉",
            "清华大学", "北京大学", "浙江大学", "深圳大学",
            "科技园", "创业大街", "孵化器", "创新中心",
            "咖啡厅", "书店", "图书馆", "实验室", "会议室"
        ]
    
    def generate_single(self, scenario_type=None):
        """
        生成单条模拟数据
        
        Args:
            scenario_type: 指定场景类型，如果为None则随机选择
            
        Returns:
            dict: 包含原始文本和元数据
        """
        # 随机选择场景
        if scenario_type is None:
            scenario_type = random.choice(list(self.scenarios.keys()))
        
        scenario = self.scenarios[scenario_type]
        
        # 构建提示词
        prompt = self._build_prompt(scenario_type, scenario)
        
        # 调用LLM生成文本
        try:
            generated_text = self.llm.chat(prompt)
            
            # 提取预设的实体信息（用于后续评估对比）
            preset_entities = self._extract_preset_entities(generated_text)
            
            return {
                "id": f"test_{random.randint(10000, 99999)}",
                "scenario_type": scenario_type,
                "raw_text": generated_text,
                "preset_entities": preset_entities,
                "complexity_level": self._calculate_complexity(generated_text),
                "metadata": {
                    "word_count": len(generated_text),
                    "entity_count": len(preset_entities.get("人物", [])) + len(preset_entities.get("地点", []))
                }
            }
        except Exception as e:
            print(f"生成数据失败: {e}")
            return None
    
    def generate_batch(self, count=100, output_file=None):
        """
        批量生成测试数据
        
        Args:
            count: 生成数量
            output_file: 输出文件路径
            
        Returns:
            list: 生成的数据列表
        """
        print(f"开始生成 {count} 条测试数据...")
        
        test_data = []
        scenario_types = list(self.scenarios.keys())
        
        for i in range(count):
            # 轮询选择场景类型以保证多样性
            scenario_type = scenario_types[i % len(scenario_types)]
            
            data = self.generate_single(scenario_type)
            if data:
                test_data.append(data)
                print(f"  已生成 {i+1}/{count}: {data['id']} [{scenario_type}]")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            print(f"\n数据已保存到: {output_file}")
        
        print(f"成功生成 {len(test_data)} 条测试数据")
        return test_data
    
    def _build_prompt(self, scenario_type, scenario):
        """构建生成提示词"""
        # 随机选择角色和地点
        roles = random.sample(scenario["roles"], min(2, len(scenario["roles"])))
        location = random.choice(scenario["locations"])
        names = random.sample(self.names_pool, 2)
        
        prompt = f"""请扮演一名物联网专业的大学生，写一段 200-300 字的日常记事。

场景类型：{scenario_type}
要求：
1. {scenario["requirements"]}
2. 涉及的人物姓名：{names[0]}、{names[1]}
3. 地点：{location}
4. 内容要自然真实，像真实的日记或笔记
5. 可以适当包含一些口语化表达和情感色彩

请直接输出记事内容，不要包含标题或其他说明文字。"""
        
        return prompt
    
    def _extract_preset_entities(self, text):
        """从生成的文本中提取预设实体（用于黄金标准对比）"""
        entities = {
            "人物": [],
            "地点": [],
            "事件": [],
            "时间": []
        }
        
        # 简单规则提取（实际可以使用NER模型）
        # 提取人物姓名（常见中文姓名模式）
        for name in self.names_pool:
            if name in text:
                entities["人物"].append(name)
        
        # 提取地点
        for location in self.locations_pool:
            if location in text:
                entities["地点"].append(location)
        
        # 去重
        entities["人物"] = list(set(entities["人物"]))
        entities["地点"] = list(set(entities["地点"]))
        
        return entities
    
    def _calculate_complexity(self, text):
        """计算文本复杂度"""
        word_count = len(text)
        entity_count = len(self._extract_preset_entities(text)["人物"]) + \
                      len(self._extract_preset_entities(text)["地点"])
        
        # 复杂度分层
        if word_count > 250 and entity_count >= 4:
            return "high"
        elif word_count > 150 and entity_count >= 2:
            return "medium"
        else:
            return "low"


if __name__ == "__main__":
    # 测试生成器
    generator = TestDataGenerator()
    
    # 生成单条示例
    print("=" * 50)
    print("生成单条示例数据:")
    print("=" * 50)
    sample = generator.generate_single("工作会议")
    if sample:
        print(f"\n场景类型: {sample['scenario_type']}")
        print(f"复杂度: {sample['complexity_level']}")
        print(f"\n原始文本:\n{sample['raw_text']}")
        print(f"\n预设实体: {json.dumps(sample['preset_entities'], ensure_ascii=False)}")
    
    # 批量生成
    print("\n" + "=" * 50)
    print("批量生成测试数据:")
    print("=" * 50)
    
    output_path = os.path.join(os.path.dirname(__file__), "test_data.json")
    test_data = generator.generate_batch(count=10, output_file=output_path)
