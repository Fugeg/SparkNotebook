"""
StepFun API (step-3.5-flash) 响应时间测试脚本

测试内容：
1. 简单对话响应时间
2. 查询翻译响应时间
3. 技术灵感报告生成响应时间
"""
import os
import time
import statistics
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graphrag.utils.stepfun_client import StepFunClient


class StepFunTimingTest:
    """StepFun API 响应时间测试类"""
    
    def __init__(self):
        self.client = StepFunClient()
        # 强制使用 step-3.5-flash 模型
        self.client.model = 'step-3.5-flash'
        self.results = []
        
    def check_availability(self):
        """检查 API 是否可用"""
        if not self.client.is_available():
            print("❌ 错误: STEPFUN_API_KEY 未设置，请先配置环境变量")
            print("   当前模型:", self.client.model)
            return False
        print(f"✅ StepFun API 已配置")
        print(f"   模型: {self.client.model}")
        print(f"   Base URL: {self.client.base_url}")
        return True
    
    def test_simple_chat(self, iterations=3):
        """测试简单对话响应时间"""
        print(f"\n{'='*60}")
        print(f"测试 1: 简单对话响应时间 ({iterations} 次)")
        print(f"{'='*60}")
        
        prompt = "你好，请简单介绍一下自己"
        system_prompt = "你是一个AI助手，请简洁回答"
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            try:
                response = self.client.chat(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=0.7,
                    max_tokens=500
                )
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                
                print(f"  第 {i+1} 次:")
                print(f"    响应时间: {elapsed:.2f} 秒")
                print(f"    回复长度: {len(response)} 字符")
                print(f"    首段回复: {response[:80]}...")
                
            except Exception as e:
                print(f"  第 {i+1} 次: ❌ 错误 - {e}")
        
        if times:
            self._print_stats("简单对话", times)
        return times
    
    def test_translation(self, iterations=3):
        """测试查询翻译响应时间"""
        print(f"\n{'='*60}")
        print(f"测试 2: 查询翻译响应时间 ({iterations} 次)")
        print(f"{'='*60}")
        
        test_queries = [
            "找一个关于图像识别的项目",
            "推荐一个LoRa通信的开源项目",
            "有关机器学习框架的项目"
        ]
        
        times = []
        for i in range(min(iterations, len(test_queries))):
            query = test_queries[i]
            start_time = time.time()
            try:
                result = self.client.translate_and_expand_query(query)
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                
                print(f"  第 {i+1} 次:")
                print(f"    输入: {query}")
                print(f"    输出: {result}")
                print(f"    响应时间: {elapsed:.2f} 秒")
                
            except Exception as e:
                print(f"  第 {i+1} 次: ❌ 错误 - {e}")
        
        if times:
            self._print_stats("查询翻译", times)
        return times
    
    def test_inspiration_report(self, iterations=1):
        """测试技术灵感报告生成响应时间"""
        print(f"\n{'='*60}")
        print(f"测试 3: 技术灵感报告生成响应时间 ({iterations} 次)")
        print(f"{'='*60}")
        
        user_query = "我想做一个图像识别的项目"
        github_context = """
Project: OpenCV
Stars: 50000
Description: Open Source Computer Vision Library
Language: C++/Python

Project: YOLO
Stars: 30000
Description: Real-time object detection
Language: Python
"""
        local_notes = "之前想过做一个水果识别的应用"
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            try:
                response = self.client.generate_inspiration_report(
                    user_query=user_query,
                    github_context=github_context,
                    local_notes=local_notes
                )
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                
                print(f"  第 {i+1} 次:")
                print(f"    响应时间: {elapsed:.2f} 秒")
                print(f"    报告长度: {len(response)} 字符")
                print(f"    报告预览:\n{response[:300]}...")
                
            except Exception as e:
                print(f"  第 {i+1} 次: ❌ 错误 - {e}")
        
        if times:
            self._print_stats("灵感报告生成", times)
        return times
    
    def test_different_token_lengths(self):
        """测试不同 token 长度下的响应时间"""
        print(f"\n{'='*60}")
        print(f"测试 4: 不同输出长度响应时间对比")
        print(f"{'='*60}")
        
        max_tokens_list = [100, 500, 1000, 2000]
        prompt = "请介绍一下Python编程语言的特点"
        
        results = []
        for max_tokens in max_tokens_list:
            print(f"\n  max_tokens={max_tokens}:")
            start_time = time.time()
            try:
                response = self.client.chat(
                    prompt=prompt,
                    max_tokens=max_tokens
                )
                elapsed = time.time() - start_time
                results.append({
                    'max_tokens': max_tokens,
                    'time': elapsed,
                    'length': len(response)
                })
                print(f"    响应时间: {elapsed:.2f} 秒")
                print(f"    输出长度: {len(response)} 字符")
            except Exception as e:
                print(f"    ❌ 错误: {e}")
        
        return results
    
    def _print_stats(self, test_name, times):
        """打印统计信息"""
        print(f"\n  📊 {test_name} 统计:")
        print(f"    平均响应时间: {statistics.mean(times):.2f} 秒")
        print(f"    最小响应时间: {min(times):.2f} 秒")
        print(f"    最大响应时间: {max(times):.2f} 秒")
        if len(times) > 1:
            print(f"    标准差: {statistics.stdev(times):.2f} 秒")
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"\n{'#'*60}")
        print(f"# StepFun API (step-3.5-flash) 响应时间测试")
        print(f"# 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*60}")
        
        if not self.check_availability():
            return
        
        # 运行各项测试
        chat_times = self.test_simple_chat(iterations=3)
        translation_times = self.test_translation(iterations=3)
        report_times = self.test_inspiration_report(iterations=1)
        length_results = self.test_different_token_lengths()
        
        # 汇总报告
        print(f"\n{'='*60}")
        print(f"测试汇总报告")
        print(f"{'='*60}")
        
        all_times = chat_times + translation_times + report_times
        if all_times:
            print(f"\n总体统计:")
            print(f"  总请求数: {len(all_times)}")
            print(f"  平均响应时间: {statistics.mean(all_times):.2f} 秒")
            print(f"  最小响应时间: {min(all_times):.2f} 秒")
            print(f"  最大响应时间: {max(all_times):.2f} 秒")
            print(f"  总耗时: {sum(all_times):.2f} 秒")
        
        print(f"\n{'='*60}")
        print(f"测试完成!")
        print(f"{'='*60}")


if __name__ == "__main__":
    test = StepFunTimingTest()
    test.run_all_tests()
