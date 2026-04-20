import os
import dashscope
import requests
from http import HTTPStatus
from dotenv import load_dotenv
from http import HTTPStatus

load_dotenv()

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY')

class LLMModel:
    def __init__(self):
        self.model = 'qwen-plus-latest'
    
    def classify_intent(self, text):
        """意图分类：1-闲聊 2-记录新记忆 3-搜索记忆 4-GitHub灵感"""
        # 系统提示词
        system_prompt = """# Role
你是一个高精度的意图分类引擎。分析用户输入，准确分类到以下四个意图之一。

# Intent Definitions (意图定义)

## 1 (闲聊)
- 日常问候、打招呼、自我介绍
- 情感表达、闲聊
- 通用知识问答
- 简单的陈述，**没有明确要求记住**

## 2 (记录新记忆)
- 用户**明确要求记录**：包含"记住"、"记下"、"提醒我"等关键词
- 重要的个人信息：生日、地址、密码等
- 待办事项、计划、日程
- 重要的喜好、偏好（明确说"我不喜欢/喜欢..."）

## 3 (搜索记忆)
- 提问关于过去的信息
- "你还记得...吗"、"我之前说过..."
- 询问自己的喜好、计划等

## 4 (GitHub灵感)
- 询问 GitHub 上的开源项目、技术趋势
- 请求基于 GitHub 项目给出技术建议或灵感
- 涉及代码实现思路、技术选型讨论
- 关键词：GitHub、开源项目、代码参考、技术方案、LoRa、网关、AI项目等

# Examples (示例)

## 闲聊 (1)
[输入]: "早上好"
[输出]: 1

[输入]: "你好，我是小明"
[输出]: 1

[输入]: "今天天气不错"
[输出]: 1

[输入]: "帮我写个Python排序"
[输出]: 1

## 记录新记忆 (2)
[输入]: "我下周三要去北京出差，帮我记一下"
[输出]: 2

[输入]: "记住我的生日是10月5日"
[输出]: 2

## 搜索记忆 (3)
[输入]: "你还记得我最喜欢的颜色吗"
[输出]: 3

[输入]: "我生日是什么时候"
[输出]: 3

## GitHub灵感 (4)
[输入]: "我想做一个基于LoRa的网关，GitHub有什么参考项目？"
[输出]: 4

[输入]: "推荐一些好用的AI开源工具"
[输出]: 4

[输入]: "我想开发一个RAG系统，有什么开源方案可以参考？"
[输出]: 4

[输入]: "GitHub上有哪些热门的知识图谱项目？"
[输出]: 4

# Rules (规则)
1. **关键区分**：是否有明确的"记录"意图（关键词：记住、记下、提醒、帮我记等）
2. **自我介绍**（"我是xxx"、"我叫xxx"）属于闲聊(1)，不是记录记忆(2)
3. **GitHub相关**的技术咨询、开源项目推荐属于(4)
4. 输出格式：**仅输出数字（1、2、3 或 4）**，不要任何其他文字。"""

        # 用户输入
        user_prompt = f"用户输入：{text}\n\n请只返回一个数字（1、2、3 或 4）："
        
        try:
            # 使用 messages 格式调用，支持系统提示词
            response = dashscope.Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.01,
                max_tokens=10
            )
            intent = response.output.text.strip()
            if intent in ["1", "2", "3"]:
                return int(intent)
            return 1
        except Exception as e:
            print(f"分类意图失败: {e}")
            return 1
    
    def extract_information(self, text):
        prompt = f"""**背景:** 你是 AI 记事本的"首席信息分析师"。你的任务是将用户输入的、非结构化的、混乱的自然语言信息，精确地解析和拆解成结构化的、相互关联的【信息单元】。

**绝对禁止（反幻觉规则）:**
0.**在记事本模式,即(2,3)模式下,绝对不能编造、推测或扩展任何内容。**
1. **严禁编造**: 你只能提取用户输入中**明确存在**的信息，绝对不能编造、推测或扩展任何内容。
2. **禁止想象**: 如果用户没有提到某个细节，你不能"脑补"或"合理推测"。
3. **禁止总结不存在的内容**: 不要生成用户没有说过的"核心主张"、"落地实践"、"金句"等。
4. **只提取事实**: 只提取用户明确陈述的事实，不要添加你的解释或分析。

**任务:** 将用户输入的一段话，转换成一个包含多种信息单元的 JSON 数组。

**第一步：识别核心 [直接信息单位]**
从原文中提取直接信息单位，**只提取明确存在的**。
* `经历`: 用户明确描述的发生的事情。
* `灵感`: 用户明确提到的想法或解决方案。
* `提醒`: 用户明确提到的待办事项或目标。
* `闲绪`: 用户明确表达的情绪或思绪。

**第二步：提取所有 [间接信息单位]**
在原文中找出**明确提到的**可命名实体。
* `人物`: 用户明确提到的人名。
* `事件`: 用户明确提到的活动或事件名称。
* `地点`: 用户明确提到的地理位置。

**第三步：构建所有 [信息链接单位]**
基于前两步提取的单元，创建链接描述**明确存在的关系**。
* `关系`: 用户明确描述的人与人之间的关系。
* `线索`: 用户明确描述的单元之间的连接。

**输入输出格式:**
* **输入**: "{text}"
* **输出**: 严格遵循以下格式的 JSON 数组。必须为每个单元生成 `temp_id`，`related_ids` 只引用明确存在的关系。

```json
[
  {{
    "temp_id": "1",
    "type": "经历",
    "content": "用户明确描述的内容片段",
  }},
  {{
    "temp_id": "2",
    "type": "人物",
    "content": "用户明确提到的人名",
  }},
  {{
    "temp_id": "3",
    "type": "关系",
    "content": "用户明确描述的关系",
    "related_ids": ["2"]
  }}
]
```

**重要提醒:**
* **只提取原文**: `content` 必须是用户原文中的内容或准确摘要，不能添加你自己的内容。
* **禁止扩展**: 不要生成用户没有明确提到的信息。
* **禁止推测**: 不要推测用户的意图或补充细节。

请直接输出 JSON 数组，不要包含其他解释性文字。"""
        try:
            response = dashscope.Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=0,
                max_tokens=2000
            )
            return response.output.text
        except Exception as e:
            print(f"提取信息失败：{e}")
            return "[]"
    
    def evaluate_relevance(self, query, nodes):
        prompt = f"""请评估以下节点与查询的相关性，返回每个节点的相关性评分（0-10）：

查询：{query}

节点：
{chr(10).join([f"{i+1}. {node['type']}: {node['content']}" for i, node in enumerate(nodes)])}

返回格式：
节点1: 评分
节点2: 评分
..."""
        try:
            response = dashscope.Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=0,
                max_tokens=500
            )
            return response.output.text
        except Exception as e:
            print(f"评估相关性失败: {e}")
            return ""

    def estimate_required_tokens(self, query: str) -> int:
        """
        根据查询类型估算所需 Token 数量

        Args:
            query: 用户输入的查询

        Returns:
            建议的 max_tokens 值
        """
        LONG_FORM_KEYWORDS = [
            "作文", "文章", "报告", "总结", "详细说明", "详细描述",
            "800字", "1000字", "1500字", "2000字", "写一篇",
            "介绍一下", "详细介绍一下", "全面介绍",
            "分析一下", "详细分析", "深度分析",
            "对比一下", "详细对比", "对比分析"
        ]

        SHORT_FORM_KEYWORDS = [
            "是什么", "什么是", "多少", "几个",
            "谁", "哪里", "什么时候", "怎么", "如何"
        ]

        query_lower = query.lower()

        if any(kw in query for kw in LONG_FORM_KEYWORDS):
            if any(kw in query for kw in ["800字", "1000字", "1500字", "2000字"]):
                import re
                match = re.search(r'(\d+)字', query)
                if match:
                    char_count = int(match.group(1))
                    return min(int(char_count * 2.5), 4000)
            return 2000

        if any(kw in query for kw in SHORT_FORM_KEYWORDS):
            return 300

        return 500

    def generate_response(self, query, context, username="用户"):
        """基于检索到的上下文生成回答（搜索记忆模式 - 模式 2）"""
        max_tokens = self.estimate_required_tokens(query)

        system_prompt = f"""# Role
你是「AI 记事本助手」，{username} 的个人知识管理助手。

# 当前模式：搜索记忆 (模式 2)
用户正在提问过去的事情，需要你回忆知识图谱中的信息。

# 你的任务
基于注入的上下文片段，进行多跳推理，综合生成准确的答案。

# 回答策略
1. **直接给出答案** - 不要绕弯子
2. **延展提醒** - 如果知识图谱中有相关联的扩展信息，可以做有价值的提醒
3. **找不到时坦诚回答** - "在你的知识库中没有找到关于此的记录"，绝对不要编造

# 回答风格
- 简洁直接，像朋友一样对话
- 不要使用 JSON、代码块或技术性语言
- 第一人称：称呼用户为"你"，自称为"我"

# 绝对禁止
- 编造用户的个人信息（反幻觉）
- 使用通用预训练数据覆盖检索到的上下文"""

        # 构建上下文信息
        if context:
            context_text = "\n".join([f"- [{node.get('type', 'unknown')}] {node.get('content', '')}" for node in context[:10]])
        else:
            context_text = "（无相关上下文）"

        # 用户查询
        user_prompt = f"""用户问题：{query}

从知识图谱检索到的相关记忆：
{context_text}

请基于以上记忆回答用户的问题。如果记忆不足，请坦诚说明。"""

        try:
            response = dashscope.Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            return response.output.text
        except Exception as e:
            print(f"生成回答失败: {e}")
            return "抱歉，检索记忆时出现了错误。"
    
    def chat(self, text, username="用户"):
        """日常闲聊模式"""
        max_tokens = self.estimate_required_tokens(text)

        system_prompt = f"""# Role
你是「AI 记事本助手」，{username} 的个人 AI 伙伴。

# 你的任务
保持自然、友好、懂技术的交流，像朋友一样聊天。

# 回答策略
- **简洁明快** - 不要长篇大论
- **极客风格** - 符合开发者/技术人员的语境
- **自然对话** - 像朋友一样聊天，不要机械
- **不要提及模式** - 绝对不要告诉用户你处于什么模式或状态

# 回答风格
- 简短、直接、有温度
- 不要使用 JSON、代码块或技术性语言
- 第一人称：称呼用户为"你"，自称为"我"
- 如果用户问"你是什么模式"，回答"我就是你的 AI 伙伴，随时陪你聊天"，不要提技术细节

# 示例
用户："你好"
你："你好！有什么我可以帮你的？"

用户："今天天气不错"
你："是啊，适合出去转转或者写代码 😄"

用户："我是小明"
你："你好小明！很高兴认识你。"

用户："你现在是什么模式"
你："我就是你的 AI 伙伴，随时陪你聊天~有什么想说的，我都在这儿听着呢！"""

        # 添加用户身份说明（在 user_prompt 中）
        user_prompt = f"当前登录用户：{username}\n\n用户输入：{text}\n\n请回复："

        # 用户输入
        user_prompt = f"用户：{text}"

        try:
            response = dashscope.Generation.call(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=max_tokens
            )
            return response.output.text
        except Exception as e:
            print(f"聊天失败: {e}")
            return "抱歉，我现在有点累，请稍后再试。"

    def speech_to_text(self, audio_file_path):
        """语音转文本 (STT) - 使用录音文件识别，通过OSS上传获取URL"""
        print("=" * 50)
        print("开始语音转文本处理")
        print(f"输入音频文件: {audio_file_path}")
        
        import os
        oss_url = None
        converted_file = None
        try:
            import subprocess
            import wave
            from dashscope.audio.asr import Transcription
            from graphrag.utils.oss_helper import OSSHelper
            
            # 检查原始音频文件信息
            print(f"\n[1] 检查原始音频文件...")
            if os.path.exists(audio_file_path):
                file_size = os.path.getsize(audio_file_path)
                print(f"  文件大小: {file_size} bytes")
                
                # 获取音频基本信息
                try:
                    with wave.open(audio_file_path, 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        duration = frames / float(rate)
                        print(f"  音频格式: {channels}声道, {sampwidth}字节, {rate}Hz")
                        print(f"  时长: {duration:.2f}秒")
                except Exception as e:
                    print(f"  无法读取音频信息: {e}")
            else:
                print(f"  文件不存在!")
                return None
            
            # 2. 转换音频格式为16kHz采样率
            print(f"\n[2] 转换音频格式为16kHz...")
            converted_file = audio_file_path + ".converted.wav"
            cmd = f"ffmpeg -y -i {audio_file_path} -ar 16000 -ac 1 -acodec pcm_s16le {converted_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"  转换失败，使用原始文件: {result.stderr[:200]}")
                converted_file = audio_file_path
            else:
                conv_size = os.path.getsize(converted_file)
                print(f"  转换成功: {converted_file} ({conv_size} bytes)")
            
            # 3. 上传音频文件到OSS
            print(f"\n[3] 上传音频到OSS...")
            oss_helper = OSSHelper()
            oss_url = oss_helper.upload_file(converted_file)
            if oss_url:
                print(f"  上传成功!")
                print(f"  OSS URL: {oss_url[:80]}...")
            else:
                print(f"  上传失败!")
                return None
            
            # 4. 调用DashScope语音识别API
            print(f"\n[4] 调用DashScope语音识别...")
            print(f"  模型: paraformer-v2")
            print(f"  语言提示: zh, en")
            
            response = Transcription.call(
                model='paraformer-v2',
                file_urls=[oss_url],
                language_hints=['zh', 'en']
            )
            
            print(f"  API响应状态码: {response.status_code}")
            print(f"  API响应内容: {response.output}")
            
            if response.status_code == HTTPStatus.OK:
                task_id = response.output.task_id
                print(f"  任务ID: {task_id}")
                
                import time
                max_retries = 30
                print(f"\n[5] 等待语音识别完成...")
                for i in range(max_retries):
                    print(f"  检查进度... ({i+1}/{max_retries})")
                    result = Transcription.fetch(task_id)
                    if result.status_code == HTTPStatus.OK:
                        status = result.output.task_status
                        print(f"    任务状态: {status}")
                        
                        if status == 'COMPLETED' or status == 'SUCCEEDED':
                            print(f"\n[6] 任务完成，提取转写结果...")
                            results = result.output.results
                            print(f"    Results类型: {type(results)}")
                            print(f"    Results内容: {results}")
                            
                            if isinstance(results, dict):
                                transcription_url = results.get('transcription_url')
                            else:
                                transcription_url = results[0].get('transcription_url') if hasattr(results[0], 'transcription_url') else results[0]['transcription_url']
                            
                            print(f"    转写URL: {transcription_url}")
                            
                            import requests
                            trans_resp = requests.get(transcription_url)
                            print(f"    转写响应状态: {trans_resp.status_code}")
                            
                            if trans_resp.status_code == 200:
                                trans_data = trans_resp.json()
                                print(f"    转写JSON: {trans_data}")
                                
                                transcription = trans_data.get('transcription_text', '')
                                if not transcription:
                                    transcription = trans_data.get('text', '')
                                if not transcription:
                                    transcription = trans_data.get('sentences', [{}])[0].get('text', '') if trans_data.get('sentences') else ''
                                if not transcription:
                                    transcription = trans_data.get('results', [{}])[0].get('transcription_text', '') if trans_data.get('results') else ''
                                if not transcription:
                                    transcripts = trans_data.get('transcripts', [])
                                    if transcripts and len(transcripts) > 0:
                                        transcription = transcripts[0].get('text', '')
                                
                                print(f"\n===== 语音识别结果 =====")
                                print(transcription)
                                print("=" * 50)
                                return transcription
                            else:
                                print(f"    获取转写结果失败: HTTP {trans_resp.status_code}")
                                return None
                        elif status == 'FAILED':
                            error_msg = result.output.get('message', '未知错误')
                            error_code = result.output.get('code', '')
                            print(f"    任务失败!")
                            print(f"    错误代码: {error_code}")
                            print(f"    错误信息: {error_msg}")
                            
                            if 'DECODE_ERROR' in str(error_msg):
                                print("    原因: 音频文件无法解码")
                            elif 'NO_VALID_FRAGMENT' in str(error_msg):
                                print("    原因: 音频中没有检测到有效语音内容")
                            return None
                    else:
                        print(f"    检查失败: HTTP {result.status_code}")
                    time.sleep(1)
                
                print("  等待超时")
                return None
            else:
                print(f"  API调用失败!")
                print(f"  错误码: {response.code}")
                print(f"  错误信息: {response.message}")
                return None
                
        except Exception as e:
            print(f"语音转文本异常: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # 清理OSS临时文件
            if oss_url:
                try:
                    oss_helper.delete_file_from_url(oss_url)
                    print("\n[清理] OSS临时文件已删除")
                except Exception as e:
                    print(f"\n[清理] OSS文件删除失败: {e}")
            
            # 清理转换后的临时文件
            if converted_file and converted_file != audio_file_path:
                try:
                    if os.path.exists(converted_file):
                        os.remove(converted_file)
                        print("[清理] 转换临时文件已删除")
                except Exception as e:
                    print(f"[清理] 转换文件删除失败: {e}")
            
            print("=" * 50)
