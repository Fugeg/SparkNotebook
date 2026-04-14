"""
Prompt 版本管理器
支持多版本 Prompt 的快速切换和 A/B 测试
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class PromptVersion:
    """Prompt 版本定义"""
    name: str
    file: str
    is_stable: bool = False
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PromptVersionManager:
    """Prompt 版本管理器"""
    
    def __init__(self, config_path: str = "harness/config/harness.yaml"):
        """
        初始化版本管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.versions: Dict[str, PromptVersion] = {}
        self.active_version: str = "v1"
        self.ab_test_enabled: bool = False
        self.v2_traffic_percentage: int = 5
        
        self._load_config()
        self._load_prompts()
    
    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            print(f"⚠️  配置文件不存在: {self.config_path}")
            return
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        deployment = config.get("deployment", {})
        
        # 加载版本定义
        versions_config = deployment.get("prompt_versions", {})
        for version_id, version_data in versions_config.items():
            self.versions[version_id] = PromptVersion(
                name=version_data.get("name", version_id),
                file=version_data.get("file", f"prompts/{version_id}.txt"),
                is_stable=version_data.get("is_stable", False),
                description=version_data.get("description", "")
            )
        
        # 加载当前激活版本
        self.active_version = deployment.get("active_version", "v1")
        
        # 加载 A/B 测试配置
        ab_test = deployment.get("ab_test", {})
        self.ab_test_enabled = ab_test.get("enabled", False)
        self.v2_traffic_percentage = ab_test.get("v2_traffic_percentage", 5)
        
        print(f"✅ 已加载 {len(self.versions)} 个 Prompt 版本")
    
    def _load_prompts(self):
        """加载所有 Prompt 文件内容"""
        for version_id, version in self.versions.items():
            prompt_path = Path(version.file)
            if prompt_path.exists():
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    version.metadata["content"] = f.read()
            else:
                print(f"⚠️  Prompt 文件不存在: {prompt_path}")
                version.metadata["content"] = ""
    
    def get_prompt(self, version: Optional[str] = None) -> str:
        """
        获取指定版本的 Prompt
        
        Args:
            version: 版本 ID，None 则使用当前激活版本
        
        Returns:
            Prompt 内容
        """
        version_id = version or self.active_version
        
        if version_id not in self.versions:
            raise ValueError(f"未知的版本: {version_id}")
        
        return self.versions[version_id].metadata.get("content", "")
    
    def switch_version(self, version: str) -> bool:
        """
        切换到指定版本
        
        Args:
            version: 目标版本 ID
        
        Returns:
            是否切换成功
        """
        if version not in self.versions:
            print(f"❌ 未知版本: {version}")
            return False
        
        old_version = self.active_version
        self.active_version = version
        
        # 更新配置文件
        self._save_config()
        
        print(f"✅ 已切换版本: {old_version} → {version}")
        print(f"   版本名称: {self.versions[version].name}")
        print(f"   稳定状态: {'✅ 稳定' if self.versions[version].is_stable else '⚠️ 实验'}")
        
        return True
    
    def get_version_for_request(self, request_id: Optional[str] = None) -> str:
        """
        根据 A/B 测试策略获取版本
        
        Args:
            request_id: 请求 ID，用于一致性哈希
        
        Returns:
            版本 ID
        """
        if not self.ab_test_enabled:
            return self.active_version
        
        # 简单的流量切分
        import hashlib
        if request_id:
            hash_val = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        else:
            import random
            hash_val = random.randint(0, 99)
        
        # 根据百分比决定版本
        if (hash_val % 100) < self.v2_traffic_percentage:
            return "v2" if "v2" in self.versions else self.active_version
        else:
            return "v1" if "v1" in self.versions else self.active_version
    
    def list_versions(self) -> Dict[str, Dict[str, Any]]:
        """列出所有版本信息"""
        return {
            vid: {
                "name": v.name,
                "is_stable": v.is_stable,
                "description": v.description,
                "is_active": vid == self.active_version
            }
            for vid, v in self.versions.items()
        }
    
    def create_version(self, 
                       version_id: str, 
                       name: str,
                       content: str,
                       description: str = "",
                       is_stable: bool = False) -> bool:
        """
        创建新版本
        
        Args:
            version_id: 版本 ID
            name: 版本名称
            content: Prompt 内容
            description: 版本描述
            is_stable: 是否稳定版本
        
        Returns:
            是否创建成功
        """
        if version_id in self.versions:
            print(f"❌ 版本已存在: {version_id}")
            return False
        
        # 保存 Prompt 文件
        prompt_file = f"prompts/{version_id}.txt"
        prompt_path = Path(prompt_file)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # 创建版本对象
        self.versions[version_id] = PromptVersion(
            name=name,
            file=prompt_file,
            is_stable=is_stable,
            description=description,
            metadata={"content": content}
        )
        
        # 保存配置
        self._save_config()
        
        print(f"✅ 已创建新版本: {version_id}")
        return True
    
    def update_version_content(self, version_id: str, content: str) -> bool:
        """更新版本内容"""
        if version_id not in self.versions:
            print(f"❌ 版本不存在: {version_id}")
            return False
        
        version = self.versions[version_id]
        version.metadata["content"] = content
        
        # 保存文件
        with open(version.file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已更新版本内容: {version_id}")
        return True
    
    def enable_ab_test(self, v2_percentage: int = 5):
        """启用 A/B 测试"""
        self.ab_test_enabled = True
        self.v2_traffic_percentage = v2_percentage
        self._save_config()
        print(f"✅ A/B 测试已启用: v2 流量占比 {v2_percentage}%")
    
    def disable_ab_test(self):
        """禁用 A/B 测试"""
        self.ab_test_enabled = False
        self._save_config()
        print("✅ A/B 测试已禁用")
    
    def _save_config(self):
        """保存配置到文件"""
        # 读取现有配置
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # 更新 deployment 部分
        config["deployment"] = {
            "prompt_versions": {
                vid: {
                    "name": v.name,
                    "file": v.file,
                    "is_stable": v.is_stable,
                    "description": v.description
                }
                for vid, v in self.versions.items()
            },
            "active_version": self.active_version,
            "ab_test": {
                "enabled": self.ab_test_enabled,
                "v2_traffic_percentage": self.v2_traffic_percentage
            }
        }
        
        # 保存
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    def print_status(self):
        """打印当前状态"""
        print(f"\n{'='*60}")
        print("📋 Prompt 版本管理状态")
        print(f"{'='*60}")
        print(f"当前激活版本: {self.active_version}")
        print(f"A/B 测试: {'✅ 启用' if self.ab_test_enabled else '❌ 禁用'}")
        if self.ab_test_enabled:
            print(f"  v2 流量占比: {self.v2_traffic_percentage}%")
        print(f"\n可用版本:")
        for vid, info in self.list_versions().items():
            active = "👉 " if info["is_active"] else "   "
            stable = "✅" if info["is_stable"] else "⚠️"
            print(f"  {active}{vid}: {info['name']} [{stable}]")
            if info["description"]:
                print(f"      {info['description']}")
        print(f"{'='*60}\n")


# 全局实例（单例模式）
_prompt_manager: Optional[PromptVersionManager] = None


def get_prompt_manager(config_path: str = "harness/config/harness.yaml") -> PromptVersionManager:
    """获取 Prompt 版本管理器单例"""
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptVersionManager(config_path)
    return _prompt_manager
