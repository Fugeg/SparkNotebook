"""
Checkpointer - 状态检查点
支持 Agent 执行状态的持久化和断点续传
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from abc import ABC, abstractmethod


class BaseCheckpointer(ABC):
    """Checkpointer 基类"""
    
    @abstractmethod
    def save(self, checkpoint_id: str, state: Dict[str, Any]) -> bool:
        """保存状态"""
        pass
    
    @abstractmethod
    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """加载状态"""
        pass
    
    @abstractmethod
    def list_checkpoints(self) -> list:
        """列出所有检查点"""
        pass
    
    @abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        pass


class FileCheckpointer(BaseCheckpointer):
    """文件存储 Checkpointer"""
    
    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """
        初始化文件 Checkpointer
        
        Args:
            checkpoint_dir: 检查点存储目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, checkpoint_id: str, state: Dict[str, Any]) -> bool:
        """
        保存状态到文件
        
        Args:
            checkpoint_id: 检查点 ID
            state: 状态字典
        
        Returns:
            是否保存成功
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            
            # 添加元数据
            checkpoint_data = {
                "checkpoint_id": checkpoint_id,
                "timestamp": datetime.now().isoformat(),
                "state": state
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"❌ 保存检查点失败: {e}")
            return False
    
    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        从文件加载状态
        
        Args:
            checkpoint_id: 检查点 ID
        
        Returns:
            状态字典，不存在则返回 None
        """
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            return checkpoint_data.get("state")
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            return None
    
    def list_checkpoints(self) -> list:
        """列出所有检查点"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                checkpoints.append({
                    "checkpoint_id": data.get("checkpoint_id"),
                    "timestamp": data.get("timestamp"),
                    "file": str(checkpoint_file)
                })
            except:
                pass
        
        # 按时间排序
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def delete(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            return True
        return False
    
    def get_latest(self, prefix: str = "") -> Optional[Dict[str, Any]]:
        """
        获取最新的检查点
        
        Args:
            prefix: ID 前缀过滤
        
        Returns:
            最新的状态字典
        """
        checkpoints = self.list_checkpoints()
        
        for cp in checkpoints:
            if not prefix or cp["checkpoint_id"].startswith(prefix):
                return self.load(cp["checkpoint_id"])
        
        return None


class MemoryCheckpointer(BaseCheckpointer):
    """内存存储 Checkpointer（适用于测试）"""
    
    def __init__(self):
        """初始化内存 Checkpointer"""
        self._storage: Dict[str, Dict[str, Any]] = {}
    
    def save(self, checkpoint_id: str, state: Dict[str, Any]) -> bool:
        """保存状态到内存"""
        self._storage[checkpoint_id] = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "state": state
        }
        return True
    
    def load(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """从内存加载状态"""
        data = self._storage.get(checkpoint_id)
        return data["state"] if data else None
    
    def list_checkpoints(self) -> list:
        """列出所有检查点"""
        return [
            {
                "checkpoint_id": data["checkpoint_id"],
                "timestamp": data["timestamp"]
            }
            for data in self._storage.values()
        ]
    
    def delete(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        if checkpoint_id in self._storage:
            del self._storage[checkpoint_id]
            return True
        return False
    
    def clear(self):
        """清空所有检查点"""
        self._storage.clear()


class AgentStateManager:
    """
    Agent 状态管理器
    简化 Checkpointer 的使用
    """
    
    def __init__(self, checkpointer: Optional[BaseCheckpointer] = None):
        """
        初始化状态管理器
        
        Args:
            checkpointer: Checkpointer 实例，None 则使用文件存储
        """
        self.checkpointer = checkpointer or FileCheckpointer()
        self.current_checkpoint_id: Optional[str] = None
    
    def checkpoint(self, 
                   task_id: str, 
                   step: int, 
                   state: Dict[str, Any]) -> bool:
        """
        创建检查点
        
        Args:
            task_id: 任务 ID
            step: 当前步骤
            state: 状态数据
        
        Returns:
            是否成功
        """
        checkpoint_id = f"{task_id}_step_{step}"
        
        # 添加步骤信息
        checkpoint_state = {
            "task_id": task_id,
            "step": step,
            "data": state,
            "checkpoint_time": datetime.now().isoformat()
        }
        
        success = self.checkpointer.save(checkpoint_id, checkpoint_state)
        
        if success:
            self.current_checkpoint_id = checkpoint_id
            print(f"✅ 检查点已保存: {checkpoint_id}")
        
        return success
    
    def resume(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        恢复任务状态
        
        Args:
            task_id: 任务 ID
        
        Returns:
            最新的状态数据
        """
        # 查找该任务的最新检查点
        checkpoints = self.checkpointer.list_checkpoints()
        
        task_checkpoints = [
            cp for cp in checkpoints 
            if cp["checkpoint_id"].startswith(f"{task_id}_step_")
        ]
        
        if not task_checkpoints:
            print(f"⚠️  未找到任务检查点: {task_id}")
            return None
        
        # 获取最新的
        latest = task_checkpoints[0]
        state = self.checkpointer.load(latest["checkpoint_id"])
        
        if state:
            self.current_checkpoint_id = latest["checkpoint_id"]
            step = state.get("step", 0)
            print(f"✅ 已从步骤 {step} 恢复任务: {task_id}")
        
        return state
    
    def get_progress(self, task_id: str) -> int:
        """
        获取任务进度（最后保存的步骤）
        
        Args:
            task_id: 任务 ID
        
        Returns:
            最后保存的步骤号，无检查点则返回 0
        """
        state = self.resume(task_id)
        return state.get("step", 0) if state else 0


# 便捷函数
def create_checkpointer(checkpoint_type: str = "file", **kwargs) -> BaseCheckpointer:
    """
    创建 Checkpointer
    
    Args:
        checkpoint_type: 类型 (file, memory)
        **kwargs: 其他参数
    
    Returns:
        Checkpointer 实例
    """
    if checkpoint_type == "file":
        return FileCheckpointer(**kwargs)
    elif checkpoint_type == "memory":
        return MemoryCheckpointer()
    else:
        raise ValueError(f"未知的 Checkpointer 类型: {checkpoint_type}")
