# -*- coding: utf-8 -*-
"""
Adapter Registry - 适配器注册表

管理所有模型适配器的注册和获取。
"""

from typing import Dict, Type, Optional, List
import logging

from .base import ModelAdapter

logger = logging.getLogger(__name__)

# 全局适配器注册表
_ADAPTERS: Dict[str, Type[ModelAdapter]] = {}


def register_adapter(name: str):
    """
    装饰器：注册适配器
    
    Usage:
        @register_adapter("zimage")
        class ZImageAdapter(ModelAdapter):
            ...
    """
    def decorator(cls: Type[ModelAdapter]):
        if name in _ADAPTERS:
            logger.warning(f"Adapter '{name}' already registered, overwriting.")
        _ADAPTERS[name] = cls
        logger.debug(f"Registered adapter: {name}")
        return cls
    return decorator


def get_adapter(name: str, **kwargs) -> ModelAdapter:
    """
    获取适配器实例
    
    Args:
        name: 适配器名称（支持别名）
        **kwargs: 传递给适配器构造函数的参数
        
    Returns:
        适配器实例
        
    Raises:
        ValueError: 如果适配器不存在
    """
    # 名称标准化和别名映射
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    
    aliases = {
        "zimage": "zimage",
        "z_image": "zimage",
        "z_image_turbo": "zimage",
        "zimage_turbo": "zimage",
        "longcat": "longcat",
        "longcat_image": "longcat",
        "long_cat": "longcat",
    }
    
    resolved_name = aliases.get(name_lower, name_lower)
    
    if resolved_name not in _ADAPTERS:
        available = list_adapters()
        raise ValueError(
            f"Unknown adapter: '{name}'. Available adapters: {available}"
        )
    
    return _ADAPTERS[resolved_name](**kwargs)


def list_adapters() -> List[str]:
    """
    列出所有已注册的适配器
    
    Returns:
        适配器名称列表
    """
    return list(_ADAPTERS.keys())


def auto_detect_adapter(model_path: str) -> Optional[str]:
    """
    根据模型路径自动检测适配器类型
    
    Args:
        model_path: 模型路径
        
    Returns:
        适配器名称，无法检测时返回 None
    """
    import os
    from pathlib import Path
    
    path = Path(model_path)
    
    # 检查目录名或文件名中的关键词
    path_lower = str(path).lower()
    
    if "zimage" in path_lower or "z-image" in path_lower:
        return "zimage"
    
    if "longcat" in path_lower or "long_cat" in path_lower:
        return "longcat"
    
    if "flux" in path_lower:
        return "longcat"  # LongCat 基于 FLUX 架构
    
    # 检查配置文件
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.exists():
            import json
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # 根据配置内容判断
                model_type = config.get("_class_name", "").lower()
                
                if "zimage" in model_type:
                    return "zimage"
                elif "longcat" in model_type or "flux" in model_type:
                    return "longcat"
            except Exception:
                pass
    
    logger.warning(f"Could not auto-detect adapter for: {model_path}")
    return None


# 延迟导入适配器实现（避免循环导入）
def _ensure_adapters_loaded():
    """确保所有适配器已加载"""
    if not _ADAPTERS:
        from . import zimage_adapter  # noqa: F401
        from . import longcat_adapter  # noqa: F401


# 在模块加载时自动加载适配器
try:
    _ensure_adapters_loaded()
except ImportError:
    pass  # 适配器文件可能还不存在

