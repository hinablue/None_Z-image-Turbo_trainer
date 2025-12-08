# -*- coding: utf-8 -*-
"""
Model Adapters for Multi-Model Training Support.

支持的模型:
- Z-Image (Turbo)
- LongCat-Image
- 更多模型待扩展...
"""

from .base import ModelAdapter
from .registry import get_adapter, register_adapter, list_adapters

__all__ = [
    "ModelAdapter",
    "get_adapter",
    "register_adapter", 
    "list_adapters",
]

