#!/usr/bin/env python3
"""
GPU 检测脚本（用于 shell 脚本）
支持 CUDA、MPS 和 CPU
"""
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, "webui-vue", "api"))

try:
    from core.gpu_info import get_gpu_info_simple

    gpu_info = get_gpu_info_simple()

    if gpu_info["available"]:
        print(f"{gpu_info['name']}|{gpu_info['memory']}|{gpu_info['type']}")
        sys.exit(0)
    else:
        print("CPU|N/A|cpu")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    print("CPU|N/A|cpu")
    sys.exit(1)

