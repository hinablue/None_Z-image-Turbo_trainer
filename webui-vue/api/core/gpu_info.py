"""
统一的 GPU 信息获取模块
支持 CUDA、MPS (Apple Silicon) 和 CPU
"""
import torch
import psutil
from typing import Dict, Any, List


def get_gpu_info() -> Dict[str, Any]:
    """
    获取 GPU 信息，支持 CUDA、MPS 和 CPU
    返回格式与 nvidia-smi 兼容
    """
    gpus = []

    # 优先检测 CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        try:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device = torch.cuda.device(i)
                props = torch.cuda.get_device_properties(i)

                # 获取显存信息
                memory_total = props.total_memory / (1024 ** 3)  # GB
                memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
                # 使用 reserved 作为已使用显存（更准确）
                memory_used = memory_reserved

                gpus.append({
                    "index": i,
                    "name": props.name,
                    "memoryTotal": round(memory_total, 1),
                    "memoryUsed": round(memory_used, 1),
                    "memoryPercent": round((memory_used / memory_total) * 100) if memory_total > 0 else 0,
                    "utilization": 0,  # PyTorch 无法直接获取利用率，需要 nvidia-smi
                    "temperature": 0  # PyTorch 无法直接获取温度，需要 nvidia-smi
                })
        except Exception as e:
            print(f"[GPU] Failed to get CUDA info: {e}")

    # 检测 MPS (Apple Silicon) - 仅在 CUDA 不可用时检测
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # MPS 使用共享系统内存
            memory = psutil.virtual_memory()
            memory_total = memory.total / (1024 ** 3)  # GB
            memory_available = memory.available / (1024 ** 3)  # GB
            memory_used = memory_total - memory_available

            # MPS 不支持利用率和温度监控
            gpus.append({
                "index": 0,
                "name": "Apple MPS",
                "memoryTotal": round(memory_total, 1),
                "memoryUsed": round(memory_used, 1),
                "memoryPercent": round((memory_used / memory_total) * 100) if memory_total > 0 else 0,
                "utilization": 0,  # MPS 不支持利用率监控
                "temperature": 0  # MPS 不支持温度监控
            })
        except Exception as e:
            print(f"[GPU] Failed to get MPS info: {e}")

    # 返回格式兼容旧版（单卡）和新版（多卡）
    if len(gpus) == 0:
        # 无法检测到 GPU
        return {
            "name": "CPU",
            "memoryTotal": 0,
            "memoryUsed": 0,
            "memoryPercent": 0,
            "utilization": 0,
            "temperature": 0,
            "gpus": [],
            "count": 0
        }
    elif len(gpus) == 1:
        # 单卡，保持向后兼容
        gpu = gpus[0]
        return {
            "name": gpu["name"],
            "memoryTotal": gpu["memoryTotal"],
            "memoryUsed": gpu["memoryUsed"],
            "memoryPercent": gpu["memoryPercent"],
            "utilization": gpu["utilization"],
            "temperature": gpu["temperature"],
            "gpus": gpus,
            "count": 1
        }
    else:
        # 多卡，汇总信息 + 详细列表
        total_memory = sum(g["memoryTotal"] for g in gpus)
        used_memory = sum(g["memoryUsed"] for g in gpus)
        avg_utilization = sum(g["utilization"] for g in gpus) // len(gpus)
        max_temp = max(g["temperature"] for g in gpus)

        return {
            "name": f"{len(gpus)}x {gpus[0]['name']}",
            "memoryTotal": round(total_memory, 1),
            "memoryUsed": round(used_memory, 1),
            "memoryPercent": round((used_memory / total_memory) * 100) if total_memory > 0 else 0,
            "utilization": avg_utilization,
            "temperature": max_temp,
            "gpus": gpus,
            "count": len(gpus)
        }


def get_gpu_info_simple() -> Dict[str, Any]:
    """
    获取简化的 GPU 信息（用于脚本）
    返回: {"name": str, "memory": str, "available": bool, "type": str}
    """
    # 优先检测 CUDA
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            memory_total = props.total_memory / (1024 ** 3)  # GB
            memory_str = f"{memory_total:.1f} GB"
            return {
                "name": props.name,
                "memory": memory_str,
                "available": True,
                "type": "cuda"
            }
        except Exception:
            pass

    # 检测 MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            memory = psutil.virtual_memory()
            memory_total = memory.total / (1024 ** 3)  # GB
            memory_str = f"{memory_total:.1f} GB (Shared)"
            return {
                "name": "Apple MPS",
                "memory": memory_str,
                "available": True,
                "type": "mps"
            }
        except Exception:
            pass

    return {
        "name": "CPU",
        "memory": "N/A",
        "available": False,
        "type": "cpu"
    }

