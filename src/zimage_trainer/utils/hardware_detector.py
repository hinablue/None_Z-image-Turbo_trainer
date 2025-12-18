"""
硬件检测和自动优化配置模块
自动检测 GPU 类型并优化训练参数

支持检测:
- GPU 类型和显存
- xformers 可用性
- SDPA 支持
"""
import torch
import psutil
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# xformers 检测
XFORMERS_AVAILABLE = False
XFORMERS_VERSION = None

try:
    import xformers
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    XFORMERS_VERSION = getattr(xformers, "__version__", "unknown")
except ImportError:
    pass
except Exception:
    pass

class HardwareDetector:
    """硬件检测和自动优化器"""
    
    def __init__(self):
        self.gpu_info = self.detect_gpu()
        self.cpu_info = self.detect_cpu()
        self.memory_info = self.detect_memory()
        self.xformers_info = self.detect_xformers()
        self.attention_info = self.detect_attention_backends()
        
    def detect_gpu(self) -> Dict[str, Any]:
        """检测 GPU 信息"""
        gpu_info = {
            "available": False,
            "device_count": 0,
            "device_name": "CPU",
            "compute_capability": None,
            "memory_total": 0,
            "memory_free": 0,
            "gpu_tier": "unknown",
            "is_mps": False
        }

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["device_name"] = torch.cuda.get_device_name(0)
            gpu_info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            gpu_info["compute_capability"] = torch.cuda.get_device_properties(0).major, torch.cuda.get_device_properties(0).minor
            gpu_info["memory_free"] = gpu_info["memory_total"] - torch.cuda.memory_allocated() / (1024**3)
            gpu_info["gpu_tier"] = self._classify_gpu_tier(gpu_info["device_name"], gpu_info["memory_total"])
            
            logger.info(f"🖥️ 检测到 CUDA GPU: {gpu_info['device_name']}")
            logger.info(f"[VRAM] GPU Memory: {gpu_info['memory_total']:.1f}GB (Free: {gpu_info['memory_free']:.1f}GB)")
            logger.info(f"[TIER] GPU Tier: {gpu_info['gpu_tier']}")

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info["available"] = True
            gpu_info["device_count"] = 1
            gpu_info["device_name"] = "Apple MPS"
            gpu_info["is_mps"] = True
            # MPS doesn't expose detailed memory/tier info, use system memory as a proxy
            mem_info = self.detect_memory()
            gpu_info["memory_total"] = mem_info.get("total", 0)
            gpu_info["memory_free"] = mem_info.get("available", 0)
            gpu_info["gpu_tier"] = self._classify_gpu_tier("Apple MPS", gpu_info["memory_total"])
            
            logger.info("🍏 检测到 Apple MPS")
            logger.info(f"[Shared Memory] System Memory: {gpu_info['memory_total']:.1f}GB")
            logger.info(f"[TIER] GPU Tier (based on shared memory): {gpu_info['gpu_tier']}")
        
        else:
            logger.info("[INFO] No GPU detected, will use CPU training")

        return gpu_info
    
    def detect_cpu(self) -> Dict[str, Any]:
        """检测 CPU 信息"""
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        return {
            "count": cpu_count,
            "frequency": cpu_freq.max if cpu_freq else None,
            "arch": "x86_64"
        }
    
    def detect_memory(self) -> Dict[str, Any]:
        """检测系统内存信息"""
        memory = psutil.virtual_memory()
        
        return {
            "total": memory.total / (1024**3),  # GB
            "available": memory.available / (1024**3),  # GB
            "percent": memory.percent
        }
    
    def detect_xformers(self) -> Dict[str, Any]:
        """检测 xformers 可用性和功能"""
        info = {
            "available": XFORMERS_AVAILABLE,
            "version": XFORMERS_VERSION,
            "memory_efficient_attention": False,
        }
        
        if not XFORMERS_AVAILABLE:
            logger.info("[WARN] xformers not installed")
            return info
        
        try:
            info["memory_efficient_attention"] = hasattr(xops, "memory_efficient_attention")
            
            logger.info(f"[OK] xformers {XFORMERS_VERSION} available")
                
        except Exception as e:
            logger.warning(f"xformers 功能检测失败: {e}")
        
        return info
    
    def detect_attention_backends(self) -> Dict[str, Any]:
        """检测所有可用的注意力后端"""
        backends = {
            "torch_sdpa": False,
            "xformers": XFORMERS_AVAILABLE,
            "recommended": "torch",
        }
        
        # 检查 PyTorch SDPA
        try:
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                backends["torch_sdpa"] = True
        except Exception:
            pass
        
        # 推荐后端
        if self.gpu_info.get("is_mps"):
            if backends["torch_sdpa"]:
                backends["recommended"] = "torch_sdpa"
        elif torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            
            # SM80+ (A100, H100, RTX 30xx/40xx)
            if cc[0] >= 8:
                if XFORMERS_AVAILABLE:
                    backends["recommended"] = "xformers"
                elif backends["torch_sdpa"]:
                    backends["recommended"] = "torch_sdpa"
            # SM70+ (V100, T4, RTX 20xx)
            elif cc[0] >= 7:
                if XFORMERS_AVAILABLE:
                    backends["recommended"] = "xformers"
                elif backends["torch_sdpa"]:
                    backends["recommended"] = "torch_sdpa"
            else:
                if backends["torch_sdpa"]:
                    backends["recommended"] = "torch_sdpa"
        
        logger.info(f"[RECOMMEND] Attention backend: {backends['recommended']}")
        return backends
    
    def _classify_gpu_tier(self, device_name: str, memory_total: float) -> str:
        """
        基于显存的严格分级
        """
        # For MPS, device_name is "Apple MPS", memory is shared system memory
        if "Apple" in device_name:
            if memory_total >= 64:
                return "tier_s" # M2/M3 Ultra
            elif memory_total >= 32:
                return "tier_a" # M2/M3 Max
            elif memory_total >= 16:
                return "tier_b" # M2/M3 Pro
            else:
                return "unsupported"

        # 门槛下调 1G，因为系统报告的显存略小于标称值 (如 24G 报告为 23.5G)
        if memory_total >= 31:
            return "tier_s"        # 32GB级 (A100/H100/Pro 6000/A6000/5090) - 全开
        elif memory_total >= 23:
            return "tier_a"        # 24GB级 (3090/4090)
        elif memory_total >= 15:
            return "tier_b"        # 16GB级 (4080/4070TiS/A4000)
        else:
            return "unsupported"   # <15GB - 不支持
    
    def get_optimized_config(self, config=None):
        """
        基于硬件生成动态优化配置
        """
        if config is None:
            config = {}
            
        memory_gb = self.gpu_info['memory_total']
        gpu_tier = self.gpu_info['gpu_tier']

        if self.gpu_info.get("is_mps"):
            logger.info(f"[CONFIG] Detected Apple MPS. Applying MPS-specific optimizations.")
            return {
                'mixed_precision': 'fp16',  # bfloat16 not well supported on MPS
                'gradient_checkpointing': True,
                'memory_efficient_preprocessing': True,
                'sdpa_enabled': True, # Use PyTorch's SDPA
                'attention_backend': 'torch_sdpa',
                'dataloader_num_workers': 4,
                'xformers_enabled': False,
            }
        
        if gpu_tier == 'unsupported':
            logger.warning(f"[WARN] Detected VRAM ({memory_gb:.1f}GB) below minimum (16GB). Extreme mode enabled.")
            # 极限压榨模式：尝试使用更多显存，虽然风险高
            return {
                'mixed_precision': 'fp16',
                'gradient_checkpointing': True,
                'memory_efficient_preprocessing': True,
                'max_memory_mb': int(memory_gb * 1024 * 1.0), # 100% 显存
                'spda_enabled': False,
                'block_swap_enabled': True,
                'block_swap_block_size': 1024, # 增大块大小
                'block_swap_max_cache_blocks': int(memory_gb * 80), # 增加缓存块
                'block_swap_swap_threshold': 0.99, # 99% 阈值 (极限)
                'dataloader_num_workers': 4, 
            }

        # 1. 确定混合精度类型
        cc_major, cc_minor = self.gpu_info.get("compute_capability", (0, 0))
        cc = float(f"{cc_major}.{cc_minor}")
        use_bf16 = cc >= 8.0
        
        # 基础配置
        optimized = {
            'mixed_precision': 'bf16' if use_bf16 else 'fp16',
            'gradient_checkpointing': True,
            'memory_efficient_preprocessing': True,
            'memory_monitoring_enabled': True,
            'comp_cache_compress': True,
        }
        
        # 动态计算 max_memory_mb (极限压榨: 100%)
        safe_memory_ratio = 1.0
        optimized['max_memory_mb'] = int(memory_gb * 1024 * safe_memory_ratio)
        
        # 基于 Tier 的配置
        if gpu_tier == 'tier_s':
            # Tier S (32GB+: A100/H100/5090): 全性能模式
            optimized.update({
                'blocks_to_swap': 0,
                'sdpa_enabled': True,
                'attention_backend': self.attention_info.get("recommended", "torch_sdpa"),
                'dataloader_num_workers': 16,
                'xformers_enabled': self.xformers_info.get('available', False),
            })
            
        elif gpu_tier == 'tier_a':
            # Tier A (24GB级: 3090/4090): 高性能模式
            optimized.update({
                'blocks_to_swap': 0,
                'sdpa_enabled': True,
                'attention_backend': self.attention_info.get("recommended", "torch_sdpa"),
                'dataloader_num_workers': 8,
                'xformers_enabled': self.xformers_info.get('available', False),
            })
            
        elif gpu_tier == 'tier_b':
            # Tier B (16GB级: 4080/4070TiS/P100): 内存优化模式
            optimized.update({
                'blocks_to_swap': 4,
                'sdpa_enabled': True,
                'attention_backend': 'torch_sdpa', # Force SDPA for memory saving
                'gradient_checkpointing': True,
                'dataloader_num_workers': 2,
                'gradient_accumulation_steps': 4,
                'xformers_enabled': self.xformers_info.get('available', False),
            })
            
            # 16G 使用更保守的显存比例 (95%)
            optimized['max_memory_mb'] = int(memory_gb * 1024 * 0.95)
            
        mode_desc = {
            'tier_s': 'Full Performance (no compression)',
            'tier_a': 'High Performance (LoRA optimized)',
            'tier_b': 'Balanced (light block swap)',
            'unsupported': 'Extreme (risky)'
        }.get(gpu_tier, 'Unknown')
        logger.info(f"[CONFIG] Hardware tier: {gpu_tier.upper()} (VRAM: {memory_gb:.1f}GB) - {mode_desc}")
        for key, value in optimized.items():
            logger.info(f"   {key}: {value}")
        
        return optimized
    
    def print_detection_summary(self):
        """打印硬件检测摘要"""
        # Use logger instead of print to avoid Windows GBK encoding issues
        logger.info("")
        logger.info("=" * 60)
        logger.info("[Hardware Detection Report]")
        logger.info("=" * 60)
        logger.info(f"GPU: {self.gpu_info['device_name']}")
        if not self.gpu_info.get("is_mps"):
            logger.info(f"VRAM: {self.gpu_info['memory_total']:.1f}GB")
        logger.info(f"GPU Tier: {self.gpu_info['gpu_tier']}")
        logger.info(f"CPU: {self.cpu_info['count']} cores")
        logger.info(f"System Memory: {self.memory_info['total']:.1f}GB")
        logger.info(f"Available Memory: {self.memory_info['available']:.1f}GB")
        logger.info("-" * 60)
        logger.info("Attention Backend:")
        xf_status = f"[OK] {self.xformers_info.get('version', '')}" if self.xformers_info.get('available') else "[NO]"
        logger.info(f"  xformers: {xf_status}")
        logger.info(f"  PyTorch SDPA: {'[OK]' if self.attention_info.get('torch_sdpa') else '[NO]'}")
        logger.info(f"  Recommended: {self.attention_info.get('recommended', 'torch')}")
        logger.info("=" * 60)