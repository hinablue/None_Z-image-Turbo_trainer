"""
L2 Ratio 调度器

支持训练过程中按 epoch 动态调整 L2 混合比例。

调度模式:
- constant: 固定值
- linear_increase: 线性增加 (适合加速蒸馏)
- linear_decrease: 线性减少 (适合Turbo微调)
- step: 阶梯式 (自定义每阶段的值)
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class L2RatioScheduler:
    """L2 Ratio 调度器"""
    
    MODES = ['constant', 'linear_increase', 'linear_decrease', 'step']
    
    def __init__(
        self,
        mode: str = 'constant',
        initial_ratio: float = 0.3,
        final_ratio: float = 0.3,
        num_epochs: int = 10,
        milestones: Optional[List[int]] = None,
        ratios: Optional[List[float]] = None,
    ):
        """
        Args:
            mode: 调度模式 ('constant', 'linear_increase', 'linear_decrease', 'step')
            initial_ratio: 起始比例
            final_ratio: 结束比例
            num_epochs: 总训练 epoch 数
            milestones: 阶梯模式的切换点 (epoch 索引, 从1开始)
            ratios: 阶梯模式每阶段的值 (长度应为 len(milestones) + 1)
        """
        if mode not in self.MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.MODES}")
        
        self.mode = mode
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.num_epochs = num_epochs
        self.milestones = milestones or []
        self.ratios = ratios or [initial_ratio]
        
        # 验证阶梯模式参数
        if mode == 'step':
            if len(self.ratios) != len(self.milestones) + 1:
                # 自动插值
                self._interpolate_step_ratios()
        
        logger.info(f"[L2 Scheduler] Mode: {mode}")
        logger.info(f"[L2 Scheduler] Initial: {initial_ratio}, Final: {final_ratio}")
        if mode == 'step':
            logger.info(f"[L2 Scheduler] Milestones: {self.milestones}, Ratios: {self.ratios}")
    
    def _interpolate_step_ratios(self):
        """自动插值阶梯比例"""
        n_stages = len(self.milestones) + 1
        if self.mode == 'linear_increase' or self.initial_ratio < self.final_ratio:
            # 递增
            step = (self.final_ratio - self.initial_ratio) / (n_stages - 1) if n_stages > 1 else 0
            self.ratios = [self.initial_ratio + i * step for i in range(n_stages)]
        else:
            # 递减
            step = (self.initial_ratio - self.final_ratio) / (n_stages - 1) if n_stages > 1 else 0
            self.ratios = [self.initial_ratio - i * step for i in range(n_stages)]
    
    def get_ratio(self, epoch: int) -> float:
        """
        获取指定 epoch 的 L2 ratio
        
        Args:
            epoch: 当前 epoch (从 1 开始)
        
        Returns:
            当前应使用的 L2 ratio
        """
        if self.mode == 'constant':
            return self.initial_ratio
        
        elif self.mode == 'linear_increase':
            # 线性增加
            if self.num_epochs <= 1:
                return self.final_ratio
            progress = (epoch - 1) / (self.num_epochs - 1)
            return self.initial_ratio + progress * (self.final_ratio - self.initial_ratio)
        
        elif self.mode == 'linear_decrease':
            # 线性减少
            if self.num_epochs <= 1:
                return self.final_ratio
            progress = (epoch - 1) / (self.num_epochs - 1)
            return self.initial_ratio - progress * (self.initial_ratio - self.final_ratio)
        
        elif self.mode == 'step':
            # 阶梯式
            stage = 0
            for i, milestone in enumerate(self.milestones):
                if epoch > milestone:
                    stage = i + 1
            return self.ratios[min(stage, len(self.ratios) - 1)]
        
        return self.initial_ratio
    
    def get_schedule_info(self) -> str:
        """获取调度信息字符串（用于日志和前端显示）"""
        if self.mode == 'constant':
            return f"固定值: {self.initial_ratio}"
        
        elif self.mode == 'linear_increase':
            return f"渐进增加: {self.initial_ratio} → {self.final_ratio}"
        
        elif self.mode == 'linear_decrease':
            return f"渐进减少: {self.initial_ratio} → {self.final_ratio}"
        
        elif self.mode == 'step':
            stages = []
            prev = 0
            for i, milestone in enumerate(self.milestones):
                stages.append(f"Epoch {prev+1}-{milestone}: {self.ratios[i]:.2f}")
                prev = milestone
            stages.append(f"Epoch {prev+1}-{self.num_epochs}: {self.ratios[-1]:.2f}")
            return "阶梯式: " + " | ".join(stages)
        
        return "未知模式"


def create_l2_scheduler_from_args(args) -> Optional[L2RatioScheduler]:
    """从训练参数创建 L2 调度器"""
    
    # 检查是否启用 RAFT 模式
    raft_mode = getattr(args, 'raft_mode', False)
    if not raft_mode:
        return None
    
    # 获取调度参数
    # 获取调度参数
    l2_schedule_mode = getattr(args, 'l2_schedule_mode', 'constant')
    
    # initial_ratio: 如果参数为 None，回退到 free_stream_ratio，再没有则 0.3
    l2_initial_ratio = getattr(args, 'l2_initial_ratio', None)
    if l2_initial_ratio is None:
        l2_initial_ratio = getattr(args, 'free_stream_ratio', 0.3)
        
    l2_final_ratio = getattr(args, 'l2_final_ratio', None)
    if l2_final_ratio is None:
        l2_final_ratio = l2_initial_ratio
    num_epochs = getattr(args, 'num_train_epochs', 10)
    
    # 阶梯模式参数
    l2_milestones = getattr(args, 'l2_milestones', [])
    l2_ratios = getattr(args, 'l2_ratios', [])
    
    # 处理字符串格式的 milestones
    if isinstance(l2_milestones, str):
        l2_milestones = [int(x.strip()) for x in l2_milestones.split(',') if x.strip()]
    
    return L2RatioScheduler(
        mode=l2_schedule_mode,
        initial_ratio=l2_initial_ratio,
        final_ratio=l2_final_ratio,
        num_epochs=num_epochs,
        milestones=l2_milestones,
        ratios=l2_ratios if l2_ratios else None,
    )
