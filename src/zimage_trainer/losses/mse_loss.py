# -*- coding: utf-8 -*-
"""
📐 标准 MSE/L2 损失函数模块

提供标准的均方误差损失，适用于 Rectified Flow 训练。

损失类型：
- MSELoss: 标准 L2 损失 (Mean Squared Error)
- L2CosineLoss: L2 + Cosine 方向损失组合
- CharbonnierLoss: 平滑 L1 损失 (Robust L1)

数学公式：
- MSE: L = mean((v_pred - v_target)²)
- L2+Cosine: L = λ_l2 * MSE + λ_cos * (1 - cos_sim)
- Charbonnier: L = mean(sqrt((v_pred - v_target)² + ε²))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MSELoss(nn.Module):
    """
    标准 L2/MSE 损失
    
    L = mean((v_pred - v_target)²)
    
    适用于：
    - Rectified Flow v-prediction 训练
    - 需要严格像素级匹配的场景
    """
    
    def __init__(
        self,
        reduction: str = "mean",  # "mean", "sum", "none"
    ):
        super().__init__()
        self.reduction = reduction
        logger.info(f"[MSELoss] 初始化标准 L2 损失 (reduction={reduction})")
    
    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 MSE Loss
        
        Args:
            pred_v: 模型预测的速度 [B, C, H, W]
            target_v: 目标速度 [B, C, H, W]
            return_components: 是否返回损失分量
        
        Returns:
            loss: 总损失
            components: 损失分量字典
        """
        loss = F.mse_loss(pred_v, target_v, reduction=self.reduction)
        
        components = {"mse": loss}
        
        if return_components:
            return loss, components
        return loss, components


class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss (Robust L1 / 平滑 L1)
    
    L = mean(sqrt((v_pred - v_target)² + ε²))
    
    优势：
    - 在 0 点可微分，训练更稳定
    - 对离群值比 L2 更鲁棒
    - 保持边缘锐利度优于 L2
    """
    
    def __init__(
        self,
        epsilon: float = 1e-6,  # 平滑系数
        reduction: str = "mean",
    ):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        logger.info(f"[CharbonnierLoss] 初始化 Charbonnier 损失 (ε={epsilon})")
    
    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 Charbonnier Loss
        """
        diff = pred_v - target_v
        loss_per_element = torch.sqrt(diff ** 2 + self.epsilon ** 2)
        
        if self.reduction == "mean":
            loss = loss_per_element.mean()
        elif self.reduction == "sum":
            loss = loss_per_element.sum()
        else:  # none
            loss = loss_per_element
        
        components = {"charbonnier": loss}
        
        if return_components:
            return loss, components
        return loss, components


class L2CosineLoss(nn.Module):
    """
    L2 + Cosine 组合损失
    
    L = λ_l2 * MSE + λ_cos * (1 - cosine_similarity)
    
    设计理念：
    - L2 (MSE): 保证像素级精度
    - Cosine: 保证速度方向一致性
    
    适用于 Rectified Flow，其中速度方向比幅度更重要。
    """
    
    def __init__(
        self,
        lambda_l2: float = 1.0,      # L2 损失权重
        lambda_cosine: float = 0.1,  # Cosine 损失权重
    ):
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_cosine = lambda_cosine
        
        logger.info(f"[L2CosineLoss] 初始化组合损失")
        logger.info(f"  L2 权重: {lambda_l2}")
        logger.info(f"  Cosine 权重: {lambda_cosine}")
    
    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 L2 + Cosine Loss
        
        Args:
            pred_v: 模型预测的速度 [B, C, H, W]
            target_v: 目标速度 [B, C, H, W]
            return_components: 是否返回损失分量
        
        Returns:
            loss: 总损失
            components: 损失分量字典
        """
        # L2 损失
        loss_l2 = F.mse_loss(pred_v, target_v)
        
        # Cosine 方向损失
        pred_flat = pred_v.view(pred_v.shape[0], -1)
        target_flat = target_v.view(target_v.shape[0], -1)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        loss_cosine = 1.0 - cos_sim
        
        # 组合损失
        loss = self.lambda_l2 * loss_l2 + self.lambda_cosine * loss_cosine
        
        components = {
            "mse": loss_l2,
            "cosine": loss_cosine,
        }
        
        if return_components:
            return loss, components
        return loss, components


class L1CosineLoss(nn.Module):
    """
    Charbonnier (L1) + Cosine 组合损失
    
    L = λ_l1 * Charbonnier + λ_cos * (1 - cosine_similarity)
    
    这是 acrf_trainer 使用的标准损失函数。
    """
    
    def __init__(
        self,
        lambda_l1: float = 1.0,      # Charbonnier 损失权重
        lambda_cosine: float = 0.1,  # Cosine 损失权重
        epsilon: float = 1e-6,        # Charbonnier 平滑系数
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_cosine = lambda_cosine
        self.epsilon = epsilon
        
        logger.info(f"[L1CosineLoss] 初始化组合损失")
        logger.info(f"  Charbonnier(L1) 权重: {lambda_l1}")
        logger.info(f"  Cosine 权重: {lambda_cosine}")
    
    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        return_components: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算 Charbonnier + Cosine Loss
        """
        # Charbonnier 损失 (Robust L1)
        diff = pred_v - target_v
        loss_l1 = torch.sqrt(diff ** 2 + self.epsilon ** 2).mean()
        
        # Cosine 方向损失
        pred_flat = pred_v.view(pred_v.shape[0], -1)
        target_flat = target_v.view(target_v.shape[0], -1)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        loss_cosine = 1.0 - cos_sim
        
        # 组合损失
        loss = self.lambda_l1 * loss_l1 + self.lambda_cosine * loss_cosine
        
        components = {
            "charbonnier": loss_l1,
            "cosine": loss_cosine,
        }
        
        if return_components:
            return loss, components
        return loss, components


# === 便捷函数接口 ===

def compute_mse_loss(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
) -> torch.Tensor:
    """计算标准 MSE Loss"""
    return F.mse_loss(pred_v, target_v)


def compute_charbonnier_loss(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """计算 Charbonnier Loss"""
    diff = pred_v - target_v
    return torch.sqrt(diff ** 2 + epsilon ** 2).mean()


def compute_cosine_loss(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
) -> torch.Tensor:
    """计算 Cosine 方向损失"""
    pred_flat = pred_v.view(pred_v.shape[0], -1)
    target_flat = target_v.view(target_v.shape[0], -1)
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
    return 1.0 - cos_sim
