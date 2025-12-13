# -*- coding: utf-8 -*-
"""
LongCat-Image Latent Cache Script (Standalone)

将图片编码为 latent 并缓存到磁盘。

Usage:
    python -m longcat_image.cache_latents \
        --vae /path/to/vae \
        --input_dir /path/to/images \
        --output_dir /path/to/cache \
        --resolution 1024
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple, Union

import torch
from PIL import Image
from safetensors.torch import save_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# LongCat-Image architecture identifier
ARCHITECTURE = "lc"


def load_vae(
    vae_path: str,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Load VAE.
    """
    from diffusers import AutoencoderKL
    
    logger.info(f"Loading VAE from {vae_path}")
    
    if os.path.isdir(vae_path):
        # Load from directory
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
    elif vae_path.endswith(".safetensors"):
        # Load from single file
        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
    else:
        raise ValueError(f"Unsupported VAE path format: {vae_path}")
    
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    logger.info(f"VAE loaded: scaling={getattr(vae.config, 'scaling_factor', 'N/A')}, "
                f"shift={getattr(vae.config, 'shift_factor', 'N/A')}")
    return vae


def find_images(input_dir: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def resize_image(image: Image.Image, resolution: int, bucket_no_upscale: bool = True) -> Image.Image:
    """调整图片大小，保持宽高比"""
    w, h = image.size
    
    # 计算目标尺寸
    aspect = w / h
    if aspect > 1:
        # 横向图片
        new_w = resolution
        new_h = int(resolution / aspect)
    else:
        # 纵向图片
        new_h = resolution
        new_w = int(resolution * aspect)
    
    # 对齐到 16 的倍数（VAE 8x + Patch 2x = 16x）
    # Pipeline 中要求 height % (vae_scale_factor * 2) == 0
    # 通常 vae_scale_factor=8, 所以是 16
    alignment = 16
    new_w = (new_w // alignment) * alignment
    new_h = (new_h // alignment) * alignment
    
    # 不放大
    if bucket_no_upscale:
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        new_w = (new_w // alignment) * alignment
        new_h = (new_h // alignment) * alignment
    
    # 确保至少有一个 patch
    new_w = max(new_w, alignment)
    new_h = max(new_h, alignment)
    
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


def process_image(
    image_path: Path,
    vae,
    resolution: int,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    input_root: Path = None,
) -> None:
    """处理单张图片"""
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    image = resize_image(image, resolution)
    w, h = image.size
    
    # 转换为 tensor
    import numpy as np
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # 归一化到 [-1, 1]
    img_tensor = img_tensor * 2.0 - 1.0
    img_tensor = img_tensor.to(device=device, dtype=dtype)
    
    # 编码
    with torch.no_grad():
        latent = vae.encode(img_tensor).latent_dist.sample()
    
    # 应用 scaling 和 shift (Flux/LongCat Pipeline format)
    # latent = (latent - shift) * scale
    scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
    shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
    latent = (latent - shift_factor) * scaling_factor
    
    # 保存
    latent = latent.cpu()
    F, H, W = 1, latent.shape[2], latent.shape[3]
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    
    # 计算输出路径 (保持目录结构)
    if input_root:
        try:
            rel_path = image_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
    else:
        target_dir = output_dir
        
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名格式: {name}_{WxH}_{arch}.safetensors
    # 注意：这里 W和H 是 resize 后的图片尺寸，不是 latent 尺寸
    name = image_path.stem
    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
    
    # 保存为 safetensors (Unpacked format)
    # Key name format: latents_{F}x{H_latent}x{W_latent}_{dtype}
    # H_latent = h // 8
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.squeeze(0)}
    save_file(sd, str(output_file))


def main():
    parser = argparse.ArgumentParser(description="Cache latents for LongCat-Image training")
    parser.add_argument("--vae", type=str, required=True, help="VAE model path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 VAE
    print(f"Loading VAE: {args.vae}", flush=True)
    vae = load_vae(args.vae, device=device, dtype=dtype)
    print("VAE loaded successfully", flush=True)
    
    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)
    print(f"Progress: 0/{total}", flush=True)  # 立即输出初始进度
    
    # 处理图片（使用简单进度输出，便于 WebUI 解析）
    processed = 0
    skipped = 0
    
    for i, image_path in enumerate(images, 1):
        # 检查是否已存在
        name = image_path.stem
        existing = list(output_dir.glob(f"{name}_*_{ARCHITECTURE}.safetensors"))
        if args.skip_existing and existing:
            skipped += 1
            print(f"Progress: {i}/{total}", flush=True)  # 跳过时也更新进度
            continue
        
        try:
            process_image(image_path, vae, args.resolution, output_dir, device, dtype, input_root=Path(args.input_dir))
            processed += 1
            print(f"Progress: {i}/{total}", flush=True)
        except Exception as e:
            print(f"Error: {image_path}: {e}", flush=True)
            print(f"Progress: {i}/{total}", flush=True)  # 出错时也更新进度
            continue
    
    print(f"Latent caching completed! Processed: {processed}, Skipped: {skipped}", flush=True)
    
    # 清理显存
    del vae
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VAE unloaded, GPU memory released", flush=True)


if __name__ == "__main__":
    main()
