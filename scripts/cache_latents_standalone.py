#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Image Latent Cache Script (Standalone - Single GPU)

简化版缓存脚本，仅支持单卡处理，确保稳定性。

Usage:
    python scripts/cache_latents_standalone.py \
        --vae /path/to/vae \
        --input_dir /path/to/images \
        --output_dir /path/to/cache \
        --resolution 1024
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
import numpy as np
from safetensors.torch import save_file
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Z-Image architecture identifier
ARCHITECTURE = "zi"


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
    
    aspect = w / h
    if aspect > 1:
        new_w = resolution
        new_h = int(resolution / aspect)
    else:
        new_h = resolution
        new_w = int(resolution * aspect)
    
    # 对齐到 8 的倍数
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    
    # 不放大
    if bucket_no_upscale:
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
    
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Cache latents for Z-Image training")
    parser.add_argument("--vae", type=str, required=True, help="VAE model path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--resolution", type=int, default=1024, help="Target resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (unused, for compatibility)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs (ignored, single GPU only)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)
    
    if total == 0:
        print("No images to process", flush=True)
        return
    
    # 单卡模式
    print(f"Using single GPU mode", flush=True)
    print(f"Progress: 0/{total}", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # 加载 VAE
    print(f"Loading VAE: {args.vae}", flush=True)
    if os.path.isdir(args.vae):
        vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
    else:
        vae = AutoencoderKL.from_single_file(args.vae, torch_dtype=dtype)
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    print("VAE loaded successfully", flush=True)
    
    input_root = Path(args.input_dir)
    processed = 0
    skipped = 0
    
    for i, image_path in enumerate(images, 1):
        name = image_path.stem
        existing = list(output_dir.glob(f"{name}_*_{ARCHITECTURE}.safetensors"))
        if args.skip_existing and existing:
            skipped += 1
            print(f"Progress: {i}/{total}", flush=True)
            continue
        
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            image = resize_image(image, args.resolution)
            w, h = image.size
            
            # 转换为 tensor
            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor * 2.0 - 1.0
            img_tensor = img_tensor.to(device=device, dtype=dtype)
            
            # 编码
            with torch.no_grad():
                latent = vae.encode(img_tensor).latent_dist.sample()
            
            scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
            shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
            latent = (latent - shift_factor) * scaling_factor
            
            # 保存
            latent = latent.cpu()
            F, H, W = 1, latent.shape[2], latent.shape[3]
            
            try:
                rel_path = image_path.relative_to(input_root)
                target_dir = output_dir / rel_path.parent
            except ValueError:
                target_dir = output_dir
            
            target_dir.mkdir(parents=True, exist_ok=True)
            output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"
            
            sd = {f"latents_{F}x{H}x{W}_bf16": latent.squeeze(0)}
            save_file(sd, str(output_file))
            
            processed += 1
            print(f"Progress: {i}/{total}", flush=True)
            
        except Exception as e:
            print(f"Error: {image_path}: {e}", flush=True)
            print(f"Progress: {i}/{total}", flush=True)
    
    print(f"Latent caching completed! Processed: {processed}, Skipped: {skipped}", flush=True)
    
    # 清理
    del vae
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VAE unloaded, GPU memory released", flush=True)


if __name__ == "__main__":
    main()
