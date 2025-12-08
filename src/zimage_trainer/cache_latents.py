# -*- coding: utf-8 -*-
"""
Z-Image Latent Cache Script (Standalone)

将图片编码为 latent 并缓存到磁盘。

Usage:
    python -m zimage_trainer.cache_latents \
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
from safetensors.torch import save_file
from tqdm import tqdm

from .utils.vae_utils import load_vae

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

    # 对齐到 8 的倍数（VAE 要求）
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

    # 应用 scaling 和 shift (Pipeline format)
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
    name = image_path.stem
    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"

    # 保存为 safetensors
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.squeeze(0)}
    save_file(sd, str(output_file))


def process_batch(
    image_paths: List[Path],
    vae,
    resolution: int,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    input_root: Path = None,
) -> Tuple[int, int]:
    """
    批量处理图片

    Returns:
        (processed_count, error_count)
    """
    import numpy as np

    # 加载所有图片并获取尺寸
    images_data = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            image = resize_image(image, resolution)
            w, h = image.size

            img_array = np.array(image).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)

            images_data.append((img_tensor, image_path, w, h))
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")
            continue

    if not images_data:
        return 0, len(image_paths)

    # 按尺寸分组，相同尺寸的可以批量处理
    size_groups = {}
    for img_tensor, image_path, w, h in images_data:
        size_key = (w, h)
        if size_key not in size_groups:
            size_groups[size_key] = []
        size_groups[size_key].append((img_tensor, image_path, w, h))

    processed = 0
    errors = 0

    # 处理每个尺寸组
    for (w, h), group in size_groups.items():
        # 将相同尺寸的图片堆叠成批次
        batch_tensors = [item[0] for item in group]
        batch_paths = [item[1] for item in group]

        # 堆叠为批次: (B, 3, H, W)
        batch_tensor = torch.stack(batch_tensors).to(device=device, dtype=dtype)

        # 归一化到 [-1, 1]
        batch_tensor = batch_tensor * 2.0 - 1.0

        try:
            # 批量编码
            with torch.no_grad():
                latents = vae.encode(batch_tensor).latent_dist.sample()

            # 应用 scaling 和 shift (Pipeline format)
            scaling_factor = getattr(vae.config, 'scaling_factor', 0.3611)
            shift_factor = getattr(vae.config, 'shift_factor', 0.1159)
            latents = (latents - shift_factor) * scaling_factor

            # 移动到 CPU
            latents = latents.cpu()

            # 保存每张图片的 latent
            F, H, W_latent = 1, latents.shape[2], latents.shape[3]
            dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"

            for i, image_path in enumerate(batch_paths):
                try:
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
                    name = image_path.stem
                    output_file = target_dir / f"{name}_{w}x{h}_{ARCHITECTURE}.safetensors"

                    # 保存为 safetensors
                    sd = {f"latents_{F}x{H}x{W_latent}_{dtype_str}": latents[i]}
                    save_file(sd, str(output_file))
                    processed += 1
                except Exception as e:
                    logger.error(f"Failed to save {image_path}: {e}")
                    errors += 1

        except Exception as e:
            logger.error(f"Failed to encode batch: {e}")
            # 如果批量处理失败，回退到单张处理
            for img_tensor, image_path, w, h in group:
                try:
                    process_image(image_path, vae, resolution, output_dir, device, dtype, input_root)
                    processed += 1
                except Exception as e2:
                    logger.error(f"Failed to process {image_path}: {e2}")
                    errors += 1

    return processed, errors


def main():
    parser = argparse.ArgumentParser(description="Cache latents for Z-Image training")
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

    # 过滤已存在的图片
    images_to_process = []
    skipped = 0
    input_root = Path(args.input_dir)

    for image_path in images:
        if args.skip_existing:
            # 计算实际的输出路径（与保存逻辑一致）
            if input_root:
                try:
                    rel_path = image_path.relative_to(input_root)
                    target_dir = output_dir / rel_path.parent
                except ValueError:
                    target_dir = output_dir
            else:
                target_dir = output_dir

            # 检查该目录下是否有匹配的缓存文件
            # 文件名格式: {name}_{WxH}_{arch}.safetensors
            name = image_path.stem
            pattern = f"{name}_*_{ARCHITECTURE}.safetensors"
            existing = list(target_dir.glob(pattern))
            if existing:
                skipped += 1
                continue
        images_to_process.append(image_path)

    # 批量处理图片
    processed = 0
    errors = 0
    batch_size = max(1, args.batch_size)  # 确保至少为 1

    # 创建进度条
    pbar = tqdm(total=total, desc="Caching latents", unit="image")

    # 将图片分批
    for batch_start in range(0, len(images_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(images_to_process))
        batch_images = images_to_process[batch_start:batch_end]

        # 批量处理
        batch_processed, batch_errors = process_batch(
            batch_images,
            vae,
            args.resolution,
            output_dir,
            device,
            dtype,
            input_root=input_root
        )

        processed += batch_processed
        errors += batch_errors

        # 更新进度条: 已处理的批次数量
        pbar.update(len(batch_images))

    pbar.close()

    print(f"Latent caching completed! Processed: {processed}, Skipped: {skipped}, Errors: {errors}", flush=True)

    # 清理显存
    del vae
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("VAE unloaded, GPU memory released", flush=True)


if __name__ == "__main__":
    main()
