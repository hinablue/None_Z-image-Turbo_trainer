# -*- coding: utf-8 -*-
"""
Z-Image Text Encoder Cache Script (Standalone)

将文本描述编码并缓存到磁盘。

Usage:
    python -m zimage_trainer.cache_text_encoder \
        --text_encoder /path/to/qwen_3_4b.safetensors \
        --input_dir /path/to/images \
        --output_dir /path/to/cache
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import save_file
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Z-Image architecture identifier
ARCHITECTURE = "zi"


def load_text_encoder(model_path: str, device: torch.device, dtype: torch.dtype):
    """加载 Qwen3 文本编码器"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading text encoder: {model_path}")

    tokenizer_path = model_path

    # 如果是 safetensors 文件，需要加载为 HuggingFace 格式
    if model_path.endswith('.safetensors'):
        # 尝试从同目录加载 tokenizer
        model_dir = Path(model_path).parent
        tokenizer_path = str(model_dir)

        # 加载模型权重
        from safetensors.torch import load_file
        state_dict = load_file(model_path)

        # 创建模型配置
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            state_dict=state_dict,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        # 从目录加载
        # 检查 tokenizer 是否在兄弟目录
        path_obj = Path(model_path)
        if not (path_obj / "tokenizer.json").exists() and (path_obj.parent / "tokenizer").exists():
            tokenizer_path = str(path_obj.parent / "tokenizer")
            logger.info(f"Tokenizer not found in {model_path}, using {tokenizer_path}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    model.to(device)
    model.eval()

    return tokenizer, model


def get_caption(image_path: Path) -> Optional[str]:
    """获取图片对应的文本描述"""
    # 尝试多种文件名格式
    txt_paths = [
        image_path.with_suffix('.txt'),
        image_path.with_suffix('.caption'),
        image_path.parent / f"{image_path.stem}.txt",
    ]

    for txt_path in txt_paths:
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()

    return None


def encode_text(
    tokenizer,
    model,
    text: str,
    max_length: int = 512,
    device: torch.device = None,
) -> torch.Tensor:
    """编码文本为嵌入向量"""
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",  # 使用固定长度padding，与训练时一致
    )

    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Encode
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 使用最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]
        # 去掉 batch 维度
        embed = hidden_states.squeeze(0)

    return embed


def encode_text_batch(
    tokenizer,
    model,
    texts: List[str],
    max_length: int = 512,
    device: torch.device = None,
) -> List[torch.Tensor]:
    """批量编码文本为嵌入向量"""
    if not texts:
        return []

    # Tokenize batch
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",  # 使用固定长度padding，与训练时一致
    )

    if device:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    # Encode batch
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # 使用最后一层隐藏状态
        hidden_states = outputs.hidden_states[-1]
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        # 返回列表，每个元素是一个样本的嵌入
        embeds = [hidden_states[i] for i in range(hidden_states.shape[0])]

    return embeds


def process_caption(
    image_path: Path,
    tokenizer,
    model,
    output_dir: Path,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
    input_root: Path = None,
) -> bool:
    """处理单个文本描述"""
    # 获取 caption
    caption = get_caption(image_path)
    if caption is None:
        logger.warning(f"No caption found for {image_path}")
        return False

    # 编码
    embed = encode_text(tokenizer, model, caption, max_length, device)
    embed = embed.to(dtype=dtype)

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

    # 保存
    name = image_path.stem
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"
    output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"

    # 使用 varlen 格式（与 musubi-tuner 兼容）
    sd = {f"varlen_vl_embed_{dtype_str}": embed.cpu()}
    save_file(sd, str(output_file))

    return True


def process_captions_batch(
    image_paths: List[Path],
    tokenizer,
    model,
    output_dir: Path,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
    input_root: Path = None,
) -> int:
    """批量处理文本描述，返回成功处理的数量"""
    # 收集所有有效的 caption
    valid_items = []
    for image_path in image_paths:
        caption = get_caption(image_path)
        if caption is not None:
            valid_items.append((image_path, caption))
        else:
            logger.warning(f"No caption found for {image_path}")

    if not valid_items:
        return 0

    # 批量编码
    texts = [caption for _, caption in valid_items]
    embeds = encode_text_batch(tokenizer, model, texts, max_length, device)

    # 保存每个结果
    success_count = 0
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"

    for (image_path, _), embed in zip(valid_items, embeds):
        try:
            embed = embed.to(dtype=dtype)

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

            # 保存
            name = image_path.stem
            output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"

            # 使用 varlen 格式（与 musubi-tuner 兼容）
            sd = {f"varlen_vl_embed_{dtype_str}": embed.cpu()}
            save_file(sd, str(output_file))

            success_count += 1
        except Exception as e:
            logger.error(f"Error saving {image_path}: {e}")

    return success_count


def find_images(input_dir: str) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def main():
    parser = argparse.ArgumentParser(description="Cache text embeddings for Z-Image training")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text encoder path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory (with .txt captions)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing (default: 1)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载文本编码器
    print(f"Loading Text Encoder: {args.text_encoder}", flush=True)
    tokenizer, model = load_text_encoder(args.text_encoder, device, dtype)
    print("Text Encoder loaded successfully", flush=True)

    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)

    # 处理
    success = 0
    skipped = 0
    input_root = Path(args.input_dir)

    # 创建进度条
    pbar = tqdm(total=total, desc="Caching text embeddings", unit="image")

    # 如果 batch_size 为 1，使用原来的逐个处理方式（保持兼容性）
    if args.batch_size == 1:
        for image_path in images:
            # 检查是否已存在
            name = image_path.stem

            # 计算预期输出路径
            try:
                rel_path = image_path.relative_to(input_root)
                target_dir = output_dir / rel_path.parent
            except ValueError:
                target_dir = output_dir

            output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"

            if args.skip_existing and output_file.exists():
                skipped += 1
                pbar.update(1)
                continue

            try:
                if process_caption(image_path, tokenizer, model, output_dir, args.max_length, device, dtype, input_root=input_root):
                    success += 1
                pbar.update(1)
            except Exception as e:
                print(f"Error: {image_path}: {e}", flush=True)
                pbar.update(1)
                continue
    else:
        # 批量处理模式
        # 先过滤掉已存在的文件
        to_process = []
        for image_path in images:
            name = image_path.stem
            try:
                rel_path = image_path.relative_to(input_root)
                target_dir = output_dir / rel_path.parent
            except ValueError:
                target_dir = output_dir
            output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"

            if args.skip_existing and output_file.exists():
                skipped += 1
                pbar.update(1)
            else:
                to_process.append(image_path)

        # 分批处理
        for batch_start in range(0, len(to_process), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(to_process))
            batch_images = to_process[batch_start:batch_end]

            try:
                batch_success = process_captions_batch(
                    batch_images, tokenizer, model, output_dir,
                    args.max_length, device, dtype, input_root=input_root
                )
                success += batch_success
                pbar.update(len(batch_images))
            except Exception as e:
                # 如果批量处理失败，回退到逐个处理
                logger.warning(f"Batch processing failed, falling back to individual processing: {e}")
                for image_path in batch_images:
                    try:
                        if process_caption(image_path, tokenizer, model, output_dir, args.max_length, device, dtype, input_root=input_root):
                            success += 1
                        pbar.update(1)
                    except Exception as e2:
                        print(f"Error: {image_path}: {e2}", flush=True)
                        pbar.update(1)

    pbar.close()

    print(f"Text encoding completed! Processed: {success}, Skipped: {skipped}", flush=True)

    # 清理显存
    del model
    del tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Text Encoder unloaded, GPU memory released", flush=True)


if __name__ == "__main__":
    main()
