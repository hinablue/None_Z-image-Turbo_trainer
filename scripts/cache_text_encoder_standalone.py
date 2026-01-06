#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Image Text Encoder Cache Script (Standalone - Single GPU)

简化版缓存脚本，仅支持单卡处理。

Usage:
    python scripts/cache_text_encoder_standalone.py \
        --text_encoder /path/to/qwen_3_4b \
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
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Z-Image architecture identifier
ARCHITECTURE = "zi"


def find_images(input_dir: str) -> List[Path]:
    """查找目录中的所有图片 (递归)"""
    input_path = Path(input_dir)
    extensions = ('.jpg', '.jpeg', '.png', '.webp')
    images = set()
    for ext in extensions:
        images.update(input_path.rglob(f'*{ext}'))
        images.update(input_path.rglob(f'*{ext.upper()}'))
    return sorted(list(images))


def get_caption(image_path: Path) -> Optional[str]:
    """获取图片对应的文本描述"""
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


def main():
    parser = argparse.ArgumentParser(description="Cache text embeddings for Z-Image training")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text encoder path")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output cache directory")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--skip_existing", action="store_true", help="Skip existing cache files")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载文本编码器
    print(f"Loading Text Encoder: {args.text_encoder}", flush=True)
    
    model_path = args.text_encoder
    tokenizer_path = model_path
    
    # 检查 tokenizer 路径
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
    print("Text Encoder loaded successfully", flush=True)
    
    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)
    print(f"Progress: 0/{total}", flush=True)
    
    input_root = Path(args.input_dir)
    success = 0
    skipped = 0
    
    for i, image_path in enumerate(images, 1):
        name = image_path.stem
        
        # 计算输出路径
        try:
            rel_path = image_path.relative_to(input_root)
            target_dir = output_dir / rel_path.parent
        except ValueError:
            target_dir = output_dir
        
        output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
        
        if args.skip_existing and output_file.exists():
            skipped += 1
            print(f"Progress: {i}/{total}", flush=True)
            continue
        
        # 获取 caption
        caption = get_caption(image_path)
        if caption is None:
            logger.warning(f"No caption found for {image_path}")
            print(f"Progress: {i}/{total}", flush=True)
            continue
        
        try:
            # Tokenize
            inputs = tokenizer(
                caption,
                return_tensors="pt",
                max_length=args.max_length,
                truncation=True,
                padding="max_length",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Encode
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                embed = hidden_states.squeeze(0).to(dtype=dtype)
            
            # 保存
            target_dir.mkdir(parents=True, exist_ok=True)
            sd = {"varlen_vl_embed_bf16": embed.cpu()}
            save_file(sd, str(output_file))
            
            success += 1
            print(f"Progress: {i}/{total}", flush=True)
            
        except Exception as e:
            print(f"Error: {image_path}: {e}", flush=True)
            print(f"Progress: {i}/{total}", flush=True)
    
    print(f"Text encoding completed! Processed: {success}, Skipped: {skipped}", flush=True)
    
    # 清理
    del model
    del tokenizer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Text Encoder unloaded, GPU memory released", flush=True)


if __name__ == "__main__":
    main()
