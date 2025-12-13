# -*- coding: utf-8 -*-
"""
LongCat-Image Text Encoder Cache Script (Standalone)

将文本描述编码并缓存到磁盘。
带有 System Prompt 模板处理，以匹配 Pipeline 行为。

Usage:
    python -m longcat_image.cache_text_encoder \
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# LongCat-Image architecture identifier
ARCHITECTURE = "lc"

# Templates from Pipeline
PROMPT_TEMPLATE_ENCODE_PREFIX = '<|im_start|>system\nAs an image captioning expert, generate a descriptive text prompt based on an image content, suitable for input to a text-to-image model.<|im_end|>\n<|im_start|>user\n'
PROMPT_TEMPLATE_ENCODE_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n'

def load_text_encoder(model_path: str, device: torch.device, dtype: torch.dtype):
    """加载 Text Encoder"""
    from transformers import AutoModel, AutoTokenizer
    
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
        model = AutoModel.from_pretrained(
            str(model_dir),
            state_dict=state_dict,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    else:
        # 从目录加载
        path_obj = Path(model_path)
        if not (path_obj / "tokenizer.json").exists() and (path_obj.parent / "tokenizer").exists():
            tokenizer_path = str(path_obj.parent / "tokenizer")
        
        model = AutoModel.from_pretrained(
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


def encode_text_with_template(
    tokenizer,
    model,
    text: str,
    max_length: int = 512,
    device: torch.device = None,
) -> torch.Tensor:
    """
    编码文本为嵌入向量，包含模板处理和切片。
    模仿 LongCatImagePipeline.encode_prompt 逻辑。
    """
    # 1. Tokenize text (User Prompt)
    # 不加 special tokens, 手动拼接
    text_tokens = tokenizer(text, add_special_tokens=False)['input_ids']
    
    # 截断
    if len(text_tokens) > max_length:
        text_tokens = text_tokens[:max_length]
    
    # Padding to max_length (Pipeline uses pad to max_tokenizer_len)
    # 注意：Pipeline 先拼接再 padding 吗？不，Pipeline 是先 pad text, 再拼 prefix/suffix
    # "text_tokens_and_mask = self.tokenizer.pad(..."
    
    # 重新实现 Pipeline 逻辑：
    # 1. Tokenize Text -> Pad to max_length
    text_batch = tokenizer.pad(
        {'input_ids': [text_tokens]},
        max_length=max_length,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    text_input_ids = text_batch.input_ids[0]
    text_mask = text_batch.attention_mask[0]
    
    # 2. Tokenize Prefix & Suffix
    prefix_tokens = tokenizer(PROMPT_TEMPLATE_ENCODE_PREFIX, add_special_tokens=False)['input_ids']
    suffix_tokens = tokenizer(PROMPT_TEMPLATE_ENCODE_SUFFIX, add_special_tokens=False)['input_ids']
    
    prefix_ids = torch.tensor(prefix_tokens, dtype=text_input_ids.dtype)
    suffix_ids = torch.tensor(suffix_tokens, dtype=text_input_ids.dtype)
    
    prefix_mask = torch.ones(len(prefix_tokens), dtype=text_mask.dtype)
    suffix_mask = torch.ones(len(suffix_tokens), dtype=text_mask.dtype)
    
    # 3. Concat: Prefix + Text(Padded) + Suffix
    input_ids = torch.cat((prefix_ids, text_input_ids, suffix_ids), dim=-1)
    attention_mask = torch.cat((prefix_mask, text_mask, suffix_mask), dim=-1)
    
    # 4. Forward
    if device:
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1] # [1, seq_len, dim]
        
    # 5. Slice: Remove Prefix and Suffix
    # Slice range: [len(prefix) : -len(suffix)]
    # Pipeline 使用 hardcoded indices (36, 5)，这里我们动态计算更稳健
    start_idx = len(prefix_tokens)
    end_idx = len(suffix_tokens)
    
    # [1, total_len, dim] -> [1, max_length, dim] -> [max_length, dim]
    # Slice out the text part
    # Update: Pipeline logic:
    # prompt_embeds = prompt_embeds[:, self.prompt_template_encode_start_idx: -self.prompt_template_encode_end_idx ,:]
    
    embed = hidden_states[:, start_idx : -end_idx, :]
    embed = embed.squeeze(0) # [max_length, dim]
    
    return embed


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
    embed = encode_text_with_template(tokenizer, model, caption, max_length, device)
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
    
    # 使用 varlen 格式
    sd = {f"varlen_vl_embed_{dtype_str}": embed.cpu()}
    save_file(sd, str(output_file))
    
    return True


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
    parser = argparse.ArgumentParser(description="Cache text embeddings for LongCat-Image training")
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
    tokenizer, model = load_text_encoder(args.text_encoder, device, dtype)
    print("Text Encoder loaded successfully", flush=True)
    
    # 查找图片
    images = find_images(args.input_dir)
    total = len(images)
    print(f"Found {total} images", flush=True)
    print(f"Progress: 0/{total}", flush=True)
    
    processed = 0
    skipped = 0
    
    for i, image_path in enumerate(images, 1):
        # 检查是否已存在
        name = image_path.stem
        # 匹配 _lc_te.safetensors
        output_file = output_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
        
        # 如果是递归结构，需要重新计算路径 (fix bug: 上面的 output_file 计算是错的，如果是递归目录)
        # 正确逻辑在 loop 内的 try block 中
        # 为了高效 check，这里还是先用 input_dir 算一下
        if args.input_dir:
            try:
                rel_path = image_path.relative_to(args.input_dir)
                target_dir = output_dir / rel_path.parent
            except ValueError:
                target_dir = output_dir
            output_file = target_dir / f"{name}_{ARCHITECTURE}_te.safetensors"
            
        if args.skip_existing and output_file.exists():
            skipped += 1
            print(f"Progress: {i}/{total}", flush=True)
            continue
        
        try:
            if process_caption(image_path, tokenizer, model, output_dir, args.max_length, device, dtype, input_root=Path(args.input_dir)):
                processed += 1
            print(f"Progress: {i}/{total}", flush=True)
        except Exception as e:
            print(f"Error: {image_path}: {e}", flush=True)
            print(f"Progress: {i}/{total}", flush=True)
            continue
    
    print(f"Text encoding completed! Processed: {processed}, Skipped: {skipped}", flush=True)
    
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
