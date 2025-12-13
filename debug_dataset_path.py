"""调试脚本：检查 DATASET_PATH 的解析情况"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 模拟 config.py 的加载过程
API_DIR = Path(__file__).parent / "webui-vue" / "api"
WEBUI_DIR = API_DIR.parent
PROJECT_ROOT = Path(__file__).parent

print(f"[DEBUG] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[DEBUG] .env file path = {PROJECT_ROOT / '.env'}")
print(f"[DEBUG] .env file exists = {(PROJECT_ROOT / '.env').exists()}")

# 加载 .env 前
print(f"\n--- 加载 .env 前 ---")
print(f"DATASET_PATH (os.environ) = {os.environ.get('DATASET_PATH', '<未设置>')}")
print(f"DATASET_PATH (os.getenv) = {os.getenv('DATASET_PATH', '<未设置>')}")

# 加载 .env
load_dotenv(PROJECT_ROOT / ".env")

# 加载 .env 后
print(f"\n--- 加载 .env 后 ---")
print(f"DATASET_PATH (os.environ) = {os.environ.get('DATASET_PATH', '<未设置>')}")
print(f"DATASET_PATH (os.getenv) = {os.getenv('DATASET_PATH', '<未设置>')}")

# 解析为路径
def _resolve_path(env_key: str, default: str) -> Path:
    value = os.getenv(env_key, "")
    if not value:
        value = default
    if not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p

DATASETS_DIR = _resolve_path("DATASET_PATH", "./datasets")

print(f"\n--- 最终结果 ---")
print(f"DATASETS_DIR = {DATASETS_DIR}")
print(f"DATASETS_DIR.exists() = {DATASETS_DIR.exists()}")
print(f"DATASETS_DIR.resolve() = {DATASETS_DIR.resolve()}")
