import sys
from pathlib import Path
import os
import traceback

# 设置路径
current_dir = Path(r"d:\AI\None_Z-image-Turbo_trainer\webui-vue\api")
project_root = Path(r"d:\AI\None_Z-image-Turbo_trainer")
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(project_root))

print(f"Python: {sys.version}")
print(f"Paths: {sys.path[:2]}")

try:
    print("Attempting to import routers.system...")
    from routers import system
    print("Success!")
except Exception:
    traceback.print_exc()
