#!/bin/bash
# ============================================================================
# None Trainer - Linux/Mac 一键部署脚本
# ============================================================================
# 
# 使用方法:
#   chmod +x setup.sh
#   ./setup.sh
#
# ============================================================================

set -e

echo "=============================================="
echo "   None Trainer - 一键部署脚本"
echo "=============================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 检查 Python 版本
echo -e "${BLUE}[1/8]${NC} 检查 Python 版本..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python3，请先安装 Python 3.10+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "  Python 版本: ${GREEN}$PYTHON_VERSION${NC}"

# 检查 CUDA
echo ""
echo -e "${BLUE}[2/8]${NC} 检查 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo -e "  GPU: ${GREEN}$GPU_NAME${NC}"
    echo -e "  显存: ${GREEN}$GPU_MEMORY${NC}"
else
    echo -e "${YELLOW}警告: 未检测到 NVIDIA GPU${NC}"
fi

# 创建虚拟环境
echo ""
echo -e "${BLUE}[3/8]${NC} 创建虚拟环境..."
VENV_DIR="$SCRIPT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "  ${YELLOW}虚拟环境已存在，跳过创建${NC}"
else
    # 使用 --system-site-packages 继承系统已安装的 PyTorch/Flash Attention
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo -e "  ${GREEN}虚拟环境已创建: $VENV_DIR${NC}"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
echo -e "  ${GREEN}虚拟环境已激活${NC}"

# 升级 pip
echo ""
echo -e "${BLUE}[4/8]${NC} 升级 pip..."
pip install --upgrade pip -q

# 检查 PyTorch
echo ""
echo -e "${BLUE}[5/8]${NC} 检查 PyTorch..."
if python3 -c "import torch; print(torch.__version__)" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'CPU')")
    echo -e "  ${GREEN}PyTorch 已安装: $TORCH_VERSION (CUDA: $CUDA_VERSION)${NC}"
else
    echo -e "  ${RED}PyTorch 未安装！${NC}"
    echo -e "  ${YELLOW}请先手动安装 PyTorch，参考 README.md${NC}"
    echo ""
    echo "  安装命令示例："
    echo "    # CUDA 12.8"
    echo "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
    echo ""
    echo "    # CUDA 12.1"
    echo "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    exit 1
fi

# 检查 Flash Attention
echo ""
if python3 -c "import flash_attn" 2>/dev/null; then
    FLASH_VERSION=$(python3 -c "import flash_attn; print(flash_attn.__version__)")
    echo -e "  ${GREEN}Flash Attention 已安装: $FLASH_VERSION${NC}"
else
    echo -e "  ${YELLOW}Flash Attention 未安装（可选，建议安装以提升性能）${NC}"
fi

# 安装依赖
echo ""
echo -e "${BLUE}[6/8]${NC} 安装 Python 依赖..."
pip install -r requirements.txt -q

# 安装 diffusers 最新版
echo -e "  安装 diffusers (git 最新版)..."
pip install git+https://github.com/huggingface/diffusers.git -q

# 安装本项目
pip install -e . -q

# 创建 .env 文件
if [ ! -f ".env" ]; then
    cp env.example .env
    echo -e "  ${GREEN}已创建 .env 配置文件${NC}"
fi

# 检查 Node.js
echo ""
echo -e "${BLUE}[7/8]${NC} 检查 Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "  ${GREEN}Node.js 已安装: $NODE_VERSION${NC}"
else
    echo -e "  ${RED}Node.js 未安装！${NC}"
    echo -e "  ${YELLOW}请先安装 Node.js 18+${NC}"
    echo "  下载地址: https://nodejs.org/"
    echo ""
    echo -e "  ${YELLOW}跳过前端构建，后端服务仍可运行${NC}"
    echo ""
    echo "=============================================="
    echo -e "${GREEN}✅ 后端部署完成！${NC}"
    echo "=============================================="
    echo ""
    echo "后续步骤:"
    echo "  1. 安装 Node.js 后运行: cd webui-vue && npm install && npm run build"
    echo "  2. 编辑 .env 配置模型路径"
    echo "  3. 运行 ./start.sh 启动服务"
    exit 0
fi

# 构建前端
echo ""
echo -e "${BLUE}[8/8]${NC} 构建前端..."
cd "$SCRIPT_DIR/webui-vue"

if [ ! -d "node_modules" ]; then
    echo -e "  安装前端依赖..."
    npm install --silent
fi

echo -e "  构建前端..."
npm run build --silent

cd "$SCRIPT_DIR"
echo -e "  ${GREEN}前端构建完成${NC}"

# 完成
echo ""
echo "=============================================="
echo -e "${GREEN}✅ 部署完成！${NC}"
echo "=============================================="
echo ""
echo "后续步骤:"
echo "  1. 编辑 .env 配置模型路径"
echo "  2. 运行 ./start.sh 启动服务"
echo ""
echo "常用命令:"
echo "  ./start.sh          # 启动 Web UI"
echo "  ./start.sh --help   # 查看帮助"
echo ""

