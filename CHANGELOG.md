# Changelog

## [mps-v1.0.0] - 2025-12-18

此版本為針對 Apple Silicon (MPS) 的首次重大重構，旨在提升在 M 系列晶片上的性能和穩定性。

### ✨ 新增

- **MPS 支援**:
  - 新增對 Apple MPS (Metal Performance Shaders) 的初步支援，允許在 Mac 上進行硬體加速訓練。
  - `hardware_detector.py` 和 `hardware.py` 現在可以檢測 MPS 設備，並自動選擇 `torch_sdpa` 作為推薦的注意力後端。
  - 為 MPS 設備提供了特定的優化配置，預設使用 `fp16` 混合精度以獲得最佳性能。
  - 在模型加載工具 (`zimage_utils.py`) 中增加了動態設備選擇，會自動選擇 `cuda`、`mps` 或 `cpu`。

### 🗑️ 移除

- **Flash Attention**:
  - 為了統一跨平台體驗並解決潛在的相容性問題，已完全移除 Flash Attention 相關的代碼。
  - PyTorch 2.0+ 的 `scaled_dot_product_attention` (SDPA) 現在是所有平台的預設和推薦選項。
  - 從 `setup.sh` 和 `setup.bat` 腳本中移除了 Flash Attention 的安裝檢查。
  - 從 `README.md` 和 `requirements.txt` 中刪除了所有關於 Flash Attention 的安裝說明和參考。

### 🐛 修復

- **MPS dataloader 穩定性**:
  - 在 `hardware_detector.py` 和 `hardware.py` 中，將 MPS 設備的 `dataloader_num_workers` 設置為 `0`，以解決在 Apple Silicon 上使用多進程數據加載時可能出現的穩定性問題。
- **NaN 問題**:
  - 在 `style_structure_loss.py` 中，對 `ssim` 計算和 `StyleStructureLoss` 的 `forward` 方法強制使用 `float32` 精度，以避免在 `fp16` 混合精度訓練中因數值不穩定而導致的 `NaN` (Not a Number) 錯誤。
  - `frequency_aware_loss.py` 已經具備 `float32` 轉換，無需修改，但確認了其穩健性。