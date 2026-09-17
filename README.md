# 基於 EfficientNet 與 Swin Transformer 交叉注意力特徵融合之胸腔 X 光多標籤分類系統

## 1. 專案名稱與簡介（Project Title & Overview）

本專案建構一套結合 EfficientNet 與 Swin Transformer 的深度融合系統，為分類胸部 X 光（Chest X-ray）14 種肺部病理特徵與無異常（No Finding）之多標籤臨床輔助診斷。

本專案旨在克服醫學影像診斷中「極端類別不平衡」與「多病灶疾病判定困難」等挑戰。傳統單一卷積神經網路受限於局部感受野，難以捕捉大範圍的病灶特徵；而視覺轉換器雖具備長距離建模優勢，卻缺乏局部歸納偏置（Local Inductive Bias）。

本專案提出雙流並行特徵融合機制：以 EfficientNet 擷取的局部病灶特徵為主動查詢向量（Query），和 Swin Transformer 提供的全域特徵（Key / Value）進行交叉注意力（Cross-Attention）結合，並加入殘差連接（Residual Connection）、可學習位置嵌入（Positional Embedding）與可學習動態權重融合（Learning Weight）。在僅配置單張消費級 NVIDIA RTX 4060（8GB VRAM）的有限硬體環境下，本模型於 NIH ChestX-ray14 資料集上達到 Macro F1-score 0.3930、14 種疾病平均 AUC 0.8492 的優異表現，兼顧診斷精確度與低硬體資源的可行性。

## 2. 功能特色（Features）

- **雙流混合特徵融合架構（Dual-Stream Parallel Architecture）**：以 EfficientNet-B5 負責捕捉微小病灶的邊界與局部特徵，並以 Swin Transformer（`swin_base_patch4_window12_384`）建立全域關係，以達到互補。
- **交叉注意力機制（Cross-Attention）**：將 CNN 卷積特徵設為查詢向量（Query），Swin Transformer 特徵投影設為鍵與值向量（Key & Value），透過 Multi-Head Cross-Attention（8 個注意力頭）自關聯局部異常與全域上下文資訊。
- **殘差連接與位置嵌入組件（Residual Connection & Positional Embedding）**：在注意力模組與前饋網路輸出端皆配置殘差連接，防止梯度消失；引入可學習位置編碼，避免特徵展平過程遺失重要的解位置資訊。
- **可學習動態加權決策（Learnable Softmax Weighting）**：同時引導 CNN 分類頭、Swin 分類頭與 Attention 分類頭進行預測，並透過一組反向傳播自動學習的 Softmax 權重參數（$\alpha, \beta, \gamma$）動態調節三者貢獻，避免人工設定權重的偏差。
- **各病徵獨立動態最佳門檻搜尋（Per-Class Best Threshold Tuning）**：針對 15 個標籤於驗證集上以 0.01 步長搜尋最大化 F1-score 的最佳決策門檻，解決醫學多標籤分類中固定 0.5 門檻在極度不平衡資料下的失真問題。
- **消費級硬體高效能訓練優化（Low-Resource Training Optimization）**：利用 PyTorch 自動混合精度（AMP GradScaler）搭配梯度累積步數（Accumulation Steps = 2，以批次大小 8 模擬等效批次 16），僅需 8GB 顯示記憶體即可完成大尺寸（384×384）雙流模型的完整微調。

## 3. 系統流程／架構（System Architecture）

### 系統整體作業流程

```text
輸入胸部 X 光影像（384×384，單通道轉三通道正規化）
        ↓
並行輸入雙流骨幹特徵提取網路
├── CNN 分支：EfficientNet-B5 ──→ 局部空間特徵圖 (B, 144, 2048)
└── Transformer 分支：Swin-B ───→ 全域階層特徵圖 (B, 144, 1024)
        ↓
特徵降維投影與空間位置引導（Linear + LayerNorm + GELU + Positional Embedding）
├── Query (Q) 來自 CNN 局部特徵（維度 D=1024）
└── Key / Value (K, V) 來自 Swin 全域特徵（維度 D=1024）
        ↓
多頭交叉注意力機制（Multi-Head Cross-Attention, Num Heads = 8）
        ↓
殘差連接與前饋網路重組（Residual Connection + FFN + LayerNorm）
        ↓
三路全局平均池化（GAP）與獨立分類頭預測
├── Logits_eff  (CNN Head)
├── Logits_swin (Swin Head)
└── Logits_attn (Cross-Attention Head)
        ↓
可學習權重軟最大化動態融合：Logits_final = α · Logits_eff + β · Logits_swin + γ · Logits_attn
        ↓
各疾病獨立最佳門檻判斷（Per-class F1 Thresholding）
        ↓
產出 15 類多標籤預測結果、指標報表與混淆分析圖表
```

## 4. 專案結構（Project Structure）

```text
exp_original__/
├── results/
├── config.py
├── dataloader.py
├── dataset.py
├── evaluate.py
├── model_manager.py
├── print_config.py
├── train_backbone.py
├── train_fusion.py
```

| 檔案／目錄 | 用途說明 |
| --- | --- |
| `config.py` | 專案全域設定檔，定義疾病標籤、模型名稱、影像尺寸、批次大小、資料路徑等等超參數。 |
| `dataset.py` | 定義 `NIHDataset` 資料集載入類別、與影像增強方法。 |
| `dataloader.py` | 讀取資料集 CSV 檔案，建立 DataLoader。 |
| `train_fusion.py` | 定義 `DualStreamModel` 雙流交叉注意力融合模型，與模型的訓練與評估。 |
| `train_backbone.py` | 單骨幹網路（如 EfficientNet-B5）的訓練與評估。 |
| `model_manager.py` | 管理 AMP 混合精度、梯度累積、早停機制與權重儲存。 |
| `evaluate.py` | 負責搜尋各標籤最佳 F1 門檻，計算各個評估指標分數並輸出 CSV。 |
| `print_config.py` | 印出實驗超參數與環境設定，供使用者確認。 |

## 5. 安裝與快速開始（Installation & Quick Start）

### 硬體與環境建議

- **作業系統**：Windows 10 / 11
- **中央處理器**：Intel Core i7-14700F 或同級以上處理器
- **圖形處理器**：NVIDIA GeForce RTX 4060（8GB VRAM）或以上支援 CUDA 之 GPU
- **系統記憶體**：16GB DDR5 / DDR4
- **Python 版本**：Python 3.10 或 3.11

### 1. 複製專案

```bash
git clone <repository-url>
cd exp_original__
```

### 2. 建立與啟動虛擬環境

**Windows (PowerShell / CMD)：**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. 安裝套件

根據本地環境安裝對應 CUDA 版本的 PyTorch 與專案套件：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm pandas numpy scikit-learn pillow tqdm matplotlib
```

### 4. 資料集設定與路徑確認

確保已自 NIH 下載 ChestX-ray14 影像資料集（包含 `images_001` 至 `images_012` 資料夾），並於 `config.py` 修改對應路徑：

```python
# config.py
FILE_PATH = "F:/CheSwinV3"                   # 專案主目錄
IMAGE_ROOT = "F:/NIH_ChestXray14"            # NIH 影像解壓縮目錄
DATASET_NAME = "0927_DataSet_pateint_split"   # 依病人分割之資料集目錄（內含 train.csv, val.csv, test.csv）
IMAGE_SIZE = 384
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2                       # 梯度累積步數（等效 Batch Size = 16）
NUM_EPOCHS = 10
OPTIMIZER_LR = 1e-4
```

### 5. 啟動模型訓練

**步驟 A：訓練單一 EfficientNet-B5 骨幹作為基準或權重來源**
```bash
python train_backbone.py
```

**步驟 B：啟動雙流交叉注意力融合模型訓練**
```bash
python train_fusion.py
```

## 6. 使用範例（Usage / Examples）

### 範例一：雙流特徵融合模型之微調執行

在 `train_fusion.py` 中設定預訓練骨幹權重路徑，並選擇僅訓練融合模組或進行整個模型的微調：

```python
# train_fusion.py
EFFB5_PATH = r'F:\CheSwinV3\ModelResult\0912_eff\0912_eff_epoch5_checkpoint.pth'
SWINB_PATH = r'F:\CheSwinV3\ModelResult\0912_swin\0912_swin_epoch10_checkpoint.pth'

model = DualStreamModel(num_classes=15)
load_old_effb5_to_dual(model, EFFB5_PATH)
load_pretrained_swin_to_dual(model, SWINB_PATH)

# 凍結骨幹，僅訓練交叉注意力與融合權重
for param in model.backbone_cnn.parameters(): param.requires_grad = False
for param in model.backbone_swin.parameters(): param.requires_grad = False
model.weights.requires_grad = True

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-3)
manager = ModelManager(model, optimizer)
manager.train(train_loader, val_loader)
```

執行命令：
```bash
python train_fusion.py
```

訓練結束時會呈現模型評估：
```text
[ModelManager] 使用現有資料夾: F:/CheSwinV3/ModelResult/0513_Effb5_swinb_384_original_attention
Epoch 1/10
Training: 100%|█████████████████████████| 19473/19473 [12:15<00:00, loss=0.1452]
[ModelManager] 最佳 F1-score: 0.3930 | 最佳 Epoch: 7
```

### 範例二：動態決策門檻搜尋與測試集推論評估

透過 `evaluate.py` 進行測試集評估，系統自動依據驗證集選出之最佳 F1 門檻進行分類判定並印出格式化報表：

```python
from evaluate import evaluate
results, y_true, y_pred, y_prob, chosen_thresholds = evaluate(
    model=model,
    loader=test_loader,
    threshold=None, # 自動搜尋最佳 F1 門檻
    save_file_path="test_result.csv",
    printr=True
)
```

## 7. 效能指標／實驗結果（Results / Performance）

以下實驗數據皆來自碩士論文實際評估與紀錄，在 NIH ChestX-ray14 獨立測試集（Test Set）進行評估。

### 1. 五種不同模型與架構之綜合效能比較（論文表 4.16）

| 模型名稱／分類架構 | Macro F1-score | Precision | Sensitivity | Specificity | 全 15 類 AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 單一 EfficientNet-B5 骨幹 | 0.3820 | 0.3290 | 0.4727 | 0.9124 | 0.8395 |
| 單一 Swin Transformer 骨幹 | 0.3627 | 0.3179 | 0.4402 | 0.9193 | 0.8307 |
| 二階段十四分類架構（串聯） | 0.3482 | 0.3210 | 0.4189 | **0.9242** | 0.7845 |
| 二階段長尾分群架構（串聯） | 0.3431 | 0.3074 | 0.4258 | 0.9159 | 0.7801 |
| **交叉注意力特徵融合架構（本專案）** | **0.3930** | **0.3478** | **0.4840** | 0.9113 | **0.8449** |

> **實驗洞察**：
> 1. 二階段串聯架構（先二分類再疾病分類）雖然在獨立任務表現佳，但整體串聯時遭遇「誤差向下傳遞」問題（第一階段誤判強制傳遞至第二階段），使 F1 與 AUC 顯著低於單一模型。
> 2. 本研究之交叉注意力融合架構打破串聯限制，透過雙流互補融合，取得所有模型中最佳的 **F1-score (0.3930)** 與 **AUC (0.8449)**。

### 2. 交叉注意力特徵融合模組消融實驗（論文表 4.14 與表 4.15）

| 模組配置狀況 | Macro F1 | Precision | Sensitivity | Specificity | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 基準交叉注意力（無殘差、無位置編碼） | 0.3781 | 0.3257 | 0.4778 | 0.9095 | 0.8267 |
| ＋ 導入殘差連接（Residual Connection） | 0.3871 | 0.3325 | **0.4886** | 0.9111 | 0.8402 |
| ＋ **導入可學習位置嵌入（Positional Embedding）** | **0.3930** | **0.3478** | 0.4840 | **0.9113** | **0.8449** |

> **消融分析**：殘差連接能有效防止深層反向傳播時特徵退化；而位置編碼則補足特徵展平（Flatten）時遺失之解剖空間幾何資訊，使模型綜合判別品質最佳。

## 8. 限制與注意事項（Limitations / Notes）

- **硬體資源與超參數連動**：本專案在 8GB VRAM 環境下訓練時將批次大小設為 8，並透過梯度累積（Accumulation Steps = 2）模擬批次大小 16。若移至更低顯存設備，需進一步降低批次大小並增加累積步數，以避免顯存不足（CUDA Out of Memory）。
- **資料路徑設定**：`config.py` 預設為本機路徑（如 `F:/CheSwinV3`、`F:/NIH_ChestXray14`），更換主機，需手動修正為相符之絕對路徑。
- **資料集目錄規範**：影像讀取邏輯預設 NIH ChestX-ray14 解壓縮為 `images_001/images/` 至 `images_012/images/`，需保持該目錄結構以利 `NIHDataset` 建立索引。
- **預訓練權重依賴性**：執行 `train_fusion.py` 前，需具備已在該資料集上訓練完之 EfficientNet-B5 與 Swin-B 的 `.pth` 檢查點；若直接隨機初始化權重訓練，最終效能將受到影響。

