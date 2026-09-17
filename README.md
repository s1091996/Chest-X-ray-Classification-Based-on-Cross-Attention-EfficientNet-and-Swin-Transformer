# 基於 EfficientNet 與 Swin Transformer 交叉注意力特徵融合之胸腔 X 光多標籤

分類

## 1. 專案名稱與簡介（Project Title & Overview）

本專案結合卷積神經網路（CNN）與視覺轉換器（Vision Transformer），建構出特徵融合深度學習架構，用於胸部 X 光（Chest X-ray）14 種肺部病徵與無異常（No Finding）的多標籤分類任務。

開發此專案是為了解決胸部 X 光片診斷中「多種病灶同時並存」以及「不同病徵在空間尺度差異極大」的挑戰。傳統單一架構難以同時兼顧細微邊界病灶與跨區域全域解剖關聯；本系統結合 EfficientNet 的局部紋理擷取能力與 Swin Transformer 的長距離特徵關聯能力，透過交叉注意力機制（Cross-Attention）進行特徵計算注意力與加權融合。

使用者可利用本專案在 NIH ChestX-ray14 資料集上進行模型訓練、測試集評估，並產生包含準確率、F1-Score、AUC、敏感度（Sensitivity）與特異度（Specificity）等完整模型評估。

## 2. 功能特色（Features）

- **雙流混合特徵融合架構（Dual-Stream Fusion）**：結合 EfficientNet（擷取局部邊緣與病灶紋理）與 Swin Transformer（`swin_base_patch4_window12_384`，建立全域上下文），發揮 CNN 與 Transformer 的互補優勢。
- **交叉注意力重組機制（Cross-Attention Block）**：將 CNN 卷積特徵對應至 Query，Swin Transformer 特徵對應至 Key 與 Value，搭配可學習的 2D 位置編碼、殘差連接、LayerNorm 與 FFN 前饋網路，完成深層特徵交互重組。
- **可學習動態融合權重（Learnable Fusion Weights）**：模型同時具備 CNN 分類頭、Swin 分類頭與 Attention 分類頭，透過 Softmax 可學習參數自動平衡各串流在最終預測中的貢獻比例。
- **單一骨幹基線訓練（Backbone Baseline Training）**：支援獨立訓練單一模型（如 EfficientNet-B5）作為基線對照，或作為後續雙流架構特徵提取器的預訓練權重來源。
- **各類別動態最佳門檻搜尋（Per-Class Best Threshold Search）**：內建門檻搜尋演算法（以 0.01 步長搜尋 0.01～0.99 區間），在驗證集上自動找出使各病徵 F1-score 最大化的最佳分類門檻，有效改善醫學影像類別極度不平衡導致固定 0.5 門檻失真的問題。
- **完整醫學臨床指標評估（Comprehensive Medical Metrics）**：除常見的 Accuracy、Precision、Recall、F1-score 與 ROC-AUC 外，亦針對臨床需求計算各病徵的敏感度（Sensitivity / 病灶抓取率）與特異度（Specificity），並計算 Macro-Average 巨觀平均。
- **訓練流程與模型狀態管理（Model Manager & Checkpoint）**：內建 `ModelManager` 工具，整合 PyTorch 自動混合精度（AMP GradScaler）、梯度累積（Gradient Accumulation）、設定檔（`config.py`）自動備份、驗證集最佳 F1 早停機制（Early Stopping）以及多週期 Checkpoint 儲存與載入。
- **視覺化分析成果輸出（Visualization & Misclassification Analysis）**：評估時自動產出 15 類病徵個別的二元混淆矩陣圖表，以及全域假陰性（FN）、假陽性（FP）誤判矩陣與特定疾病混淆比例分析圖。

## 3. 系統流程／架構（System Architecture）

```text
胸部 X 光影像（NIH ChestX-ray14）
        ↓
資料前處理與資料增強（縮放 384×384、灰階補全三通道、隨機旋轉與平移翻轉、正規化）
        ↓
雙串流特徵提取（EfficientNet-B5 局部特徵 + Swin-B 全域特徵）
        ↓
交叉注意力機制（Cross-Attention：Q/KV 投影、位置編碼、殘差連接與 FFN）
        ↓
動態加權多頭分類（CNN Head、Swin Head、Attention Head 預測融合）
        ↓
各類別最佳決策門檻判定（Per-Class Best Threshold Tuning）
        ↓
產出 15 類多標籤預測結果、指標報表（CSV）與混淆分析視覺化圖表
```

### 主要模組職責說明

- `config.py`：全域環境與訓練超參數設定，包含影像尺寸、批次大小、學習率、類別名稱列表、標註 CSV 路徑與模型儲存路徑。
- `dataset.py`：實作 `NIHDataset` 資料集讀取邏輯，掃描影像目錄並將單通道灰階影像轉為三通道輸入，提供包含旋轉、平移與翻轉的資料增強管線。
- `dataloader.py`：載入患者切分的訓練、驗證與測試集 CSV，建立 PyTorch `DataLoader` 物件並配置資料載入工作執行緒（num_workers）。
- `train_backbone.py`：單一骨幹模型訓練主程式，支援透過 `timm` 建立模型並進行基準測試。
- `train_fusion.py`：雙流特徵融合模型（`DualStreamModel`）主程式，支援載入已預訓練好的骨幹權重，進行凍結微調或全模型微調訓練。
- `model_manager.py`：封裝訓練循環、自動混合精度加速、梯度累積、早停機制與模型權重備份/復原邏輯。
- `evaluate.py`：模型評估工具，負責進行前向推論、動態尋找最佳 F1 門檻、計算臨床評估指標並儲存為 CSV 報表。
- `print_config.py`：排版印出當前實驗各項超參數與路徑配置，方便紀錄實驗條件。

## 4. 專案結構（Project Structure）

```text
exp_original__/
├── results/
│   ├── cm/
│   │   ├── CM_Atelectasis.png
│   │   ├── CM_Cardiomegaly.png
│   │   ├── CM_Consolidation.png
│   │   ├── CM_Edema.png
│   │   ├── CM_Effusion.png
│   │   ├── CM_Emphysema.png
│   │   ├── CM_Fibrosis.png
│   │   ├── CM_Hernia.png
│   │   ├── CM_Infiltration.png
│   │   ├── CM_Mass.png
│   │   ├── CM_No_Finding.png
│   │   ├── CM_Nodule.png
│   │   ├── CM_Pleural_Thickening.png
│   │   ├── CM_Pneumonia.png
│   │   └── CM_Pneumothorax.png
│   ├── Confusion_Analysis_Pneumonia.png
│   ├── Confusion_Ratio_Analysis_Pneumonia.png
│   ├── Global_FN_Misclassification_Matrix.png
│   └── Global_FP_Misclassification_Matrix.png
├── 新增資料夾/
│   └── mainn.ipynb
├── config.py
├── dataloader.py
├── dataset.py
├── evaluate.py
├── model_manager.py
├── print_config.py
├── train_backbone.py
├── train_fusion.py
└── README.md
```

| 檔案／目錄 | 用途說明 |
| --- | --- |
| `config.py` | 專案組態設定檔，定義疾病類別、模型名稱、影像尺寸、批次大小、資料路徑與超參數。 |
| `dataset.py` | 定義 `NIHDataset` 資料集載入類別、灰階補全三通道邏輯（`RepeatIfGray`）與影像前處理管線。 |
| `dataloader.py` | 讀取訓練集、驗證集與測試集 CSV 標註檔，並建立對應的 PyTorch DataLoader。 |
| `model_manager.py` | 封裝訓練流程控制類別 `ModelManager`，管理 AMP 混合精度、梯度累積、早停與檢查點存取。 |
| `evaluate.py` | 評估指標運算模組，負責尋找各類別最佳決策門檻，計算 Acc、F1、AUC、敏感度與特異度並輸出 CSV。 |
| `print_config.py` | 格式化印出設定檔內容，方便終端機確認與檢查實驗配置。 |
| `train_backbone.py` | 單骨幹模型訓練與評估執行腳本（以 `timm` 的 EfficientNet-B5 為預設）。 |
| `train_fusion.py` | 雙流特徵融合模型訓練主程式，定義 `DualStreamModel` 並進行特徵重組與微調。 |
| `results/` | 儲存評估階段產生的圖表，包含各類別混淆矩陣（`cm/`）與全域誤判分析圖。 |
| `新增資料夾/mainn.ipynb` | 實驗開發與除錯紀錄筆記本，包含梯度監控、門檻分析與指標趨勢繪圖。 |

## 5. 安裝與快速開始（Installation & Quick Start）

### 環境需求

- 作業系統：Windows 10 / 11 或 Linux（Ubuntu 20.04+）
- Python 版本：Python 3.10 或 3.11
- 硬體建議：具備 CUDA 支援之 NVIDIA GPU（建議 VRAM 12GB 以上）
- 資料集需求：NIH ChestX-ray14 影像資料集與切分標註 CSV 檔

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

**Linux / macOS：**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安裝依賴套件

安裝 PyTorch（請根據本地 CUDA 版本選擇安裝指令，此處以 CUDA 12.1 為例）與專案必要套件：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm pandas numpy scikit-learn pillow tqdm matplotlib
```

### 4. 設定資料路徑與參數

開啟 `config.py`，將根路徑與資料夾路徑調整為本地實際存放位置：

```python
# config.py
FILE_PATH = "F:/CheSwinV3"                 # 專案根目錄
IMAGE_ROOT = "F:/NIH_ChestXray14"          # NIH ChestX-ray14 影像目錄（內含 images_001 ~ images_012）
DATASET_NAME = "0927_DataSet_pateint_split" # 資料集資料夾名稱（內含 train.csv, val.csv, test.csv）
IMAGE_SIZE = 384
BATCH_SIZE = 8
ACCUMULATION_STEPS = 2                     # 梯度累積步數（等效批次大小為 16）
NUM_EPOCHS = 10
OPTIMIZER_LR = 1e-4
```

### 5. 啟動訓練

**執行單一骨幹模型訓練：**
```bash
python train_backbone.py
```

**執行雙流注意力融合模型訓練：**
```bash
python train_fusion.py
```

## 6. 使用範例（Usage / Examples）

### 範例一：訓練單一 EfficientNet-B5 模型

執行 `train_backbone.py` 會自動讀取 `config.py` 設定，印出超參數並開始訓練：

```bash
python train_backbone.py
```

程式啟動後終端機會先印出設定清單，接著開始進行每個 Epoch 的訓練與驗證，並在早停觸發或訓練完成後自動於測試集執行評估：

```text
============================================================
                   Training Configuration                   
============================================================
🔹 MODEL_TIMM:
    efficientnet_b5
🔹 MODEL_NAME:
    0912_eff
🔹 IMAGE_SIZE:
    384
🔹 BATCH_SIZE:
    8
🔹 NUM_EPOCHS:
    10
------------------------------------------------------------
[ModelManager] 使用現有資料夾: F:/CheSwinV3/ModelResult/0912_eff
[ModelManager] Config 已備份至: F:/CheSwinV3/ModelResult/0912_eff/config.py

Epoch 1/10
Training: 100%|█████████████████████████| 19473/19473 [12:35<00:00, loss=0.1582]
[0912_eff epoch 1] Train Loss: 0.1610 | Val Acc: 0.9021 | Val F1: 0.2435 | Val AUC: 0.7612
Checkpoint saved to F:/CheSwinV3/ModelResult/0912_eff/0912_eff_epoch1_checkpoint.pth
```

### 範例二：執行雙流融合模型（Dual-Stream Model）

`train_fusion.py` 預設載入已預先訓練好的 EfficientNet-B5 與 Swin-B 骨幹權重，並可設定凍結骨幹僅微調注意力與融合權重，或進一步進行全模型微調：

```python
# train_fusion.py 關鍵微調設定
EFFB5_PATH = r'F:\CheSwinV3\ModelResult\0912_eff\0912_eff_epoch5_checkpoint.pth'
SWINB_PATH = r'F:\CheSwinV3\ModelResult\0912_swin\0912_swin_epoch10_checkpoint.pth'

model = DualStreamModel()
load_old_effb5_to_dual(model, EFFB5_PATH)
load_pretrained_swin_to_dual(model, SWINB_PATH)

# 凍結骨幹權重，僅優化融合參數
for param in model.backbone_cnn.parameters(): param.requires_grad = False
for param in model.backbone_swin.parameters(): param.requires_grad = False
model.weights.requires_grad = True
```

啟動雙流融合訓練：
```bash
python train_fusion.py
```

### 範例三：推論評估與輸出格式

評估階段會透過 `evaluate.py` 在終端機排版輸出各病徵的指標表格，並將結果存檔為 CSV 檔：

```text
Label                          |    Acc |     F1 |    AUC |   Sens |   Spec | Threshold
--------------------------------------------------------------------------------------------------------------
Atelectasis                    | 0.8857 | 0.3392 | 0.7963 | 0.3955 | 0.9250 |     0.100
Hernia                         | 0.9975 | 0.0000 | 0.8568 | 0.0000 | 0.9996 |     0.010
Cardiomegaly                   | 0.9557 | 0.3226 | 0.9284 | 0.5841 | 0.9626 |     0.070
Infiltration                   | 0.6828 | 0.3553 | 0.6969 | 0.5793 | 0.7012 |     0.210
Consolidation                  | 0.9365 | 0.1394 | 0.7409 | 0.2607 | 0.9501 |     0.070
Mass                           | 0.9675 | 0.3210 | 0.8123 | 0.2844 | 0.9865 |     0.460
Edema                          | 0.9873 | 0.2011 | 0.9136 | 0.1638 | 0.9954 |     0.240
Nodule                         | 0.9304 | 0.2323 | 0.7143 | 0.2461 | 0.9610 |     0.240
Effusion                       | 0.9208 | 0.4554 | 0.8802 | 0.5458 | 0.9450 |     0.080
Pleural_Thickening             | 0.8885 | 0.0832 | 0.7092 | 0.3015 | 0.8985 |     0.030
Emphysema                      | 0.9743 | 0.2432 | 0.8253 | 0.2526 | 0.9863 |     0.060
Pneumonia                      | 0.8859 | 0.0259 | 0.6607 | 0.2278 | 0.8903 |     0.010
Fibrosis                       | 0.9825 | 0.1405 | 0.8241 | 0.1069 | 0.9944 |     0.140
Pneumothorax                   | 0.9481 | 0.2753 | 0.8107 | 0.3726 | 0.9637 |     0.180
No Finding                     | 0.6208 | 0.7007 | 0.6936 | 0.8617 | 0.3649 |     0.340
--------------------------------------------------------------------------------------------------------------
Macro Avg                      | 0.9043 | 0.2557 | 0.7909 | 0.3455 | 0.9016
Results saved to F:/CheSwinV3/ModelResult/0912_swin/test_result5.csv
```

產生的 `test_result.csv` 格式範例：

```csv
,acc,f1,precision,recall,sensitivity,specificity,auc,threshold
Atelectasis,0.8857,0.3392,0.2969,0.3955,0.3955,0.9250,0.7963,0.1000
Cardiomegaly,0.9557,0.3226,0.2228,0.5841,0.5841,0.9626,0.9284,0.0700
Effusion,0.9208,0.4554,0.3907,0.5458,0.5458,0.9450,0.8802,0.0800
...
macro_avg,0.9043,0.2557,0.2280,0.3455,0.3455,0.9016,0.7909,
```

## 7. 效能指標／實驗結果（Results / Performance）

以下為專案實際測試集（Test Set）的真實實驗數據紀錄。

### 模型效能對比（Macro Average）

| 模型方法 | Accuracy | Macro F1 | ROC-AUC | Sensitivity | Specificity |
| --- | --- | --- | --- | --- | --- |
| 單一骨幹模型（Swin-B 基準） | 0.9043 | 0.2557 | 0.7909 | 0.3455 | 0.9016 |
| 單一骨幹模型（EfficientNet-B5 驗證基準） | 0.9096 | 0.2501 | 0.7734 | 0.3160 | 0.9014 |
| **雙流注意力融合模型（Dual-Stream Attention Fusion）** | **0.9045** | **0.3912** | **0.8438** | **0.4674** | **0.9130** |

> **說明**：雙流特徵融合模型（Dual-Stream Fusion）透過交叉注意力機制融合局部細節與全域特徵後，Macro F1-Score 自 0.2557 大幅提升至 **0.3912**，整體平均 ROC-AUC 由 0.7909 提升至 **0.8438**，病灶敏感度（Sensitivity）由 34.55% 提升至 **46.74%**。

### 雙流融合模型各類別測試表現（Dual-Stream Model 實測數據）

| 疾病類別（Disease Label） | Accuracy | F1-Score | ROC-AUC | Sensitivity | Specificity | 最佳決策門檻 |
| --- | --- | --- | --- | --- | --- | --- |
| **Atelectasis（肺不張）** | 0.8422 | 0.4220 | 0.8243 | 0.5599 | 0.8746 | 0.192 |
| **Hernia（疝氣）** | 0.9976 | 0.4954 | 0.9162 | 0.4737 | 0.9989 | 0.960 |
| **Cardiomegaly（心臟肥大）** | 0.9646 | 0.3767 | 0.9180 | 0.4199 | 0.9788 | 0.222 |
| **Infiltration（浸潤）** | 0.6989 | 0.4201 | 0.7244 | 0.6049 | 0.7196 | 0.151 |
| **Consolidation（肺實變）** | 0.9101 | 0.2495 | 0.8111 | 0.3490 | 0.9352 | 0.131 |
| **Mass（腫塊）** | 0.9302 | 0.4303 | 0.8651 | 0.4850 | 0.9558 | 0.222 |
| **Edema（肺水腫）** | 0.9588 | 0.3094 | 0.9029 | 0.3981 | 0.9721 | 0.182 |
| **Nodule（結節）** | 0.9157 | 0.3536 | 0.7982 | 0.4071 | 0.9462 | 0.172 |
| **Effusion（肋膜積水）** | 0.8696 | 0.5537 | 0.8829 | 0.6562 | 0.8996 | 0.263 |
| **Pleural Thickening（肋膜增厚）** | 0.9356 | 0.2221 | 0.8070 | 0.3365 | 0.9525 | 0.111 |
| **Emphysema（肺氣腫）** | 0.9751 | 0.5348 | 0.9427 | 0.5493 | 0.9865 | 0.354 |
| **Pneumonia（肺炎）** | 0.9418 | 0.1102 | 0.7744 | 0.2763 | 0.9506 | 0.050 |
| **Fibrosis（肺纖維化）** | 0.9824 | 0.2019 | 0.8143 | 0.1525 | 0.9947 | 0.303 |
| **Pneumothorax（氣胸）** | 0.9344 | 0.4353 | 0.8893 | 0.5113 | 0.9564 | 0.111 |
| **No Finding（無異常）** | 0.7101 | 0.7526 | 0.7866 | 0.8315 | 0.5731 | 0.394 |
| **Macro Average（巨觀平均）** | **0.9045** | **0.3912** | **0.8438** | **0.4674** | **0.9130** | — |

## 8. 限制與注意事項（Limitations / Notes）

- **硬體資源與顯存需求**：影像解析度設定為 384×384，且雙流架構同時運行 EfficientNet-B5 與 Swin-B 兩大骨幹，顯存消耗較大。建議在具備 12GB 以上顯存的 GPU 執行；若發生顯存不足（CUDA Out of Memory），可適度調小 `BATCH_SIZE` 並提高 `ACCUMULATION_STEPS` 保持等效批次大小。
- **路徑配置依賴**：`config.py` 中預設路徑包含硬編碼之磁碟機代號（如 `F:/CheSwinV3`、`F:/NIH_ChestXray14`），在其他機器或不同作業系統（如 Linux）執行前，必須先手動調整 `FILE_PATH`、`IMAGE_ROOT` 與 `DATASET_NAME`。
- **外部資料集結構要求**：`dataset.py` 預設 NIH 胸部 X 光影像存放在 `images_001/images/` 到 `images_012/images/` 等 12 個子目錄中，資料集目錄結構必須與此結構一致，否則會觸發找不到檔案的錯誤。
- **預訓練權重載入前提**：執行 `train_fusion.py` 前，需確認已具備單獨訓練完成的 EfficientNet 與 Swin-B 檢查點檔案（`.pth`），若路徑不存在將無法成功載入骨幹權重進行融合微調。

## 9. 授權（License）

本專案的授權方式尚未指定。
