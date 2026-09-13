import types
import pprint
import torch

def print_config(target_config, title="CONFIGURATION SUMMARY"):
    """
    列印指定設定檔的內容。
    
    Args:
        target_config: 要列印的設定模組 (例如 import configs.config_ovo as cfg，傳入 cfg)
        title: 標題文字 (預設為 CONFIGURATION SUMMARY)
    """
    print("="*60)
    print(f"{title:^60}")
    print("="*60)

    # 取得屬性字典
    # 如果傳入的是模組或類別實例，用 vars()；如果已經是 dict，直接用
    if isinstance(target_config, dict):
        config_dict = target_config
    else:
        try:
            config_dict = vars(target_config)
        except TypeError:
            # 萬一 vars() 失敗 (雖然對 module 通常不會)，改用 dir()
            config_dict = {k: getattr(target_config, k) for k in dir(target_config)}

    for key, value in config_dict.items():
        
        # 1. 過濾掉 Python 內建屬性
        if key.startswith("__"):
            continue

        # 2. 過濾掉導入的模組 (如 import os)
        if isinstance(value, types.ModuleType):
            continue

        # 3. 過濾掉導入的類別定義 (只看變數值)
        if isinstance(value, type):
            continue

        # --- 開始列印 ---
        print(f"🔹 {key}:")
        
        # 如果是字典或列表，用 Pretty Print
        if isinstance(value, (dict, list)):
            pprint.pprint(value, indent=4, width=60)
        
        # 如果是 PyTorch Tensor 或 Device
        elif torch.is_tensor(value):
            print(f"    Tensor shape: {value.shape}")
        
        else:
            print(f"    {value}")
        
        print("-" * 60)