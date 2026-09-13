import torch
import torch.nn as nn
import timm
import os
import sys
import torch
import pandas as pd
import torch.nn as nn
from config  import NUM_CLASSES,DEVICE,MODEL_FILE_PATH,MODEL_NAME
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from model_manager import ModelManager
from evaluate import evaluate

from model import DualStreamModel

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.3):
        super().__init__()
        
        # 1. Backbones
        self.backbone_cnn = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0, global_pool='')
        self.backbone_swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, global_pool='')
        
        self.embed_dim = 1024
        self.num_tokens = 144
        
        # 位置編碼
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        # 2. 投影層 (加入 LayerNorm 和 GELU 增加非線性)
        self.q_proj = nn.Sequential(
            nn.Linear(2048, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )
        
        # Swin 已經是 1024 維，可以只用 LayerNorm，或保留 Linear 做特徵轉換
        self.kv_proj = nn.Sequential(
            nn.Linear(1024, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU()
        )
        
        # 3. Cross-Attention 層 (加入必備的 Transformer Block 結構)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        
        # FFN (Feed Forward Network) - 幫助 Attention 後的特徵重組
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        
        # 4. 分類頭 (加入 Dropout 防止過擬合)
        self.head_cnn = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(2048, num_classes))
        self.head_swin = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(1024, num_classes))
        self.head_attn = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(self.embed_dim, num_classes))
        
        # 5. 可學習的融合權重 (初始化為等權重，不要讓 attn 從 0 開始)
        self.weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, x):
        B = x.shape[0]
        
        # 提取特徵
        feat_cnn = self.backbone_cnn(x).flatten(2).transpose(1, 2)
        feat_swin = self.backbone_swin(x)
        
        if feat_swin.dim() == 4:
            if feat_swin.shape[-1] == 1024:
                feat_swin = feat_swin.reshape(B, -1, 1024)
            else:
                feat_swin = feat_swin.flatten(2).transpose(1, 2)
                
        # 基礎預測
        cnn_pooled = feat_cnn.mean(dim=1)
        swin_pooled = feat_swin.mean(dim=1)
        logits_cnn = self.head_cnn(cnn_pooled)
        logits_swin = self.head_swin(swin_pooled)
        
        # --- 改進的 Cross Attention Block ---
        q = self.q_proj(feat_cnn) 
        k = v = self.kv_proj(feat_swin) 
        
        # 加上位置編碼 (通常加在 Q 和 K)
        q = q + self.pos_embed
        k = k + self.pos_embed
        
        # 1. Attention + Residual + LayerNorm
        attn_out, _ = self.cross_attn(q, k, v)
        q = self.norm1(q + attn_out) # 殘差連接是關鍵！
        
        # 2. FFN + Residual + LayerNorm
        ffn_out = self.ffn(q)
        attn_features = self.norm2(q + ffn_out)
        
        # Attention 分類預測
        attn_pooled = attn_features.mean(dim=1) 
        logits_attn = self.head_attn(attn_pooled)
        
        # --- 最終融合 (使用 Softmax 確保權重總和為 1) ---
        w = torch.softmax(self.weights, dim=0)
        final_logits = logits_attn
        
        # 訓練時建議返回所有 logits 以計算輔助損失 (Auxiliary Loss)
        if self.training:
            return final_logits, logits_cnn, logits_swin, logits_attn
            
        return final_logits


def load_old_effb5_to_dual(model, checkpoint_path, num_classes=NUM_CLASSES):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        # 修正：加上 .1. 以對應 nn.Sequential 中的 Linear 層
        if v.shape == (num_classes, 2048):
            new_state_dict['head_cnn.1.weight'] = v
        elif v.shape == (num_classes,) and 'head_cnn.1.bias' not in new_state_dict:
            new_state_dict['head_cnn.1.bias'] = v
        else:
            new_state_dict[f"backbone_cnn.{k}"] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ EffB5 權重已載入至 backbone_cnn 與 head_cnn")

def load_pretrained_swin_to_dual(model, checkpoint_path, num_classes=NUM_CLASSES):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '') 
        # 修正：加上 .1. 以對應 nn.Sequential 中的 Linear 層
        if v.shape == (num_classes, 1024):
            new_state_dict['head_swin.1.weight'] = v
        elif v.shape == (num_classes,) and 'head_swin.1.bias' not in new_state_dict:
            new_state_dict['head_swin.1.bias'] = v
        else:
            new_state_dict[f"backbone_swin.{k}"] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    print("✅ Swin-B 權重已載入至 backbone_swin 與 head_swin")

if __name__ == '__main__':
    from dataloader import test_loader,train_loader,val_loader
    EFFB5_PATH = r'F:\CheSwinV3\ModelResult\0912_eff\0912_eff_epoch5_checkpoint.pth'
    SWINB_PATH = r'F:\CheSwinV3\ModelResult\0912_swin\0912_swin_epoch10_checkpoint.pth'
    torch.backends.cudnn.benchmark = True
    model = DualStreamModel()

    load_old_effb5_to_dual(model, EFFB5_PATH)
    load_pretrained_swin_to_dual(model, SWINB_PATH)

    model = model.to(DEVICE)

    for param in model.backbone_cnn.parameters(): param.requires_grad = False
    for param in model.backbone_swin.parameters(): param.requires_grad = False

    for param in model.head_cnn.parameters(): param.requires_grad = False
    for param in model.head_swin.parameters(): param.requires_grad = False
    model.weights.requires_grad = True

    new_params = [p for p in model.parameters() if p.requires_grad]
    print(f"待訓練參數組數量 (只剩融合權重): {len(new_params)}") 

    optimizer = torch.optim.AdamW(new_params, lr=1e-4, weight_decay=1e-3)     
    manager = ModelManager(model, optimizer)
    manager.train(train_loader, val_loader)
    manager.model.eval()
    evaluate(model, test_loader, save_file_path=os.path.join(os.path.join(MODEL_FILE_PATH, MODEL_NAME), "test_resultaaa.csv"), printr=True)
    #evaluate(model,test_loader,save_file_path=os.path.join(os.path.join(MODEL_FILE_PATH, MODEL_NAME),"Real_test_result5.csv"),printr=True)
    '''
    new_lrs = [1e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
    for i, param_group in enumerate(manager.optimizer.param_groups):
        param_group['lr'] = new_lrs[i]
        print(f"Group {i} updated to: {param_group['lr']:.2e}")

    #manager.load_statement(5, file_path=r"C:\CheSwinV3\ModelResult\0325_Effb5_swinb_384_original_attention_AllPretrained_con_posembed\0325_Effb5_swinb_384_original_attention_AllPretrained_con_posembed_epoch5_checkpoint.pth")
    torch.cuda.empty_cache()
    #manager.train(train_loader, val_loader)
    torch.cuda.empty_cache()
    #evaluate(model,test_loader,save_file_path=os.path.join(os.path.join(MODEL_FILE_PATH, MODEL_NAME),"Real_test_result4.csv"),printr=True)

    #manager.load_statement(epoch=10,file_path=r"C:\CheSwinV3\ModelResult\0513_Effb5_swinb_384_original_attention_AllPretrained_Gating_rgs\0513_Effb5_swinb_384_original_attention_AllPretrained_Gating_rgs_epoch10_checkpoint.pth") # 載入你剛訓練好的 Attention

    # 🌟 1. 解凍所有模型參數
    for param in model.parameters():
        param.requires_grad = True
    #manager.load_statement(epoch=10,file_path=r"C:\CheSwinV3\ModelResult\0513_Effb5_swinb_384_original_attention_AllPretrained_Gating_rgs\0513_Effb5_swinb_384_original_attention_AllPretrained_Gating_rgs_epoch10_checkpoint.pth") # 載入你剛訓練好的 Attention
    # 🌟 2. 設定「差分學習率 (Differential Learning Rate)」
    # 學霸 (Backbone) 用極小的學習率微調，新同學 (Attention) 用稍大的學習率
    backbone_params = list(model.backbone_cnn.parameters()) + list(model.backbone_swin.parameters())
    attn_params = [p for n, p in model.named_parameters() if 'backbone' not in n]

    # Backbone 用 1e-5 避免崩潰，Attention 維持 1e-4 或降為 5e-5
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': attn_params, 'lr': 5e-5}
    ], weight_decay=1e-3)

    # 重新綁定優化器並繼續訓練 Epoch 8 ~ 15
    manager.optimizer = optimizer
    manager.train(train_loader, val_loader)
    evaluate(model,test_loader,save_file_path=os.path.join(os.path.join(MODEL_FILE_PATH, MODEL_NAME),"Real_test_result15.csv"),printr=True)
    '''