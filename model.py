import torch
import torch.nn as nn
import timm

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.3):
        super().__init__()
        
        # 1. Backbones
        self.backbone_cnn = timm.create_model('efficientnet_b5', pretrained=True, num_classes=0, global_pool='')
        self.backbone_swin = timm.create_model('swin_base_patch4_window12_384', pretrained=True, num_classes=0, global_pool='')
        
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