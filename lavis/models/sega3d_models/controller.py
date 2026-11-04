import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 🚀 [新增] NeRF 同款高频位置编码器
class FourierPositionalEncoding(nn.Module):
    def __init__(self, in_channels=3, num_freqs=8, d_model=256):
        super().__init__()
        # 生成 2^0, 2^1, ..., 2^7 的频率
        freqs = 2 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs * math.pi) 
        
        # 输入维度: 原坐标(3) + sin(24) + cos(24) = 51
        out_dim = in_channels * (1 + 2 * num_freqs)
        self.proj = nn.Sequential(
            nn.Linear(out_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x):
        # x shape: [..., 3]
        freqs = self.freqs.to(x.device)
        scaled = x.unsqueeze(-1) * freqs # [..., 3, 8]
        sin_feats = torch.sin(scaled)
        cos_feats = torch.cos(scaled)
        # 拼接在一起
        feats = torch.cat([x.unsqueeze(-1), sin_feats, cos_feats], dim=-1)
        feats = feats.view(*x.shape[:-1], -1) # [..., 51]
        return self.proj(feats)
    

class Iterative3DMaskDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=3, dropout=0.0, p_in_dim=32):
        super().__init__()
        self.num_layers = num_layers
        self.nhead = nhead  # 🚀 [新增] 记住头数，给 mask 复制用

        # 特征升维器：把 SparseUNet 的 32 维升到 256 维
        self.point_proj_in = nn.Sequential(
            nn.Linear(p_in_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 3D 相对坐标的高频位置编码
        self.pos_enc = FourierPositionalEncoding(in_channels=3, num_freqs=8, d_model=d_model)
        
        # 核心注意力层
        self.sa_layers = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.sa_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        self.ca_layers = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) for _ in range(num_layers)])
        self.ca_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

        self.mask_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))

    def forward(self, query, p_feats, rel_xyz):
        """
        query: [K_sel, 256] 
        p_feats: [N, 32] 
        rel_xyz: [K_sel, N, 3] 
        """
        K_sel, D = query.shape
        N = p_feats.shape[0]
        
        # 强制把点云特征升到 256 维！
        p_feats = self.point_proj_in(p_feats) # [N, 32] -> [N, 256]
        
        # 1. 准备 Query: [1, K_sel, 256]
        q = query.unsqueeze(0) 
        
        # 2. 准备 Key, Value 和 掩码特征
        base_k = p_feats.unsqueeze(0).expand(K_sel, -1, -1) 
        v = base_k
        mask_feats = self.mask_proj(p_feats) # [N, 256]
        
        # 加入 3D 位置编码
        pe = self.pos_enc(rel_xyz) # [K_sel, N, 256]
        k = base_k + pe 
        
        attn_mask = None
        final_mask_logits = None
        
        for i in range(self.num_layers):
            # A. Self-Attention 
            q_sa, _ = self.sa_layers[i](q, q, q)
            q = self.sa_norms[i](q + q_sa)
            
            q_ca = q.transpose(0, 1) # [K_sel, 1, 256]
            
            # B. Cross-Attention 
            if attn_mask is not None and i > 0:
                ca_out, _ = self.ca_layers[i](q_ca, k, v, attn_mask=attn_mask)
            else:
                ca_out, _ = self.ca_layers[i](q_ca, k, v)
                
            q_ca = self.ca_norms[i](q_ca + ca_out)
            
            # C. FFN
            q_ca = self.ffn_norms[i](q_ca + self.ffn_layers[i](q_ca))
            q = q_ca.transpose(0, 1) # 恢复为 [1, K_sel, 256] 给下一轮的 SA
            
            # D. 点乘预测 Mask
            pred_masks = torch.einsum('k q d, n d -> k n', q_ca, mask_feats) 
            final_mask_logits = pred_masks 
            
            # E. 🚀 [终极修复]: 帮 PyTorch 将 Mask 复制给每一个 Attention Head
            # pred_masks 的形状是 [K_sel, N]
            # unsqueeze(1) 变成 [K_sel, 1, N]
            # E. 🚀 [终极修复]: 帮 PyTorch 将 Mask 复制给每一个 Attention Head
            attn_mask_bool = (pred_masks.unsqueeze(1).sigmoid() < 0.5).bool() 
            
            # 🛡️ 【防 NaN 核心护盾】：如果随机乱猜导致所有点都被屏蔽，强行放开视野，防止 Softmax(0/0)
            all_masked = attn_mask_bool.all(dim=-1, keepdim=True)
            attn_mask_bool = attn_mask_bool.masked_fill(all_masked, False)
            
            # repeat_interleave 会把维度扩充以匹配 nhead
            attn_mask = attn_mask_bool.repeat_interleave(self.nhead, dim=0)
            
        return final_mask_logits

