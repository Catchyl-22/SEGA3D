
import torch.nn as nn
import torch
import math
class SinusoidalPositionalEncoding3D(nn.Module):
    def __init__(self, in_channels=6, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.in_channels = in_channels
        # 编码后的维度：原维度 + 2 * 原维度 * 频段数
        self.out_channels = in_channels * (2 * num_freqs + 1)
        
        # 生成频率分布：2^0, 2^1, ..., 2^(num_freqs-1) 乘以 pi
        freq_bands = (2 ** torch.arange(num_freqs).float()) * math.pi
        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        # 输入 x: [B, I_max, 6] 
        out = [x]
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1) # 输出: [B, I_max, in_channels * (2*num_freqs+1)]
    

class VisualTextCrossAttention(nn.Module):
    """
    轻量级跨模态融合插件：让候选框 (Query) 主动去阅读密集文本词向量 (Key/Value)
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, vis_query, text_kv, text_padding_mask=None):
        # vis_query: [B, N_vis, D]
        # text_kv: [B, L_text, D]
        # text_padding_mask: [B, L_text] (True 表示是 padding，不要看)
        attn_out, _ = self.cross_attn(
            query=vis_query,
            key=text_kv,
            value=text_kv,
            key_padding_mask=text_padding_mask
        )
        x = self.norm1(vis_query + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x