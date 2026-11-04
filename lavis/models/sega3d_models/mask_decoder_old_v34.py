import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MaskDecoder(nn.Module):
    def __init__(
        self,
        #decoder=None,
        d_text=512,
        num_layer=6,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        attn_mask=True,
        media=32
    ):
        super().__init__()
        self.num_layer = num_layer
        self.input_proj = nn.Sequential(nn.Linear(media, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.region_mask_proj = nn.Sequential(nn.Linear(1, media), nn.LayerNorm(media), nn.ReLU())
        self.lang_proj = nn.Linear(d_text, d_model)

        self.sa_layers = nn.ModuleList([])
        self.sa_ffn_layers = nn.ModuleList([])
        self.ca_layers = nn.ModuleList([])
        self.ca_ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.sa_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.sa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.ca_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.ca_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(media, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.attn_mask = attn_mask

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)

            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, mask_feats, batch_mask):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_mask):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_scores, pred_masks, attn_masks
    
    def forward(self, sp_feats, batch_offsets, text_features=None, region=None):    
        
        x = sp_feats

        if region != None:
            region_proj = self.region_mask_proj(region)
            x = x + region_proj

        batch_lap_pos_enc = None
        query = self.lang_proj(text_features)              
        inst_feats = self.input_proj(x)
        mask_feats = self.x_mask(x)
        hidden_flat = inst_feats        
        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)
        prediction_masks = []
        prediction_scores = []
        B = len(batch_offsets) - 1
        
        # 0-th prediction
        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)

        # multi-round
        for i in range(self.num_layer):

            if self.attn_mask and (i >= 2):
                cur_attn_mask = attn_masks
            else:
                cur_attn_mask = None

            query = self.sa_layers[i](query, None)
            query = self.sa_ffn_layers[i](query)

            query, _ = self.ca_layers[i](inst_feats, query, batch_mask, cur_attn_mask)
            query = self.ca_ffn_layers[i](query)

            pred_scores, pred_masks, _ = self.prediction_head(query, mask_feats, batch_mask)
            _, _, attn_masks = self.prediction_head(query, mask_feats, batch_mask)

            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
        return {
            'masks':
            pred_masks,
            'batch_mask':
            batch_mask,
            'scores':
            pred_scores,
            'aux_outputs': [{
                'masks': a,
                'scores': b
            } for a, b in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
            )],
            'hidden':
            hidden_flat,
        }

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        output = self.dropout(output) + x
        output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        output = self.net(x)
        output = output + x
        output = self.norm(output)
        return output

class Fourier3DPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 3 个坐标轴，每个轴分配 d_model // 3 的维度
        self.dim_per_axis = d_model // 3
        # 频率分布
        freqs = torch.exp(torch.arange(0, self.dim_per_axis, 2, dtype=torch.float32) * -(math.log(10000.0) / self.dim_per_axis))
        self.register_buffer("freqs", freqs)

    def forward(self, xyz):
        # xyz: [B, I, 3]
        B, I, _ = xyz.shape
        pe = []
        for i in range(3): # 对 x, y, z 分别编码
            pos = xyz[..., i].unsqueeze(-1) # [B, I, 1]
            args = pos * self.freqs # [B, I, dim_per_axis // 2]
            pe.append(torch.sin(args))
            pe.append(torch.cos(args))
        
        pe_tensor = torch.cat(pe, dim=-1) # [B, I, dim_per_axis * 3]
        # 如果 d_model 不能被 3 整除，补齐维度
        pad_dim = self.d_model - pe_tensor.shape[-1]
        if pad_dim > 0:
            pe_tensor = F.pad(pe_tensor, (0, pad_dim))
        return pe_tensor

# ==============================================================================
# 改动点 1: 新增 GeometrySelfAttention
# 以前你用的是 pytorch 原生的 nn.MultiheadAttention，它不支持传入几何 Bias。
# 现在我们需要自己写 Attention 逻辑来注入位置信息。
# ==============================================================================


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[1], k.shape[1])
            output, output_weight = self.attn(query, k, v, key_padding_mask=batch_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight = self.attn(query, k, v, key_padding_mask=batch_mask)
        # 在 CrossAttentionLayer.forward 末尾替换
        output = self.dropout(output)
        output = self.norm(output + query)
        return output, output_weight


class GeometrySelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.nhead = nhead
        self.scale = (d_model // nhead) ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # [升级] 几何感知 MLP: 
        # 输入: 3(xyz) + 1(dist) + 2(sin/cos azimuth) = 6维
        self.geo_mlp = nn.Sequential(
            nn.Linear(9, 64), 
            nn.LayerNorm(64),  # <--- 稳定特征分布的核心，防止 Bias 乱飘
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, nhead)
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, inst_centers, padding_mask=None):
        B, I, D = x.shape
        
        # 1. Standard QKV
        q = self.q_proj(x).view(B, I, self.nhead, D // self.nhead).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(B, I, self.nhead, D // self.nhead).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(B, I, self.nhead, D // self.nhead).permute(0, 2, 1, 3)
        
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        
        # 2. [关键] 计算相对几何特征 (加入方位角)
        c_i = inst_centers.unsqueeze(2)
        c_j = inst_centers.unsqueeze(1)
        rel_pos = c_i - c_j  # [B, I, I, 3]
        
        dist = torch.sqrt((rel_pos ** 2).sum(dim=-1, keepdim=True) + 1e-6)
        
        # 计算水平方位角 (Azimuth) -> 解决左右不分
        # atan2(y, x) -> [-pi, pi]
        azimuth = torch.atan2(rel_pos[..., 1], rel_pos[..., 0]).unsqueeze(-1)
        azi_sin = torch.sin(azimuth)
        azi_cos = torch.cos(azimuth)
        
        # 拼接 6 维几何特征
        # 在计算 azimuth 之前，加一行方向向量：
        direction = rel_pos / dist 

        # 拼接时把 direction 加进去，变成 9 维：
        geo_input = torch.cat([rel_pos, dist, direction, azi_sin, azi_cos], dim=-1)
        # 计算 Bias
        geo_bias = self.geo_mlp(geo_input).permute(0, 3, 1, 2) # [B, H, I, I]
        
        # 注入 Attention
        attn_weights = attn_weights + geo_bias
        
        if padding_mask is not None:
            mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask_expanded, -1e4)
            
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = (attn_probs @ v).permute(0, 2, 1, 3).reshape(B, I, D)
        
        return self.norm(x + self.dropout(self.out_proj(out)))

class GeometryAwareInstSelector(nn.Module):
    def __init__(self, d_txt, d_inst, d_model=256, nhead=8, num_layer=3, max_num_inst=200):
        super().__init__()
        self.d_model = d_model
        # =====================================================================
        # [方案二：Learnable Token Pooling] 替代粗暴的 Mean Pooling
        # =====================================================================
        # 1. 声明一个可学习的 CLS Token 参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_inst))
        
        # 2. 用于从 32 个点中提取特征的 Cross-Attention
        self.pool_attn = nn.MultiheadAttention(embed_dim=d_inst, num_heads=8, batch_first=True)
        self.pool_norm1 = nn.LayerNorm(d_inst)
        
        # 3. 提取后的 FFN 消化层
        self.pool_ffn = nn.Sequential(
            nn.Linear(d_inst, d_inst * 2),
            nn.GELU(),
            nn.Linear(d_inst * 2, d_inst)
        )
        self.pool_norm2 = nn.LayerNorm(d_inst)
        
        # =====================================================================
        # Stage 1: Path A (Semantic Scout) - 语义侦察兵
        # =====================================================================
        # 只负责看 [SEG] 特征，找“是什么”
        self.scout_txt_proj = nn.Linear(d_txt, d_model, bias=False)  # 注意这里输入是 d_txt (不是2倍)
        self.scout_inst_proj = nn.Linear(d_inst, d_model, bias=False)
        self.scout_scale = 1.0 / math.sqrt(d_model)
        
        # =====================================================================
        # Bridge: 将 A 的分数注入 B
        # =====================================================================
        # 把 A 算出来的 scalar score 映射成 vector，作为 B 的先验提示
        self.score_encoder = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # =====================================================================
        # Stage 2: Path B (Spatial Refiner) - 空间精修师
        # =====================================================================
        # 只负责看 [LOC] 特征，找“在哪里”
        self.refine_loc_proj = nn.Sequential(nn.Linear(d_txt, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.refine_inst_proj = nn.Sequential(nn.Linear(d_inst, d_model), nn.LayerNorm(d_model))
        
        # 绝对坐标编码 (给 B 开天眼，让它知道绝对位置)
        self.center_proj = nn.Sequential(nn.Linear(3, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.bbox_proj = nn.Sequential(nn.Linear(6, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        self.layers = nn.ModuleList()
        for _ in range(num_layer):
            self.layers.append(nn.ModuleDict({
                'sa': GeometrySelfAttention(d_model, nhead),  
                'ca': CrossAttentionLayer(d_model, nhead),    
                'ffn': FFN(d_model, 1024)
            }))

        # 最终打分头：只预测残差 (Correction)
        self.correction_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        
        self.fixed_scale = 10.0 
    
    def forward(self, seg_features, loc_features, inst_tokens, inst_atts, inst_centers,inst_bboxes,dense_text_feats=None, text_mask=None, inst_atts_fine=None, use_token_pooling=True,train=False):
        B, I = inst_tokens.shape[:2]
        device = inst_tokens.device

        # 数据准备
        seg = seg_features.squeeze(1)  # [B, D_txt]
        loc = loc_features.squeeze(1)  # [B, D_txt]

        # ================== 实例特征池化 ==================
        if inst_tokens.dim() == 4:
            B, I, K, D = inst_tokens.shape

            # 统一的 mask 准备（True 表示 padding 点）
            if inst_atts_fine is not None and inst_atts_fine.dim() == 3:
                mask = (inst_atts_fine == 0)  # [B, I, K]
            else:
                point_sum = inst_tokens.abs().sum(dim=-1)  # [B, I, K]
                mask = (point_sum < 1e-6)

            if use_token_pooling:
                # ---------- Learnable Token Pooling ----------
                x_flat = inst_tokens.view(B * I, K, D)
                cls_queries = self.cls_token.expand(B * I, -1, -1)  # [B*I, 1, D]
                key_padding_mask = mask.view(B * I, K)              # [B*I, K]

                # 防止全被 mask 导致 NaN（关键保护）
                all_masked = key_padding_mask.all(dim=1)
                key_padding_mask[all_masked, 0] = False

                attn_out, _ = self.pool_attn(
                    query=cls_queries,
                    key=x_flat,
                    value=x_flat,
                    key_padding_mask=key_padding_mask
                )
                # 注意：原代码中重复了一行，这里只保留一次

                pooled = self.pool_norm1(cls_queries + attn_out)
                pooled = self.pool_norm2(pooled + self.pool_ffn(pooled))
                inst_pooled = pooled.view(B, I, D)

            else:
                # ---------- Max Pooling ----------
                # 将 padding 点的特征置为 -inf，确保它们不会影响 max 结果
                masked_features = inst_tokens.masked_fill(mask.unsqueeze(-1), float('-inf'))
                inst_pooled, _ = masked_features.max(dim=2)  # [B, I, D]

                # 安全处理：如果某个物体的所有点都是 padding，max 结果全为 -inf，将其替换为 0 向量
                all_padding = mask.all(dim=2)  # [B, I]
                inst_pooled = torch.where(all_padding.unsqueeze(-1),
                                        torch.zeros_like(inst_pooled),
                                        inst_pooled)
        else:
            inst_pooled = inst_tokens

        # ================== 后续步骤完全不变 ==================
        # Step 1: Semantic Scouting
        s_txt = self.scout_txt_proj(seg)          # [B, D]
        s_inst = self.scout_inst_proj(inst_pooled) # [B, I, D]
        base_logits = (s_inst * s_txt.unsqueeze(1)).sum(dim=-1) * self.scout_scale
        base_logits = base_logits * 30.0

        # Step 2: Priority Injection
        #prior_feat = self.score_encoder(base_logits.unsqueeze(-1))  # [B, I, D]
        prior_feat = self.score_encoder(base_logits.detach().unsqueeze(-1))


        #pos = self.inst_pos_embed(torch.arange(I, device=device).unsqueeze(0).expand(B, -1))

        # =====================================================================
        # Step 3: Spatial Refining (引入 Dense Text)
        # =====================================================================
        if dense_text_feats is not None:
            # [B, L_text, D_txt] -> [B, L_text, D_model]
            t_loc_seq = self.refine_loc_proj(dense_text_feats)
            loc_bias = self.refine_loc_proj(loc).unsqueeze(1)    # [B,1,d_model]
            t_loc_seq = t_loc_seq + loc_bias  
            # PyTorch 的 key_padding_mask: True 表示屏蔽该词 (把 0 变成 True)
            key_padding_mask = (text_mask == 0) 
        else:
            # 防御性兼容：如果没有传密集特征，退化为单 Token
            t_loc_seq = self.refine_loc_proj(loc).unsqueeze(1) 
            key_padding_mask = None

        t_inst = self.refine_inst_proj(inst_pooled)  # [B, I, D]
        #abs_pos = self.center_proj(inst_centers)  # [B, I, D]
        # 🚀 [防精度爆炸补丁]：强行把输入转成和投影层权重一模一样的数据类型！
        target_dtype = next(self.bbox_proj.parameters()).dtype
        inst_bboxes = inst_bboxes.to(target_dtype)


        abs_pos = self.bbox_proj(inst_bboxes)
        x = t_inst + abs_pos + prior_feat # + pos
        padding_mask = (inst_atts == 0)

        for layer in self.layers:
            # 1. SA: 3D 看 3D
            x = layer['sa'](x, inst_centers=inst_centers, padding_mask=padding_mask)
            
            # 2. CA: 3D 看 全局文本 (调用你修好 Bug 的 CrossAttentionLayer)
            # 注意参数对应关系：source(Key/Value) = t_loc_seq, query = x
            x, _ = layer['ca'](
                source=t_loc_seq, 
                query=x, 
                batch_mask=key_padding_mask, 
                pe=None
            )
            
            # 3. FFN
            x = layer['ffn'](x)

        correction_logits = self.correction_head(x).squeeze(-1) * self.fixed_scale
        final_logits = base_logits + 2 * correction_logits
        final_logits = final_logits.masked_fill(padding_mask, -1e9)

        # 替换 txt_feats 那行
        if text_mask is not None:
            w = text_mask.float().unsqueeze(-1)              # [B, L, 1]
            denom = w.sum(dim=1, keepdim=True).clamp(min=1.0)
            txt_mean = (t_loc_seq * w).sum(dim=1, keepdim=True) / denom
        else:
            txt_mean = t_loc_seq.mean(dim=1, keepdim=True)
        

        return final_logits, {
            "inst_feats": x,
            "txt_feats": txt_mean,
            "base_logits": base_logits,
            "corr_logits": correction_logits
        }



