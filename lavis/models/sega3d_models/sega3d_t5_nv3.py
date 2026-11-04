import logging
import os
import math
#oldv1
import torch
import torch.nn as nn
import torch.nn.functional as F
import gorilla
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
import torch.distributed as dist
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.common.dist_utils import download_cached_file
from lavis.models.sega3d_models.Qformer import BertConfig, BertLMHeadModel
from lavis.models.sega3d_models.modeling_t5 import T5Config, T5ForConditionalGeneration
#from lavis.models.sega3d_models.mask_decoder import MaskDecoder
#from lavis.models.sega3d_models.mask_decoder4ins_new import MaskDecoder , TransformerInstSelector ,GeometryAwareInstSelector
from lavis.models.sega3d_models.mask_decoder_old_v34 import MaskDecoder ,GeometryAwareInstSelector
from lavis.models.sega3d_models.seg_loss import Criterion, inst_ce_loss, InfoNCELoss, FocalLoss, lovasz_hinge, dice_loss
from lavis.models.sega3d_models.controller import Iterative3DMaskDecoder
from lavis.models.sega3d_models.pos3dencoding import SinusoidalPositionalEncoding3D
from lavis.common.utils import is_url
import contextlib
from lavis.models.sega3d_models.seg_loss import get_iou
from torch_cluster import fps
#from lavis.models.sega3d_models.guided_mask_decoder import GuidedMaskDecoder

from collections import OrderedDict
import torch_scatter
import time
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch_scatter import scatter_mean
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class SEGA3DT5(BaseModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
    }

    def __init__(
        self,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        apply_lemmatizer=False,
        point_encoder_cfg=None,
        mask_decoder_cfg=None,
        mask_decoder=None,
        seg_criterion_cfg=None,
        pred_confidence=0.5
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_seg() result with lemmas.
        """
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        # PyTorch 2.x:
        torch.set_float32_matmul_precision("high") 

        from lavis.models.sega3d_models.point_extractor import PointExtractor
        self.encoder = PointExtractor(**point_encoder_cfg)
        self.mask_decoder = MaskDecoder(**mask_decoder_cfg)
        self.region_decoder = MaskDecoder(**mask_decoder_cfg)


  
        gorilla.load_checkpoint(self.encoder, point_encoder_cfg["pretrained"], strict=False, map_location='cpu')
        
        self.pc_adapter = nn.Linear(point_encoder_cfg["media"], 1408)


        # ====================================================
        # 🚀 [大改动 6] 前端几何注入：NeRF 风格 BBox 编码器
        # ====================================================
        self.num_freqs = 10  # 频段数
        self.bbox_pe_dim = 6 * (2 * self.num_freqs + 1) # 6 * 21 = 126
        self.bbox_pe = SinusoidalPositionalEncoding3D(in_channels=6, num_freqs=self.num_freqs)
        
        # 将高频编码后的 126 维特征，投影到 1408 维 (与 pc_adapter 输出对齐)
        self.bbox_proj = nn.Sequential(
            nn.Linear(self.bbox_pe_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1408) 
        )


        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)

        num_added_tokens = self.t5_tokenizer.add_tokens("[SEG]")
        self.t5_tokenizer.add_tokens("[LOC]")
        self.seg_token_idx = self.t5_tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.loc_token_idx = self.t5_tokenizer("[LOC]", add_special_tokens=False).input_ids[0]

        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = "gelu"
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)

        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))


        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data

        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)

        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
        
        self.prompt = ""
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        in_dim = self.t5_model.config.hidden_size
        out_dim = mask_decoder_cfg["d_text"]
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]
        loc_text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0),
        ]

        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.loc_text_hidden_fcs = nn.ModuleList([nn.Sequential(*loc_text_fc)])
        
        self.text_hidden_fcs.train()
        self.loc_text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True
        for param in self.loc_text_hidden_fcs.parameters():
            param.requires_grad = True
        self.criterion = Criterion(**seg_criterion_cfg)
        self.pred_confidence = pred_confidence
        self.global_step=0
        # self.use_guided = True
        self.debug = False  # 开关：需要时开，不需要时关
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.loc_text_hidden_fcs = nn.ModuleList([nn.Sequential(*loc_text_fc)])
        # ====== instance selector (multi-token ready) ======
        # 文本特征维度 = mask_decoder_cfg["d_text"]
        d_txt = out_dim
        # instance token 原始维度 = point_encoder_cfg["media"] （也就是 p_feats_raw 的通道数）
        d_inst = int(point_encoder_cfg["media"])

        # SOTA 模型内部通常使用 256 或 512 维
        transformer_dim = 256  

        # 2. 初始化 Geometry-Aware Selector
        self.inst_selector = GeometryAwareInstSelector(
            d_txt=d_txt,
            d_inst=d_inst,
            d_model=transformer_dim, # Transformer 内部隐层维度
            nhead=8,                 # 多头注意力头数
            num_layer=6,             # SOTA 通常堆叠 3-6 层，显存不够先用 3
            max_num_inst=200         # ScanNet 场景通常不超过 100 个实例，200 很安全
        )
        self.contrastive_loss = InfoNCELoss(temperature=0.07)
        self.focal_loss = FocalLoss(gamma=2.0)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)
        self.contrastive_loss = InfoNCELoss(temperature=0.07)
        self.focal_loss = FocalLoss(gamma=2.0)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

        # ====================================================
        # [新增] Dynamic MLP Refinement Module
        # ====================================================
        self.point_in_dim = int(point_encoder_cfg["media"]) + 3
        self.inst_in_dim = 256  # Transformer 内部隐层维度
        self.hidden_dim = 64

        self.num_params = (self.point_in_dim * self.hidden_dim + self.hidden_dim) + \
                          (self.hidden_dim * 1 + 1)

        self.iterative_decoder = Iterative3DMaskDecoder(d_model=256, nhead=8, num_layers=3)
        # ====================================================
        # 🚀 [大改动 1] 新增：Loopback 二审打分头 (Feature Re-pooling Score Head)
        # ====================================================
        # 接收: 重新聚合的纯净点云特征 (C维) + 文本特征 (D维) -> 输出: 最终确信度
        raw_feat_dim = int(point_encoder_cfg["media"])
        text_feat_dim = mask_decoder_cfg["d_text"] # 通常是 256
        
        self.loopback_fusion = nn.Sequential(
            nn.Linear(raw_feat_dim + 6 + text_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.spatial_radius_sq = 2.0 ** 2

        # 👇 [新增这一行] 用于将 512维的文本特征降维到 256维，对齐视觉特征
        self.text_guide_proj = nn.Linear(512, 256)

        self.bbox_head = nn.Sequential(
            nn.Linear(self.inst_in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6) # 输出 [cx, cy, cz, w, h, l]
        )
        # 🚀 [新增融合组件]: 后归一化与可学习门控
        self.fusion_norm = nn.LayerNorm(1408)
        # 初始化为 0.1，让模型初期先看形状，慢慢再学位置
        self.pos_alpha = nn.Parameter(torch.tensor(0.1))

        # ====================================================
        # [无超点高密度区域 tokenizer]
        # ====================================================
        self.scene_token_budget = 3072
        self.region_grid_candidates = (12, 14, 16, 18, 20, 22, 24)
        self.use_global_scene_token = False

        # ====================================================
        # [env patch / selector 路径配置]
        # ====================================================
        self.patch_ms_ks = (16, 32, 64)
        self.patch_ms_weights = (0.2, 0.3, 0.5)
        self.patch_ref_k = 32
        self.region_prepool_mlp = nn.Sequential(
            nn.Linear(point_encoder_cfg["media"] + 3, point_encoder_cfg["media"]),
            nn.LayerNorm(point_encoder_cfg["media"]),
            nn.ReLU(inplace=True),
            nn.Linear(point_encoder_cfg["media"], point_encoder_cfg["media"]),
        )
        # ====================================================
        # [LOC patch prior 分支配置]  —— 第一版不改 selector，只给 refine 一个 coarse prior
        # ====================================================
        self.loc_patch_k_ctx = 8              # 每个候选只看最近 K 个 patch
        self.loc_patch_hidden = 256
        self.loc_region_loss_weight = 0.2
        self.loc_logit_alpha = nn.Parameter(torch.tensor(0.1))
        self.loc_prior_alpha = nn.Parameter(torch.tensor(0.2))

        raw_feat_dim = int(point_encoder_cfg["media"])   # 你当前是 32
        loc_txt_dim = mask_decoder_cfg["d_text"]         # 通常是 256
        inst_dim = self.inst_in_dim                       # 256

        # patch feature -> hidden
        self.loc_patch_feat_proj = nn.Sequential(
            nn.Linear(raw_feat_dim, self.loc_patch_hidden),
            nn.LayerNorm(self.loc_patch_hidden),
            nn.ReLU(inplace=True),
        )

        # query = [loc_feat, cand_feat, cand_bbox]
        self.loc_patch_query_proj = nn.Sequential(
            nn.Linear(loc_txt_dim + inst_dim + 6, self.loc_patch_hidden),
            nn.LayerNorm(self.loc_patch_hidden),
            nn.ReLU(inplace=True),
        )

        # 几何偏置：rel_center(3) + rel_size(3) + dist(1) + direction(3) = 10
        self.loc_patch_geo_mlp = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # patch 级打分
        self.loc_patch_score_head = nn.Linear(self.loc_patch_hidden, 1, bias=False)

        # 把 point-wise prior(logit/prob) 投回点特征维度，注入 refine
        self.loc_prior_proj = nn.Sequential(
            nn.Linear(1, raw_feat_dim),
            nn.LayerNorm(raw_feat_dim),
            nn.ReLU(inplace=True),
        )
        self.enable_reloop_train = True
        self.enable_reloop_eval = True
        self.disable_reloop_for_sem = True
        self.disable_reloop_for_mixed = True
        self.default_candidate_mode = "ins"   # ins / sem / mixed
        self.use_reloop = False




    def forward(self, samples):
        try:
            with self.maybe_autocast():
                answer = samples["answer"]
                text_input = samples["text_input"]
                n_answers = samples["n_answers"]
                scan_ids = samples["scan_ids"]   # list[str], len=B
                device = samples["feats"].device if torch.is_tensor(samples.get("feats", None)) else torch.device("cuda")
                xyz = samples["coord_float"]
                sp_feats_raw, p_feats_raw = self.encoder(samples)
                point_offsets = samples["point_offsets"].to(p_feats_raw.device).long()  # [B+1]
                B = int(point_offsets.numel()) - 1
                C = int(p_feats_raw.shape[-1])
                K = getattr(self, "inst_token_k", 32)
                inst_tokens_fine, inst_atts_fine, inst_tokens_pool, inst_atts, inst_centers, I_max = self._extract_instance_tokens(
                    p_feats_raw, xyz, point_offsets, samples["inst2p_offsets_full"], samples["inst2p_indices_full"]
                )
                samples.update({
                    "inst_tokens_fine": inst_tokens_fine, "inst_atts_fine": inst_atts_fine,
                    "inst_tokens": inst_tokens_pool, "inst_atts": inst_atts, "inst_centers": inst_centers
                })

                B = inst_tokens_fine.shape[0]
                # ==========================================================
                # 🌊 流 A: 极速白嫖 padded_bbox (专供下游 Selector 使用)
                # ==========================================================
                padded_bbox = None
                inst_bbox_list = samples.get("inst_bbox_full", None)
                if inst_bbox_list is not None:
                    # 💡 极其聪明的一招：传入全 0 的 dummy 占位符，跳过沉重的 adapter 运算，只索取算好的 bbox
                    dummy_embeds = torch.zeros((B, I_max * K, 1408), device=device)
                    _, padded_bbox = self._fuse_bbox_and_vision(
                        dummy_embeds, inst_bbox_list, point_offsets, xyz, B, I_max, K, device, is_training=True
                    )
                pc_embeds, image_atts = self._extract_pyramid_features(
                    p_feats_raw, xyz, point_offsets,
                    inst_tokens_pool, inst_atts,
                    inst_bbox_list, B, device, is_training=True
                )

            query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)  # 768 #2, 32, 768
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=pc_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)
            if self.prompt:
                text_input = [self.prompt.format(question) for question in text_input]
            else:
                text_input = text_input
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                input_tokens = self.t5_tokenizer(
                    text_input,
                    padding="longest",
                    truncation=True,
                    max_length=200,
                    return_tensors="pt",
                ).to(pc_embeds.device)
                output_tokens = self.t5_tokenizer(
                    answer,
                    padding="longest",
                    truncation=True,
                    max_length=100,
                    return_tensors="pt",
                ).to(pc_embeds.device)
                batch_input_tokens_input_ids = []
                batch_input_tokens_atts = []
                batch_atts_t5 = []
                batch_inputs_t5 = []

                for b, n in enumerate(n_answers):
                    batch_input_tokens_input_ids += [input_tokens.input_ids[b]] * n
                    batch_input_tokens_atts += [input_tokens.attention_mask[b]] * n
                    batch_atts_t5 += [atts_t5[b]] * n
                    batch_inputs_t5 += [inputs_t5[b]] * n

                batch_input_tokens_input_ids = torch.stack(batch_input_tokens_input_ids, dim=0)
                batch_input_tokens_atts = torch.stack(batch_input_tokens_atts, dim=0)
                batch_atts_t5 = torch.stack(batch_atts_t5, dim=0)
                batch_inputs_t5 = torch.stack(batch_inputs_t5, dim=0)

                encoder_atts = torch.cat([batch_atts_t5, batch_input_tokens_atts], dim=1)

                targets = output_tokens.input_ids.masked_fill(
                    output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
                )
                inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
                inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
                outputs = self.t5_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    decoder_attention_mask=output_tokens.attention_mask,
                    return_dict=True,
                    labels=targets,
                    output_hidden_states=True
                )

                # =======================================================
                # 🚀 [新增] 提取真正的 Dense Text Features (训练侧)
                # =======================================================
                # T5 Encoder 输出包含了 [视觉 Tokens + 文本 Tokens]
                # 我们通过 vis_len 将前面的视觉特征切掉，只保留文本词向量
                vis_len = batch_inputs_t5.shape[1]
                dense_text_raw = outputs.encoder_last_hidden_state[:, vis_len:, :] # [B, L_text, H_t5]
                
                # 使用你已有的 MLP 降维，保持特征空间对齐
                dense_text_proj = self.loc_text_hidden_fcs[0](dense_text_raw) # [B, L_text, D_txt]
                
                # 提取文本的 padding mask (1表示有效词，0表示Pad)
                text_mask = batch_input_tokens_atts 
                # =======================================================

                seq_out = outputs["decoder_hidden_states"][-1]
                seg_token_index = targets == self.seg_token_idx
                loc_token_index = targets == self.loc_token_idx
            
                seg_out = seq_out[seg_token_index]
                loc_out = seq_out[loc_token_index]

                seg_features = self.text_hidden_fcs[0](seg_out).unsqueeze(1)
                loc_features = self.loc_text_hidden_fcs[0](loc_out).unsqueeze(1)

                samples["text_features"] = seg_features
                samples["loc_text_features"] = loc_features
                inst_bboxes_norm = padded_bbox if 'padded_bbox' in locals() else None

                N_env = 64
                K_env = K

                env_tokens_fine = torch.zeros((B, N_env, K, C), device=device)
                env_atts_fine = torch.zeros((B, N_env, K), dtype=torch.long, device=device)
                env_atts = torch.zeros((B, N_env), dtype=torch.long, device=device)
                env_centers = torch.zeros((B, N_env, 3), device=device)
                env_bboxes = torch.zeros((B, N_env, 6), device=device)

                scene_patch_feats_bank = []
                scene_patch_centers_bank = []
                scene_patch_bboxes_bank = []
                scene_patch_knn_bank = []

                for b in range(B):
                    p0, p1 = int(point_offsets[b].item()), int(point_offsets[b + 1].item())
                    if p1 <= p0:
                        scene_patch_feats_bank.append(p_feats_raw.new_zeros((0, C)))
                        scene_patch_centers_bank.append(xyz.new_zeros((0, 3)))
                        scene_patch_bboxes_bank.append(xyz.new_zeros((0, 6)))
                        scene_patch_knn_bank.append(torch.zeros((0, 0), dtype=torch.long, device=device))
                        continue

                    xyz_b = xyz[p0:p1]
                    p_feats_b = p_feats_raw[p0:p1]

                    patch_feats, patch_centers, patch_bbox_raw, knn_idx = self._build_patch_tokens(
                        p_feats_b=p_feats_b,
                        xyz_b=xyz_b,
                        num_seed=N_env,
                        k_list=self.patch_ms_ks,
                        k_ref=K_env,
                        random_start=self.training,
                    )
                    scene_patch_feats_bank.append(patch_feats)
                    scene_patch_centers_bank.append(patch_centers)
                    scene_patch_bboxes_bank.append(patch_bbox_raw)
                    scene_patch_knn_bank.append(knn_idx)

                    n_act = patch_feats.shape[0]
                    k_eff = knn_idx.shape[1]

                    env_tokens_fine[b, :n_act, :k_eff, :] = p_feats_b[knn_idx]
                    env_atts_fine[b, :n_act, :k_eff] = 1
                    env_atts[b, :n_act] = 1
                    env_centers[b, :n_act, :] = patch_centers

                    scene_min = xyz_b.min(dim=0)[0]
                    scene_max = xyz_b.max(dim=0)[0]
                    scene_size = (scene_max - scene_min).clamp(min=1e-3)

                    env_bboxes[b, :n_act, :3] = (patch_bbox_raw[:, :3] - scene_min) / scene_size
                    env_bboxes[b, :n_act, 3:] = patch_bbox_raw[:, 3:] / scene_size

                    if self.training:
                        env_bboxes[b, :n_act, :] += (torch.rand_like(env_bboxes[b, :n_act, :]) - 0.5) * 0.05

                # 🧩 拼接
                cmb_tokens_fine = torch.cat([samples["inst_tokens_fine"], env_tokens_fine], dim=1)
                cmb_atts_fine = torch.cat([samples["inst_atts_fine"], env_atts_fine], dim=1)
                cmb_atts = torch.cat([samples["inst_atts"], env_atts], dim=1)
                cmb_centers = torch.cat([samples["inst_centers"], env_centers], dim=1)
                cmb_bboxes = torch.cat([inst_bboxes_norm, env_bboxes], dim=1) if inst_bboxes_norm is not None else None
                # [请在 forward 和 predict_seg 两处都这样改]
                # 更改为传入拼接后的变量
                inst_logits, aux_outputs = self.inst_selector(
                    seg_features=seg_features,
                    loc_features=loc_features,
                    dense_text_feats=dense_text_proj, 
                    text_mask=text_mask,
                    inst_tokens=cmb_tokens_fine,     # <-- 改这里
                    inst_atts=cmb_atts,              # <-- 改这里
                    inst_centers=cmb_centers,        # <-- 改这里
                    inst_atts_fine=cmb_atts_fine,    # <-- 改这里
                    use_token_pooling=True,
                    inst_bboxes=cmb_bboxes,          # <-- 改这里
                    train=True # predict_seg 中记得改 False
                )
                # =======================================================
                # 🌟 插入点 3：剥离环境锚点，只保留真实的候选框，绝不污染 Loss
                # =======================================================
                I_max_true = samples["inst_tokens_fine"].shape[1]
                inst_logits = inst_logits[:, :I_max_true] 
                
                if "base_logits" in aux_outputs:
                    aux_outputs["base_logits"] = aux_outputs["base_logits"][:, :I_max_true]
                if "inst_feats" in aux_outputs:
                    aux_outputs["inst_feats"] = aux_outputs["inst_feats"][:, :I_max_true]
                if "corr_logits" in aux_outputs:
                    aux_outputs["corr_logits"] = aux_outputs["corr_logits"][:, :I_max_true]

                # [新增] 获取 Path A 的原始分数 (用于后续监督语义)
                base_logits = aux_outputs.get("base_logits", inst_logits)

                # [修改] 统一做 Mask 处理
                inst_mask = (samples["inst_atts"] == 0)
                inst_logits_masked = inst_logits.masked_fill(inst_mask, -1e9)
                base_logits_masked = base_logits.masked_fill(inst_mask, -1e9)

                samples["inst_logits"] = inst_logits_masked
                # 获取用于 Contrastive Loss 的特征
                # 如果 aux_outputs 里没有 (比如用了旧 selector)，则跳过 CL
                inst_feats_final = aux_outputs.get("inst_feats", None)
                txt_feats_final  = aux_outputs.get("txt_feats", None)

                B = int(samples["inst_tokens"].shape[0])

                # squeeze last dim if [*, I, 1]
                if inst_logits.dim() == 3 and inst_logits.size(-1) == 1:
                    inst_logits = inst_logits.squeeze(-1)  # [*, I]

                if inst_logits.shape[0] != B:
                    # inst_logits is expanded over answers: [sum(n_answers), I]
                    n_answers = samples["n_answers"]
                    if torch.is_tensor(n_answers):
                        n_answers = n_answers.tolist()

                    logits_list = []
                    s = 0
                    for b, n in enumerate(n_answers):
                        n = int(n)
                        if n <= 0:
                            logits_list.append(inst_logits.new_full((inst_logits.shape[1],), -1e9))
                            continue
                        chunk = inst_logits[s:s+n]                 # [n, I]
                        # 训练/推理都可以先用 max（任一答案强匹配即可）
                        logits_list.append(chunk.max(dim=0).values) # [I]
                        s += n

                    inst_logits = torch.stack(logits_list, dim=0)   # [B, I]
                #inst_logits = inst_logits.masked_fill(samples["inst_atts"] == 0, -20.0)


                #lllllllllllllll
                # =================================================================
                # 1. 构建 Target
                # =================================================================
                inst_ref_labels_list = samples["inst_ref_labels"]
                B = inst_logits.shape[0]
                device = inst_logits.device
                
                target_indices = torch.full((B,), -100, dtype=torch.long, device=device)
                
                for b, ref_lbl in enumerate(inst_ref_labels_list):
                    if len(ref_lbl) > 0:
                        pos = (ref_lbl > 0.5).nonzero(as_tuple=True)[0]
                        if len(pos) > 0:
                            target_indices[b] = pos[0].item()

                loss_sem = F.cross_entropy(base_logits_masked, target_indices, ignore_index=-100)

                # --- 2. 最终定位 Loss (Path B) ---
                # 监督 Refiner 找对 "在哪里" (Final Logits)
                # 权重 1.0：在语义正确的基础上，修正位置
                #loss_final = F.cross_entropy(inst_logits_masked, target_indices, ignore_index=-100)

                target_quality = samples["target_quality"].to(inst_logits_masked.device)
                loss_weights = target_quality
                # A. Cross Entropy Loss (Main)
                raw_loss = F.cross_entropy(inst_logits_masked, target_indices, reduction='none', ignore_index=-100) # [B]
                valid_mask = (target_indices != -100).float()
                loss_sum = (raw_loss * loss_weights * valid_mask).sum()
                weight_sum = (loss_weights * valid_mask).sum() + 1e-6
                loss_final = loss_sum / weight_sum

                # --- 3. 对比学习 Loss (InfoNCE) ---
                # 强迫 [LOC] 文本特征与几何特征对齐
                # --- 3. 对比学习 Loss (InfoNCE) + 强力难负样本挖掘 ---
                loss_cl = torch.tensor(0.0, device=device)
                inst_feats_final = aux_outputs.get("inst_feats", None)
                txt_feats_final  = aux_outputs.get("txt_feats", None)
                has_valid_gt = (target_indices != -100).any()

                if inst_feats_final is not None and txt_feats_final is not None and has_valid_gt:
                    import math
                    # 惩罚系数：同类错误物体的排斥力放大 10 倍
                    hard_weight = 10.0 
                    cl_loss_sum = 0.0
                    valid_cl_batches = 0
                    
                    # 获取我们在 dataset 里刚刚加好的类别标签 (这是一个 list)
                    inst_class_ids_list = samples.get("inst_class_ids", None)
                    tau = 0.07 # InfoNCE 温度系数

                    for b in range(B):
                        tgt_idx = target_indices[b].item()
                        if tgt_idx == -100: continue
                        
                        # 当前场景真实的实例数量
                        I_b = len(samples["inst2p_offsets_full"][b]) - 1
                        if tgt_idx >= I_b: continue
                        
                        # 1. 提取并归一化特征
                        # 视觉特征: [I_b, D]
                        i_f = F.normalize(inst_feats_final[b, :I_b], p=2, dim=-1)
                        # 文本特征: [D] (因为前面是 [B, 1, D])
                        t_f = F.normalize(txt_feats_final[b, 0], p=2, dim=-1)

                        # 2. 计算余弦相似度 Logits: [I_b]
                        logits = torch.matmul(i_f, t_f) / tau

                        # 3. 寻找同类别“冒牌货” (Hard Negatives)
                        if inst_class_ids_list is not None:
                            cls_ids = inst_class_ids_list[b].to(device) # [I_b]
                            tgt_class = cls_ids[tgt_idx]
                            
                            # 条件：类别相同 && 不是目标本身 && 类别已知(不为-1)
                            hard_neg_mask = (cls_ids == tgt_class) & (torch.arange(I_b, device=device) != tgt_idx) & (cls_ids != -1)
                            
                            # 4. 实施精准打击！
                            # 数学原理：在 logit 上加上 log(W)，等价于在 Softmax 的分母中将其比重放大 W 倍！
                            if hard_neg_mask.any():
                                logits[hard_neg_mask] += math.log(hard_weight)

                        # 5. 标准 Cross Entropy (拉近目标，推远所有其他，猛烈推远同类)
                        loss_b = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tgt_idx], device=device))
                        cl_loss_sum += loss_b
                        valid_cl_batches += 1

                    if valid_cl_batches > 0:
                        loss_cl = cl_loss_sum / valid_cl_batches

                # --- 4. [关键] 文本正交 Loss (Orthogonality) ---
                sim_text = F.cosine_similarity(seg_features.detach(), loc_features, dim=-1).mean()
                loss_orth = torch.abs(sim_text)

                self.global_step += 1
                # =================================================================
                # 🌟 [大改动 3] Loopback Synergy: Top-K 掩码反哺与二审训练
                # =================================================================

                B_sz, I_sz, D_sz = inst_feats_final.shape
                pred_bboxes_all = self.bbox_head(inst_feats_final)

                loss_bbox = torch.tensor(0.0, device=device)
                if inst_feats_final is not None and "target_bboxes" in samples and self.training:
                    # 利用并行的 bbox_head 预测出所有候选框的 3D Box: [B, I, 6]
                    gt_bboxes = samples["target_bboxes"].to(device) # [B, 6]
                    
                    bbox_loss_list = []
                    for b in range(B_sz): 
                        tgt_idx = target_indices[b].item()
                        if tgt_idx == -100: continue # 没有匹配到有效的 GT
                        
                        I_b = inst_feats_final.shape[1]
                        if tgt_idx >= I_b: continue # 越界保护
                            
                        # 1. 提取模型预测的该目标 BBox [6]
                        pred_box = pred_bboxes_all[b, tgt_idx] 
                        
                        # 2. 提取我们在 dataset 中生成的真实目标 BBox [6]
                        gt_box = gt_bboxes[b] 
                        
                        # 如果 gt_box 不是全 0 (在 dataset 里处理过的空 box 防护)
                        if gt_box.abs().sum() > 1e-6:
                            # 3. 使用 L1 Loss 强制预测框与真实框对齐
                            l1_loss = F.l1_loss(pred_box, gt_box)
                            bbox_loss_list.append(l1_loss)
                            
                    if len(bbox_loss_list) > 0:
                        loss_bbox = torch.stack(bbox_loss_list).mean()

                loss_spatial_aux = torch.tensor(0.0, device=device)

                # 把 aux_outputs 里纯靠 Path B 算出来的 correction_logits 拿出来
                corr_logits = aux_outputs.get("corr_logits", None)
                if corr_logits is not None:
                    corr_logits_masked = corr_logits.masked_fill(inst_mask, -1e9)
                    
                    # 强迫空间网络自己独立找出正确的目标
                    raw_spatial_loss = F.cross_entropy(corr_logits_masked, target_indices, reduction='none', ignore_index=-100)
                    loss_spatial_aux = (raw_spatial_loss * loss_weights * valid_mask).sum() / weight_sum

                # --- 总 Loss 加上 2.0 * loss_loopback ---
                loss = outputs.loss + \
                    2.0 * loss_sem + \
                    2.0 * loss_final + \
                    2.0 * loss_cl + \
                    0.1 * loss_orth + \
                    2.0 * loss_bbox + \
                    2.0 * loss_spatial_aux 
                if self._is_master() and (self.global_step % 100 == 0):
                    with torch.no_grad():
                        print(f"\n[DEBUG Step {self.global_step}] Check Performance >>>>>>>>>>>>>>>>")

                        # --- A. 解剖各项 Loss (诊断"权重踩踏"元凶) ---
                        print("  📊 [Loss Breakdown]:")
                        print(f"    - Sem (2.0)     : {loss_sem.item():.4f}")
                        print(f"    - Final (2.0)   : {loss_final.item():.4f}")
                        print(f"    - CL (2.0)      : {loss_cl.item():.4f}")
                        print(f"    - Orth (0.1)    : {loss_orth.item():.4f}")
                        print(f"    - BBox (5.0)    : {loss_bbox.item():.4f}  <-- 盯住它是不是远大于 Refine")
                        print(f"    👉 Total Loss   : {loss.item():.4f}")

                        # --- B. 计算当前 Batch 的准确率 (Batch Acc) ---
                        probs = torch.softmax(inst_logits_masked.float(), dim=1)
                        pred_ids = probs.argmax(dim=1)  # [B]
                        valid_mask = (target_indices != -100)
                        num_valid = valid_mask.sum().item()

                        if num_valid > 0:
                            correct_count = (pred_ids[valid_mask] == target_indices[valid_mask]).sum().item()
                            batch_acc = correct_count / num_valid
                            print(f"\n  🎯 Batch Acc  : {batch_acc:.2%} ({correct_count}/{num_valid})")
                        else:
                            print(f"\n  ⚠️ Batch Acc  : N/A (No valid GT in this batch)")

                        # --- C. 打印 Train mIoU (刚才在循环里截获的) ---
                        if 'train_iou_list' in locals() and len(train_iou_list) > 0:
                            mean_iou = sum(train_iou_list) / len(train_iou_list)
                            mean_pred_iou = sum(pred_iou_list) / len(pred_iou_list)
                            print(f"  😇 Oracle mIoU  : {mean_iou:.2%} (上限能力: 如果全选对能考多少分)")
                            print(f"  🏆 True mIoU    : {mean_pred_iou:.2%} (真实成绩: 包含选错框的综合得分)")
                        else:
                            print(f"  🧩 Train mIoU   : N/A (No valid masks computed in this batch)")

                        print(f"  🚪 Position Gate Alpha: {self.pos_alpha.item():.4f}")

                        # --- D. 检查是否崩塌 (Unique Prediction Count) ---
                        unique_preds = torch.unique(pred_ids)
                        n_unique = len(unique_preds)
                        print(f"  🔍 Unique Preds: {n_unique}/{B} (Diversity Check)")

                        if n_unique == 1 and B > 1:
                            print("  🔴 [WARNING] MODEL COLLAPSED! Outputting same ID!")

                        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")
                return {"loss": loss}
        except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"🔥 Crash at global_step {self.global_step}, saving batch...")
                #torch.save(samples, f"crash_batch_step_{self.global_step}.pt")
                raise e
 

    def predict_seg(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=200,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):
        # =========================
        # predict-only debug / ablation switches
        # =========================
        DEBUG_PRED = False
        ABLATE_ENV = False
        ABLATE_LOC_FEAT = False
        ABLATE_LOC_LOGIT = False

        # 只打印前几个样本，避免日志爆炸
        if not hasattr(self, "_pred_debug_count"):
            self._pred_debug_count = 0
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.bfloat16):
            text_input = samples["text_input"]
            # 1) 原始超点特征（细节最丰富）
            sp_feats_raw,p_feats_raw = self.encoder(samples) # [N_sp, C]
            # ===========================
            # NEW: per-instance subsample K tokens (not single vector pooling)
            # ===========================
            K = getattr(self, "inst_token_k", 32)   # 你也可以直接写死 64/96/100
            device = samples["feats"].device if torch.is_tensor(samples.get("feats", None)) else torch.device("cuda")
            point_offsets = samples["point_offsets"].to(device).long()
            inst2p_offsets_list = samples["inst2p_offsets_full"]
            inst2p_indices_list = samples["inst2p_indices_full"]
            xyz = samples["coord_float"]
            B = int(point_offsets.numel()) - 1
            C = int(p_feats_raw.shape[-1])
            K = getattr(self, "inst_token_k", 32)

            inst_tokens_fine, inst_atts_fine, inst_tokens_pool, inst_atts, inst_centers, I_max = self._extract_instance_tokens(
                    p_feats_raw, xyz, point_offsets, samples["inst2p_offsets_full"], samples["inst2p_indices_full"]
                )
                
            samples.update({
                "inst_tokens_fine": inst_tokens_fine, "inst_atts_fine": inst_atts_fine,
                "inst_tokens": inst_tokens_pool, "inst_atts": inst_atts, "inst_centers": inst_centers
            })


            B = inst_tokens_fine.shape[0]
            pc_embeds = inst_tokens_fine.reshape(B, I_max * K, C)   # [B, I_max*K, C]
            image_atts = inst_atts_fine.reshape(B, I_max * K)       # [B, I_max*K]
            pc_embeds = self.pc_adapter(pc_embeds)

            # ====================================================
            # 🚀 [实施前端注入] 提取 BBox，场景相对归一化 (与 Forward 绝对对齐)
            # ====================================================
            padded_bbox = None
            inst_bbox_list = samples.get("inst_bbox_full", None)
            if inst_bbox_list is not None:
                _, padded_bbox = self._fuse_bbox_and_vision(
                    torch.zeros((B, I_max * K, 1408), device=device), # Dummy 占位符
                    inst_bbox_list, point_offsets, xyz, B, I_max, K, device, is_training=False
                )
            pc_embeds, image_atts = self._extract_pyramid_features(
                p_feats_raw, xyz, point_offsets,
                inst_tokens_pool, inst_atts,
                inst_bbox_list, B, device, is_training=False
            )

        #### QQQQQ
        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        q_dtype = next(self.Qformer.parameters()).dtype  # Qformer 的权重 dtype（通常是 fp32）
        pc_embeds = pc_embeds.to(q_dtype)
        
        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1).to(q_dtype)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if isinstance(text_input, str):
            text_input = [text_input]
        
        if self.prompt:
            text_input = [prompt.format(question) for question in text_input]
        else:
            text_input = text_input

        input_tokens = self.t5_tokenizer(text_input, padding="longest", return_tensors="pt").to(pc_embeds.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        num_beams = 1
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.cuda.amp.autocast(enabled=(self.device != torch.device("cpu")), dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            # =======================================================
            # 🌟 [关键修复] 1. 我们自己手动跑一次 Encoder，完美拿到特征！
            # =======================================================
            encoder_outputs = self.t5_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                output_hidden_states=True,
                return_dict=True
            )

            # 🌟 2. 极其安全地提取 Dense Text Features
            vis_len = inputs_t5.shape[1]
            dense_text_raw = encoder_outputs.hidden_states[-1][:, vis_len:, :] 
            dense_text_proj = self.loc_text_hidden_fcs[0](dense_text_raw)
            text_mask = input_tokens.attention_mask 
            # =======================================================
            # [关键修复] 不再依赖 free generation 提取 [SEG]/[LOC]
            # 直接用固定 decoder probe，和 forward 的 teacher-forced 提取方式对齐
            # =======================================================
            probe_text = ["[SEG] [LOC]"] * B
            probe_tokens = self.t5_tokenizer(
                probe_text,
                padding="longest",
                truncation=True,
                max_length=8,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(pc_embeds.device)

            probe_targets = probe_tokens.input_ids.masked_fill(
                probe_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
            )

            probe_outputs = self.t5_model(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_atts,
                decoder_attention_mask=probe_tokens.attention_mask,
                return_dict=True,
                labels=probe_targets,
                output_hidden_states=True,
                use_cache=False,
            )

            probe_seq = probe_outputs["decoder_hidden_states"][-1]   # [B, L_probe, H]

            probe_seg_index = (probe_targets == self.seg_token_idx)
            probe_loc_index = (probe_targets == self.loc_token_idx)

            seg_out = probe_seq[probe_seg_index]   # 期望 [B, H]
            loc_out = probe_seq[probe_loc_index]   # 期望 [B, H]

            # 保险一点：如果 shape 不对，直接报出来
            if seg_out.shape[0] != B or loc_out.shape[0] != B:
                if self._is_master():
                    print("[PRED PROBE ERROR] seg_out.shape =", tuple(seg_out.shape),
                        "loc_out.shape =", tuple(loc_out.shape), "B =", B)
                raise RuntimeError("Probe extraction for [SEG]/[LOC] failed in predict_seg")

            seg_features = self.text_hidden_fcs[0](seg_out).unsqueeze(1)      # [B,1,D_txt]
            loc_features = self.loc_text_hidden_fcs[0](loc_out).unsqueeze(1)  # [B,1,D_txt]

            samples["text_features"] = seg_features
            samples["loc_text_features"] = loc_features

            # === 3) Instance selection head ===
            # [修改] 1. 把 _ 改成 aux_outputs
            # [请在 forward 和 predict_seg 两处都这样改]
            inst_bboxes_norm = padded_bbox if 'padded_bbox' in locals() else None
            N_env = 0 if ABLATE_ENV else 64
            K_env = K

            env_tokens_fine = torch.zeros((B, N_env, K, C), device=device)
            env_atts_fine = torch.zeros((B, N_env, K), dtype=torch.long, device=device)
            env_atts = torch.zeros((B, N_env), dtype=torch.long, device=device)
            env_centers = torch.zeros((B, N_env, 3), device=device)
            env_bboxes = torch.zeros((B, N_env, 6), device=device)
            scene_patch_feats_bank = []
            scene_patch_centers_bank = []
            scene_patch_bboxes_bank = []
            scene_patch_knn_bank = []

            for b in range(B):
                p0, p1 = int(point_offsets[b].item()), int(point_offsets[b + 1].item())
                if p1 <= p0:
                    scene_patch_feats_bank.append(p_feats_raw.new_zeros((0, C)))
                    scene_patch_centers_bank.append(xyz.new_zeros((0, 3)))
                    scene_patch_bboxes_bank.append(xyz.new_zeros((0, 6)))
                    scene_patch_knn_bank.append(torch.zeros((0, 0), dtype=torch.long, device=device))
                    continue

                xyz_b = xyz[p0:p1]
                p_feats_b = p_feats_raw[p0:p1]

                patch_feats, patch_centers, patch_bbox_raw, knn_idx = self._build_patch_tokens(
                    p_feats_b=p_feats_b,
                    xyz_b=xyz_b,
                    num_seed=N_env,
                    k_list=self.patch_ms_ks,
                    k_ref=K_env,
                    random_start=False,
                )
                scene_patch_feats_bank.append(patch_feats)
                scene_patch_centers_bank.append(patch_centers)
                scene_patch_bboxes_bank.append(patch_bbox_raw)
                scene_patch_knn_bank.append(knn_idx)

                n_act = patch_feats.shape[0]
                k_eff = knn_idx.shape[1]

                env_tokens_fine[b, :n_act, :k_eff, :] = p_feats_b[knn_idx]
                env_atts_fine[b, :n_act, :k_eff] = 1
                env_atts[b, :n_act] = 1
                env_centers[b, :n_act, :] = patch_centers

                scene_min = xyz_b.min(dim=0)[0]
                scene_max = xyz_b.max(dim=0)[0]
                scene_size = (scene_max - scene_min).clamp(min=1e-3)

                env_bboxes[b, :n_act, :3] = (patch_bbox_raw[:, :3] - scene_min) / scene_size
                env_bboxes[b, :n_act, 3:] = patch_bbox_raw[:, 3:] / scene_size

            # 🧩 拼接
            cmb_tokens_fine = torch.cat([samples["inst_tokens_fine"], env_tokens_fine], dim=1)
            cmb_atts_fine = torch.cat([samples["inst_atts_fine"], env_atts_fine], dim=1)
            cmb_atts = torch.cat([samples["inst_atts"], env_atts], dim=1)
            cmb_centers = torch.cat([samples["inst_centers"], env_centers], dim=1)
            cmb_bboxes = torch.cat([inst_bboxes_norm, env_bboxes], dim=1) if inst_bboxes_norm is not None else None
            # 更改为传入拼接后的变量
            inst_logits, aux_outputs = self.inst_selector(
                seg_features=seg_features,
                loc_features=loc_features,
                dense_text_feats=dense_text_proj, 
                text_mask=text_mask,
                inst_tokens=cmb_tokens_fine,     # <-- 改这里
                inst_atts=cmb_atts,              # <-- 改这里
                inst_centers=cmb_centers,        # <-- 改这里
                inst_atts_fine=cmb_atts_fine,    # <-- 改这里
                use_token_pooling=True,
                inst_bboxes=cmb_bboxes,          # <-- 改这里
                train=False # predict_seg 中记得改 False
            )
            # =======================================================
            # 🌟 插入点 3：剥离环境锚点，只保留真实的候选框，绝不污染 Loss
            # =======================================================
            I_max_true = samples["inst_tokens_fine"].shape[1]
            inst_logits = inst_logits[:, :I_max_true] 
            
            if "base_logits" in aux_outputs:
                aux_outputs["base_logits"] = aux_outputs["base_logits"][:, :I_max_true]
            if "inst_feats" in aux_outputs:
                aux_outputs["inst_feats"] = aux_outputs["inst_feats"][:, :I_max_true]
            if "corr_logits" in aux_outputs:
                aux_outputs["corr_logits"] = aux_outputs["corr_logits"][:, :I_max_true]
            
            if inst_logits.dim() == 3 and inst_logits.size(-1) == 1:
                inst_logits = inst_logits.squeeze(-1)

            inst_logits = inst_logits.masked_fill(samples["inst_atts"] == 0, -1e9)
            probs = torch.softmax(inst_logits.float(), dim=1)

        
            # =========================================================
            # NEW: Instance Selection -> Point Cloud Mapping
            # =========================================================
            inst2p_offsets_list = samples["inst2p_offsets_full"]
            inst2p_indices_list = samples["inst2p_indices_full"]
            point_offsets = samples["point_offsets"]
            
            B = inst_logits.shape[0]
            device = inst_logits.device
            
            # 结果列表
            pred_point_masks = []
            pred_scores = []

            # =========================================================
            # Hard disable reloop: selector -> top1 candidate -> point mask
            # =========================================================
            inst2p_offsets_list = samples["inst2p_offsets_full"]
            inst2p_indices_list = samples["inst2p_indices_full"]
            point_offsets = samples["point_offsets"]

            B = inst_logits.shape[0]
            device = inst_logits.device

            pred_point_masks = []
            pred_scores = []

            for b in range(B):
                I_b = len(inst2p_offsets_list[b]) - 1
                p0, p1 = int(point_offsets[b].item()), int(point_offsets[b + 1].item())
                n_points = p1 - p0

                # 默认全负
                mask_b = torch.full((n_points,), -20.0, device=device, dtype=torch.float32)
                best_score = 0.0

                if I_b > 0:
                    probs_b = probs[b, :I_b]

                    # 1. selector 直接选 top1 candidate
                    top1_idx = int(probs_b.argmax().item())
                    best_score = float(probs_b[top1_idx].item())

                    offsets_b = inst2p_offsets_list[b]
                    indices_b = inst2p_indices_list[b]

                    if not torch.is_tensor(offsets_b):
                        offsets_b = torch.as_tensor(offsets_b, dtype=torch.long, device=device)
                    else:
                        offsets_b = offsets_b.to(device).long()

                    if not torch.is_tensor(indices_b):
                        indices_b = torch.as_tensor(indices_b, dtype=torch.long, device=device)
                    else:
                        indices_b = indices_b.to(device).long()

                    # 2. 取出这个 candidate 对应的点索引
                    s = int(offsets_b[top1_idx].item())
                    e = int(offsets_b[top1_idx + 1].item())
                    pts_idx = indices_b[s:e]

                    # 3. 安全过滤，避免越界
                    pts_idx = pts_idx[(pts_idx >= 0) & (pts_idx < n_points)]

                    # 4. 填成点级 mask logits
                    if pts_idx.numel() > 0:
                        mask_b[pts_idx] = 20.0

                    if DEBUG_PRED and self._is_master() and self._pred_debug_count < 20:
                        kdbg = min(8, I_b)
                        top_prob, top_idx = probs_b.topk(kdbg)
                        print("\n[DEBUG-PRED][TOP1 ONLY]")
                        print(f"  sample={b}, I_b={I_b}")
                        print(f"  top1_idx={top1_idx}, best_score={best_score:.4f}")
                        print(f"  top{kdbg}_idx  = {top_idx.tolist()}")
                        print(f"  top{kdbg}_prob = {[round(x, 4) for x in top_prob.tolist()]}")
                        self._pred_debug_count += 1

                pred_scores.append(best_score)
                pred_point_masks.append(mask_b)

            # 3. 封装结果
            result = {
                "masks": pred_point_masks,
                "scores": torch.tensor(pred_scores, device=device),
                "is_point_level": True
            }

            return result


    # =====================================================================
    # [Helper 1] 实例特征提取与补齐 (供 forward 和 predict_seg 共用)
    # =====================================================================
    def _extract_instance_tokens(self, p_feats_raw, xyz, point_offsets, inst2p_offsets_list, inst2p_indices_list):
        B = int(point_offsets.numel()) - 1
        C = int(p_feats_raw.shape[-1])
        K = getattr(self, "inst_token_k", 32)
        device = p_feats_raw.device

        inst_tokens_fine_list, inst_atts_fine_list = [], []
        inst_tokens_pool_list, inst_centers_list = [], []
        inst_lens = []

        for b in range(B):
            p0, p1 = int(point_offsets[b].item()), int(point_offsets[b + 1].item())
            p_feats_b = p_feats_raw[p0:p1]
            xyz_b = xyz[p0:p1]
            N_b = int(p_feats_b.shape[0])

            offsets = inst2p_offsets_list[b]
            indices = inst2p_indices_list[b]

            offsets = offsets.to(device).long() if torch.is_tensor(offsets) else torch.as_tensor(offsets, dtype=torch.long, device=device)
            indices = indices.to(device).long() if torch.is_tensor(indices) else torch.as_tensor(indices, dtype=torch.long, device=device)

            if offsets.numel() < 2:
                inst_tokens_fine_list.append(p_feats_b.new_zeros((0, K, C)))
                inst_atts_fine_list.append(torch.zeros((0, K), dtype=torch.long, device=device))
                inst_tokens_pool_list.append(p_feats_b.new_zeros((0, C)))
                inst_centers_list.append(p_feats_b.new_zeros((0, 3)))
                inst_lens.append(0)
                continue

            I_b = int(offsets.numel()) - 1
            inst_lens.append(I_b)

            tok_fine = p_feats_b.new_zeros((I_b, K, C))
            att_fine = torch.zeros((I_b, K), dtype=torch.long, device=device)

            inst_ids = torch.repeat_interleave(torch.arange(I_b, device=device), offsets[1:] - offsets[:-1])
            M = min(indices.numel(), inst_ids.numel())
            indices, inst_ids = indices[:M], inst_ids[:M]

            # Scatter Mean
            pts_feats = p_feats_b.index_select(0, indices)
            tok_pool = scatter_mean(pts_feats, inst_ids, dim=0, dim_size=I_b)
            current_xyz = xyz_b.index_select(0, indices)
            center_pool = scatter_mean(current_xyz, inst_ids, dim=0, dim_size=I_b)

            # FPS for fine tokens
            for inst_id in range(I_b):
                s, e = int(offsets[inst_id].item()), int(offsets[inst_id + 1].item())
                if e <= s: continue
                pts = indices[s:e]
                pts = pts[(pts >= 0) & (pts < N_b)]
                if pts.numel() == 0: continue
                
                num_pts = pts.numel()
                if num_pts <= K:
                    sel = pts
                else:
                    candidate_xyz = xyz_b[pts]
                    ratio = K / num_pts
                    local_sampled_idx = fps(candidate_xyz, batch=None, ratio=ratio, random_start=False)
                    if local_sampled_idx.numel() > K:
                        local_sampled_idx = local_sampled_idx[:K]
                    sel = pts[local_sampled_idx]
                
                m = sel.numel()
                tok_fine[inst_id, :m] = p_feats_b.index_select(0, sel)
                att_fine[inst_id, :m] = 1

            inst_tokens_fine_list.append(tok_fine)
            inst_atts_fine_list.append(att_fine)
            inst_tokens_pool_list.append(tok_pool)
            inst_centers_list.append(center_pool)

        # Padding
        I_max = max(inst_lens) if len(inst_lens) > 0 else 0
        inst_tokens_fine = p_feats_raw.new_zeros((B, I_max, K, C))
        inst_atts_fine = torch.zeros((B, I_max, K), dtype=torch.long, device=device)
        inst_tokens_pool = p_feats_raw.new_zeros((B, I_max, C))
        inst_atts = torch.zeros((B, I_max), dtype=torch.long, device=device)
        inst_centers = p_feats_raw.new_zeros((B, I_max, 3))

        for b in range(B):
            I_b = inst_lens[b]
            if I_b <= 0: continue
            inst_tokens_fine[b, :I_b] = inst_tokens_fine_list[b]
            inst_atts_fine[b, :I_b] = inst_atts_fine_list[b]
            inst_tokens_pool[b, :I_b] = inst_tokens_pool_list[b]
            inst_atts[b, :I_b] = 1
            inst_centers[b, :I_b] = inst_centers_list[b]

        return inst_tokens_fine, inst_atts_fine, inst_tokens_pool, inst_atts, inst_centers, I_max

    # =====================================================================
    # [Helper 2] 前端 3D BBox 融合 (供 QFormer 之前调用)
    # =====================================================================
    def _fuse_bbox_and_vision(self, pc_embeds, inst_bbox_list, point_offsets, xyz, B, I_max, K, device, is_training):
        padded_bbox = pc_embeds.new_zeros((B, I_max, 6))
        for b in range(B):
            I_b = min(I_max, inst_bbox_list[b].shape[0]) if inst_bbox_list[b] is not None else 0
            if I_b > 0:
                p0, p1 = int(point_offsets[b].item()), int(point_offsets[b + 1].item())
                xyz_b = xyz[p0:p1]
                scene_min, scene_max = xyz_b.min(dim=0)[0], xyz_b.max(dim=0)[0]
                scene_size = (scene_max - scene_min).clamp(min=1e-3)

                raw_bbox = inst_bbox_list[b][:I_b].to(device).float()
                raw_bbox = self._ensure_bbox_center_size(raw_bbox)
                norm_bbox = torch.zeros_like(raw_bbox)
                norm_bbox[:, :3] = (raw_bbox[:, :3] - scene_min) / scene_size
                norm_bbox[:, 3:] = raw_bbox[:, 3:] / scene_size

                if is_training:
                    noise = (torch.rand_like(norm_bbox) - 0.5) * 0.05
                    norm_bbox = norm_bbox + noise

                padded_bbox[b, :I_b] = norm_bbox

        bbox_encoded = self.bbox_pe(padded_bbox)
        bbox_features = self.bbox_proj(bbox_encoded)
        bbox_features = bbox_features.unsqueeze(2).expand(-1, -1, K, -1).reshape(B, I_max * K, 1408)
        return self.fusion_norm(pc_embeds + self.pos_alpha * bbox_features), padded_bbox


    def _extract_pyramid_features(
        self,
        p_feats_raw,
        xyz,
        point_offsets,
        inst_tokens_pool,
        inst_atts,
        inst_bbox_list,
        B,
        device,
        is_training
    ):
        """
        QFormer 主输入:
        region tokens + instance tokens
        """
        pyramid_feats_list = []
        pyramid_bbox_list = []
        pyramid_len_list = []

        target_tokens = self.scene_token_budget

        for b in range(B):
            p0, p1 = int(point_offsets[b].item()), int(point_offsets[b + 1].item())
            feat_b = p_feats_raw[p0:p1]
            xyz_b = xyz[p0:p1]

            scene_min = xyz_b.min(dim=0)[0]
            scene_max = xyz_b.max(dim=0)[0]
            scene_size = (scene_max - scene_min).clamp(min=1e-3)

            # --------------------------------------------------
            # 1) region tokens
            # --------------------------------------------------
            region_feats, region_centers, region_bbox_raw = self._build_region_tokens_voxel(
                p_feats_b=feat_b,
                xyz_b=xyz_b,
                target_tokens=target_tokens,
                grid_candidates=self.region_grid_candidates,
            )

            region_bbox = torch.zeros((region_bbox_raw.shape[0], 6), device=device, dtype=xyz_b.dtype)
            region_bbox[:, :3] = (region_bbox_raw[:, :3] - scene_min) / scene_size
            region_bbox[:, 3:] = region_bbox_raw[:, 3:] / scene_size

            # --------------------------------------------------
            # 2) instance tokens
            # --------------------------------------------------
            I_b = int(inst_atts[b].sum().item())
            if I_b > 0 and inst_bbox_list is not None and inst_bbox_list[b] is not None:
                inst_feats = inst_tokens_pool[b, :I_b]
                raw_bbox = inst_bbox_list[b][:I_b].to(device).float()
                raw_bbox = self._ensure_bbox_center_size(raw_bbox)

                inst_bbox = torch.zeros((raw_bbox.shape[0], 6), device=device, dtype=xyz_b.dtype)
                inst_bbox[:, :3] = (raw_bbox[:, :3] - scene_min) / scene_size
                inst_bbox[:, 3:] = raw_bbox[:, 3:] / scene_size
            elif I_b > 0:
                inst_feats = inst_tokens_pool[b, :I_b]
                inst_bbox = torch.zeros((inst_feats.shape[0], 6), device=device, dtype=xyz_b.dtype)
            else:
                inst_feats = torch.zeros((0, feat_b.shape[-1]), device=device, dtype=feat_b.dtype)
                inst_bbox = torch.zeros((0, 6), device=device, dtype=xyz_b.dtype)

            # --------------------------------------------------
            # 3) 拼接 region + instance
            # --------------------------------------------------
            cat_feats = torch.cat([region_feats, inst_feats], dim=0)
            cat_bbox  = torch.cat([region_bbox, inst_bbox], dim=0)

            if is_training:
                noise = (torch.rand_like(cat_bbox) - 0.5) * 0.05
                cat_bbox = cat_bbox + noise

            pyramid_feats_list.append(cat_feats)
            pyramid_bbox_list.append(cat_bbox)
            pyramid_len_list.append(cat_feats.shape[0])

        max_len = max(pyramid_len_list)
        C = p_feats_raw.shape[-1]

        pc_embeds = p_feats_raw.new_zeros((B, max_len, C))
        image_atts = torch.zeros((B, max_len), dtype=torch.long, device=device)
        padded_pyramid_bbox = torch.zeros((B, max_len, 6), dtype=torch.float32, device=device)

        for b in range(B):
            l = pyramid_len_list[b]
            pc_embeds[b, :l] = pyramid_feats_list[b]
            image_atts[b, :l] = 1
            padded_pyramid_bbox[b, :l] = pyramid_bbox_list[b]

        pc_embeds = self.pc_adapter(pc_embeds)
        bbox_encoded = self.bbox_pe(padded_pyramid_bbox)
        bbox_features = self.bbox_proj(bbox_encoded)
        pc_embeds = self.fusion_norm(pc_embeds + self.pos_alpha * bbox_features)

        return pc_embeds, image_atts
   
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)
            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy
                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        point_encoder_cfg = cfg.get("point_encoder_cfg")
        mask_decoder_cfg = cfg.get("mask_decoder_cfg")
        mask_decoder = cfg.get("mask_decoder")
        seg_criterion_cfg = cfg.get("seg_criterion_cfg")
        pred_confidence = cfg.get("pred_confidence")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")
        prompt = cfg.get("prompt", "")
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            apply_lemmatizer=apply_lemmatizer,
            point_encoder_cfg=point_encoder_cfg,
            mask_decoder_cfg=mask_decoder_cfg,
            mask_decoder = mask_decoder,
            seg_criterion_cfg=seg_criterion_cfg,
            pred_confidence=pred_confidence
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        loacl_dir_of_model = "bert-base-uncased"
        encoder_config = BertConfig.from_pretrained(loacl_dir_of_model)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        #Qformer = BertLMHeadModel.from_pretrained("bert-base-uncased", config=encoder_config)
        Qformer = BertLMHeadModel.from_pretrained(loacl_dir_of_model, config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        print('___________________________________________________')
        print(url_or_filename)
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        # ==========================================================
        # 🚀 智能加载：过滤掉形状不匹配的权重 (专治魔改模型后的 OOM/Mismatch)
        # ==========================================================
        model_state_dict = self.state_dict()
        filtered_state_dict = {}
        
        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape != v.shape:
                print(f"⚠️ [跳过加载] 形状不匹配 {k}: Checkpoint={v.shape}, Model={model_state_dict[k].shape}")
                continue
            filtered_state_dict[k] = v
            
        # 使用过滤后的干净字典进行加载
        msg = self.load_state_dict(filtered_state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    # ===== Debug helpers (DDP-safe, rank0 only) =====
    def _is_master(self):

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def _summarize_tensor(self, x, name, max_items=3):
        try:
            if isinstance(x, list):
                n = len(x)
                if n == 0:
                    return f"{name}: list(len=0)"
                shp0 = tuple(x[0].shape)
                dtype0 = x[0].dtype
                dev0 = x[0].device
                return f"{name}: list(len={n}, first={shp0}, dtype={dtype0}, device={dev0})"
            if torch.is_tensor(x):
                return f"{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}"
            return f"{name}: <{type(x)}>"
        except Exception as e:
            return f"{name}: <summary failed: {e}>"

    @torch.no_grad()
    def _stats_tensor(self, x, name, take_sample=False):
        try:
            if not torch.is_tensor(x): 
                return f"{name}: <not tensor>"
            if x.numel() == 0:
                return f"{name}: numel=0"
            # 避免 half 转换抖动，统计时统一 float32
            xs = x.float()
            if take_sample and xs.numel() > 2_000_000:
                idx = torch.randperm(xs.numel(), device=xs.device)[:1_000_000]
                xs = xs.view(-1)[idx]
            mn = xs.min().item()
            mx = xs.max().item()
            mean = xs.mean().item()
            std = xs.std().item()
            return f"{name}: min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g}"
        except Exception as e:
            return f"{name}: <stats failed: {e}>"

    @torch.no_grad()
    def _fps_select(self, pts, k):
        n = int(pts.shape[0])
        if n == 0:
            return torch.zeros((0,), dtype=torch.long, device=pts.device)
        if n <= k:
            return torch.arange(n, device=pts.device, dtype=torch.long)

        ratio = float(k) / float(n)
        idx = fps(pts, ratio=ratio, random_start=False)
        if idx.numel() > k:
            idx = idx[:k]
        if idx.numel() < k:
            full = torch.arange(n, device=pts.device, dtype=torch.long)
            remain = full[~torch.isin(full, idx)]
            idx = torch.cat([idx, remain[:k - idx.numel()]], dim=0)
        return idx
    
    def _ensure_bbox_center_size(self, raw_bbox: torch.Tensor) -> torch.Tensor:
        """
        raw_bbox: [I,6] could be [cx,cy,cz,sx,sy,sz] OR [minx,miny,minz,maxx,maxy,maxz]
        return:   [I,6] in [cx,cy,cz,sx,sy,sz]
        """
        if raw_bbox.numel() == 0:
            return raw_bbox

        mn = raw_bbox[:, :3]
        mx_or_size = raw_bbox[:, 3:]

        frac_ge = (mx_or_size >= mn).float().mean().item()
        if frac_ge > 0.95:
            mx = mx_or_size
            center = (mn + mx) * 0.5
            size = (mx - mn).clamp(min=1e-3)
            return torch.cat([center, size], dim=1)

        out = raw_bbox.clone()
        out[:, 3:] = out[:, 3:].clamp(min=1e-3)
        return out
    
    def _build_region_tokens_voxel(
        self,
        p_feats_b,
        xyz_b,
        target_tokens=2048,
        grid_candidates=None,
    ):
        """
        轻量版 region tokenizer（方案A）:
        1) voxel/partition 分区
        2) 直接 scatter_mean 得到 region feat / center
        3) 用 scatter_min / scatter_max 得到 region bbox
        4) 若 region 太多，再对 region center 做 FPS 截断

        核心目的：
        - 保留很多 region token
        - 避免逐 region 的 Python for-loop 和 local MLP
        - 采用实例内特征平均池化
        """
        N_b = int(xyz_b.shape[0])
        C = int(p_feats_b.shape[-1])

        if N_b <= 0:
            return (
                p_feats_b.new_zeros((0, C)),
                xyz_b.new_zeros((0, 3)),
                xyz_b.new_zeros((0, 6)),
            )

        if grid_candidates is None:
            grid_candidates = self.region_grid_candidates

        scene_min = xyz_b.min(dim=0)[0]
        scene_max = xyz_b.max(dim=0)[0]
        scene_size = (scene_max - scene_min).clamp(min=1e-6)
        xyz_n = (xyz_b - scene_min) / scene_size

        # -------------------------------------------------
        # 1) 选一个 occupied voxel 数最接近 target_tokens 的 grid
        # -------------------------------------------------
        best_diff = 10**18
        best_inv = None
        best_keys = None

        for g in grid_candidates:
            ijk = torch.floor(xyz_n * g).long().clamp(min=0, max=g - 1)
            keys = ijk[:, 0] * (g * g) + ijk[:, 1] * g + ijk[:, 2]
            uniq, inv = torch.unique(keys, sorted=False, return_inverse=True)
            diff = abs(int(uniq.numel()) - int(target_tokens))
            if diff < best_diff:
                best_diff = diff
                best_inv = inv
                best_keys = uniq

        inv = best_inv
        num_regions = int(best_keys.numel())

        # -------------------------------------------------
        # 2) 直接 avgpool：region feats / region centers
        # -------------------------------------------------
        region_feats = scatter_mean(p_feats_b, inv, dim=0, dim_size=num_regions)          # [R, C]
        region_centers = scatter_mean(xyz_b.float(), inv, dim=0, dim_size=num_regions)    # [R, 3]

        # -------------------------------------------------
        # 3) 用 scatter_min / scatter_max 直接算 bbox
        #    避免每个 region 一个 Python for-loop
        # -------------------------------------------------
        region_min = torch_scatter.scatter_min(xyz_b, inv, dim=0, dim_size=num_regions)[0]  # [R,3]
        region_max = torch_scatter.scatter_max(xyz_b, inv, dim=0, dim_size=num_regions)[0]  # [R,3]

        region_bbox = torch.zeros((num_regions, 6), device=xyz_b.device, dtype=xyz_b.dtype)
        region_bbox[:, :3] = 0.5 * (region_min + region_max)
        region_bbox[:, 3:] = (region_max - region_min).clamp(min=1e-3)

        # -------------------------------------------------
        # 4) 若区域数太多，再压到 target_tokens
        # -------------------------------------------------
        if num_regions > target_tokens:
            keep_idx = self._fps_select(region_centers, target_tokens)
            region_feats = region_feats[keep_idx]
            region_centers = region_centers[keep_idx]
            region_bbox = region_bbox[keep_idx]

        return region_feats, region_centers, region_bbox
    @torch.no_grad()
    def _build_patch_tokens(
        self,
        p_feats_b,
        xyz_b,
        num_seed,
        k_list=None,
        k_ref=None,
        random_start=False,
    ):
        """
        multi-scale patch tokenizer for env tokens
        return:
        patch_feats   : [S, C]
        patch_centers : [S, 3]
        patch_bbox    : [S, 6]
        knn_idx_ref   : [S, K_ref_eff]
        """
        device = xyz_b.device
        N_b = int(xyz_b.shape[0])
        C = int(p_feats_b.shape[-1])

        if k_list is None:
            k_list = self.patch_ms_ks
        k_list = list(sorted(set(int(k) for k in k_list)))
        if k_ref is None:
            k_ref = int(self.patch_ref_k)

        if N_b <= 0 or num_seed <= 0:
            return (
                p_feats_b.new_zeros((0, C)),
                xyz_b.new_zeros((0, 3)),
                xyz_b.new_zeros((0, 6)),
                torch.zeros((0, 0), dtype=torch.long, device=device),
            )

        # 1) FPS 选 seed
        if N_b <= num_seed:
            seed_idx = torch.arange(N_b, device=device, dtype=torch.long)
        else:
            ratio = num_seed / float(N_b)
            seed_idx = fps(xyz_b, ratio=ratio, random_start=random_start)
            if seed_idx.numel() > num_seed:
                seed_idx = seed_idx[:num_seed]
            if seed_idx.numel() < num_seed:
                full = torch.arange(N_b, device=device, dtype=torch.long)
                remain = full[~torch.isin(full, seed_idx)]
                seed_idx = torch.cat([seed_idx, remain[:num_seed - seed_idx.numel()]], dim=0)

        seed_xyz = xyz_b[seed_idx]   # [S,3]
        S = int(seed_xyz.shape[0])

        dist = torch.cdist(seed_xyz.float(), xyz_b.float())   # [S, N_b]

        feats_per_scale = []
        centers_per_scale = []
        bbox_per_scale = []
        knn_dict = {}

        for k in k_list:
            K_eff = min(int(k), N_b)
            knn_idx = dist.topk(k=K_eff, dim=1, largest=False).indices
            knn_dict[int(k)] = knn_idx

            nbr_feats = p_feats_b[knn_idx]   # [S, K_eff, C]
            nbr_xyz = xyz_b[knn_idx]         # [S, K_eff, 3]

            feat_mean = nbr_feats.mean(dim=1)
            xyz_mean = nbr_xyz.mean(dim=1)
            xyz_min = nbr_xyz.min(dim=1)[0]
            xyz_max = nbr_xyz.max(dim=1)[0]
            xyz_size = (xyz_max - xyz_min).clamp(min=1e-3)

            feats_per_scale.append(feat_mean)
            centers_per_scale.append(xyz_mean)
            bbox_per_scale.append(torch.cat([xyz_mean, xyz_size], dim=-1))

        weights = torch.tensor(self.patch_ms_weights[:len(feats_per_scale)],
                            device=device, dtype=p_feats_b.dtype)
        weights = weights / weights.sum().clamp(min=1e-6)

        patch_feats = 0.0
        for w, f in zip(weights, feats_per_scale):
            patch_feats = patch_feats + w * f

        patch_centers = centers_per_scale[-1]
        patch_bbox = bbox_per_scale[-1]

        if int(k_ref) in knn_dict:
            knn_idx_ref = knn_dict[int(k_ref)]
        else:
            best_k = min(k_list, key=lambda x: abs(x - int(k_ref)))
            knn_idx_ref = knn_dict[int(best_k)]

        return patch_feats, patch_centers, patch_bbox, knn_idx_ref
    

    def _use_reloop_for_sample(self, samples, b, training: bool):
        flag = self.enable_reloop_train if training else self.enable_reloop_eval
        if not flag:
            return False

        cand_mode = samples.get("candidate_mode", None)
        if cand_mode is None:
            cand_mode_b = self.default_candidate_mode
        elif isinstance(cand_mode, (list, tuple)):
            cand_mode_b = cand_mode[b]
        else:
            cand_mode_b = cand_mode[b] if torch.is_tensor(cand_mode) else cand_mode

        if isinstance(cand_mode_b, bytes):
            cand_mode_b = cand_mode_b.decode("utf-8")
        if torch.is_tensor(cand_mode_b):
            cand_mode_b = cand_mode_b.item()

        if cand_mode_b == "sem" and self.disable_reloop_for_sem:
            return False
        if cand_mode_b == "mixed" and self.disable_reloop_for_mixed:
            return False

        return True

    def _compute_loc_patch_prior_local(
        self,
        loc_feat,            # [D_txt]
        cand_feat,           # [D_inst]
        cand_bbox,           # [6]
        cand_center,         # [3]
        patch_feats,         # [S, C]
        patch_centers,       # [S, 3]
        patch_bboxes,        # [S, 6]
        patch_knn_idx,       # [S, K_ref]
        valid_idx,           # [N_local]  当前 candidate 的 local points 在 scene 中的索引
        N_scene,             # 当前 scene 点数
    ):
        """
        输出:
            region_logits_local: [N_local]
        说明:
            不做全场景 mask，只对当前 candidate 的 local window 产出 coarse prior
        """
        cand_center_safe = cand_bbox[:3]
        cand_size_safe = cand_bbox[3:6].clamp(min=0.05)
        cand_bbox_safe = torch.cat([cand_center_safe, cand_size_safe], dim=0)
        device = patch_feats.device
        N_local = int(valid_idx.numel())

        if patch_feats.shape[0] == 0 or N_local == 0:
            return (
                patch_feats.new_full((N_local,), -20.0),
                torch.zeros((N_local,), dtype=torch.bool, device=device),
            )

        # --------------------------------------------------
        # 1) 取最近的 K_ctx 个 patch
        # --------------------------------------------------
        dist_patch = torch.norm(patch_centers - cand_center.unsqueeze(0), dim=-1)  # [S]
        K_ctx = min(self.loc_patch_k_ctx, patch_feats.shape[0])
        near_patch_idx = dist_patch.topk(k=K_ctx, largest=False).indices            # [K_ctx]

        sel_patch_feats = patch_feats[near_patch_idx]        # [K_ctx, C]
        sel_patch_centers = patch_centers[near_patch_idx]    # [K_ctx, 3]
        sel_patch_bboxes = patch_bboxes[near_patch_idx]      # [K_ctx, 6]
        sel_patch_knn = patch_knn_idx[near_patch_idx]        # [K_ctx, K_ref]

        # --------------------------------------------------
        # 2) query / patch feature
        # --------------------------------------------------
        q_in = torch.cat([loc_feat, cand_feat, cand_bbox_safe], dim=-1).unsqueeze(0)     # [1, D_txt + D_inst + 6]
        q = self.loc_patch_query_proj(q_in)                                          # [1, H]

        p = self.loc_patch_feat_proj(sel_patch_feats)                                # [K_ctx, H]

        # --------------------------------------------------
        # 3) geometry bias
        # --------------------------------------------------
        rel_center = sel_patch_centers - cand_center.unsqueeze(0)                    # [K_ctx, 3]
        rel_size = sel_patch_bboxes[:, 3:6] / cand_bbox_safe[3:6].unsqueeze(0)
        dist = torch.norm(rel_center, dim=-1, keepdim=True)                          # [K_ctx, 1]
        direction = rel_center / (dist + 1e-6)                                       # [K_ctx, 3]

        geo_in = torch.cat([rel_center, rel_size, dist, direction], dim=-1)          # [K_ctx, 10]
        geo_bias = self.loc_patch_geo_mlp(geo_in).squeeze(-1)                        # [K_ctx]

        # --------------------------------------------------
        # 4) patch logits
        # --------------------------------------------------
        pair_feat = q.expand_as(p) * p

        patch_logits_raw = self.loc_patch_score_head(pair_feat).squeeze(-1)   # [K_ctx]
        patch_logits_raw = 4.0 * torch.tanh(patch_logits_raw)                 # 限幅到大约 [-4, 4]

        geo_bias = 2.0 * torch.tanh(geo_bias)                                 # 几何偏置也限一下

        patch_logits = patch_logits_raw + geo_bias

        # --------------------------------------------------
        # 5) patch -> local point logits
        #    只映射到当前 valid_idx 对应的 local window
        # --------------------------------------------------
        point2local = torch.full((N_scene,), -1, dtype=torch.long, device=device)
        point2local[valid_idx] = torch.arange(N_local, device=device, dtype=torch.long)

        region_logits_local = patch_feats.new_full((N_local,), -20.0)
        covered_local = torch.zeros((N_local,), dtype=torch.bool, device=device)

        for j in range(K_ctx):
            pts_global = sel_patch_knn[j]
            pts_local = point2local[pts_global]
            pts_local = pts_local[pts_local >= 0]
            if pts_local.numel() == 0:
                continue

            covered_local[pts_local] = True
            region_logits_local[pts_local] = torch.maximum(
                region_logits_local[pts_local],
                patch_logits[j].expand_as(region_logits_local[pts_local])
            )

        return region_logits_local, covered_local
