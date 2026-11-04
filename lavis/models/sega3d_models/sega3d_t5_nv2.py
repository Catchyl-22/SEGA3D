import logging
import os
import math
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
from lavis.models.sega3d_models.mask_decoder_old_v34 import MaskDecoder, GeometryAwareInstSelector
from lavis.models.sega3d_models.seg_loss import Criterion, inst_ce_loss, InfoNCELoss, FocalLoss, lovasz_hinge, dice_loss
from lavis.models.sega3d_models.controller import Iterative3DMaskDecoder
from lavis.models.sega3d_models.pos3dencoding import SinusoidalPositionalEncoding3D
from lavis.common.utils import is_url
import contextlib
from lavis.models.sega3d_models.seg_loss import get_iou
from torch_cluster import fps
from collections import OrderedDict
import torch_scatter
import time
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch_scatter import scatter_mean
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class SEGA3DT5(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {'pretrain_flant5xl': 'configs/models/blip2/blip2_pretrain_flant5xl.yaml'}

    def __init__(self, num_query_token=32, t5_model='google/flan-t5-xl', prompt='', apply_lemmatizer=False, point_encoder_cfg=None, mask_decoder_cfg=None, mask_decoder=None, seg_criterion_cfg=None, pred_confidence=0.5, eval_topk=4, save_topk_diagnostics=False, recall_topk_list=None):
        """
        apply_lemmatizer: when set to True, postprocess predict_seg() result with lemmas.
        """
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.set_float32_matmul_precision('high')
        from lavis.models.sega3d_models.point_extractor import PointExtractor
        self.encoder = PointExtractor(**point_encoder_cfg)
        self.mask_decoder = MaskDecoder(**mask_decoder_cfg)
        self.region_decoder = MaskDecoder(**mask_decoder_cfg)
        gorilla.load_checkpoint(self.encoder, point_encoder_cfg['pretrained'], strict=False, map_location='cpu')
        self.pc_adapter = nn.Linear(point_encoder_cfg['media'], 1408)
        self.num_freqs = 10
        self.bbox_pe_dim = 6 * (2 * self.num_freqs + 1)
        self.bbox_pe = SinusoidalPositionalEncoding3D(in_channels=6, num_freqs=self.num_freqs)
        self.bbox_proj = nn.Sequential(nn.Linear(self.bbox_pe_dim, 256), nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Linear(256, 1408))
        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)
        num_added_tokens = self.t5_tokenizer.add_tokens('[SEG]')
        self.t5_tokenizer.add_tokens('[LOC]')
        self.seg_token_idx = self.t5_tokenizer('[SEG]', add_special_tokens=False).input_ids[0]
        self.loc_token_idx = self.t5_tokenizer('[LOC]', add_special_tokens=False).input_ids[0]
        t5_config = T5Config.from_pretrained(t5_model)
        t5_config.dense_act_fn = 'gelu'
        self.t5_model = T5ForConditionalGeneration.from_pretrained(t5_model, config=t5_config)
        self.t5_model.resize_token_embeddings(len(self.t5_tokenizer))
        for name, param in self.t5_model.named_parameters():
            param.requires_grad = False
            param.data = param.data
        self.t5_model.get_output_embeddings().requires_grad_(True)
        self.t5_model.get_input_embeddings().requires_grad_(True)
        self.t5_proj = nn.Linear(self.Qformer.config.hidden_size, self.t5_model.config.hidden_size)
        self.prompt = ''
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        in_dim = self.t5_model.config.hidden_size
        out_dim = mask_decoder_cfg['d_text']
        text_fc = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim), nn.Dropout(0.0)]
        loc_text_fc = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim), nn.Dropout(0.0)]
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
        self.eval_topk = max(1, int(eval_topk))
        self.save_topk_diagnostics = bool(save_topk_diagnostics)
        if recall_topk_list is None:
            recall_topk_list = [1, 2, 3, 4, 5, 8, 10]
        self.recall_topk_list = sorted({max(1, int(k)) for k in recall_topk_list})
        self.global_step = 0
        self.debug = False
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.loc_text_hidden_fcs = nn.ModuleList([nn.Sequential(*loc_text_fc)])
        d_txt = out_dim
        d_inst = int(point_encoder_cfg['media'])
        transformer_dim = 256
        self.inst_selector = GeometryAwareInstSelector(d_txt=d_txt, d_inst=d_inst, d_model=transformer_dim, nhead=8, num_layer=6, max_num_inst=200)
        self.contrastive_loss = InfoNCELoss(temperature=0.07)
        self.focal_loss = FocalLoss(gamma=2.0)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)
        self.contrastive_loss = InfoNCELoss(temperature=0.07)
        self.focal_loss = FocalLoss(gamma=2.0)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)
        self.point_in_dim = int(point_encoder_cfg['media']) + 3
        self.inst_in_dim = 256
        self.hidden_dim = 64
        self.num_params = self.point_in_dim * self.hidden_dim + self.hidden_dim + (self.hidden_dim * 1 + 1)
        self.iterative_decoder = Iterative3DMaskDecoder(d_model=256, nhead=8, num_layers=3)
        raw_feat_dim = int(point_encoder_cfg['media'])
        text_feat_dim = mask_decoder_cfg['d_text']
        self.loopback_fusion = nn.Sequential(nn.Linear(raw_feat_dim + 6 + text_feat_dim, 128), nn.ReLU(), nn.Linear(128, 1))
        self.spatial_radius_sq = 2.0 ** 2
        self.text_guide_proj = nn.Linear(512, 256)
        self.bbox_head = nn.Sequential(nn.Linear(self.inst_in_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 6))
        self.fusion_norm = nn.LayerNorm(1408)
        self.pos_alpha = nn.Parameter(torch.tensor(0.1))
        self.scene_token_budget = 3072
        self.region_grid_candidates = (12, 14, 16, 18, 20, 22, 24)
        self.use_global_scene_token = False
        self.patch_ms_ks = (16, 32, 64)
        self.patch_ms_weights = (0.2, 0.3, 0.5)
        self.patch_ref_k = 32
        self.region_prepool_mlp = nn.Sequential(nn.Linear(point_encoder_cfg['media'] + 3, point_encoder_cfg['media']), nn.LayerNorm(point_encoder_cfg['media']), nn.ReLU(inplace=True), nn.Linear(point_encoder_cfg['media'], point_encoder_cfg['media']))
        self.loc_patch_k_ctx = 8
        self.loc_patch_hidden = 256
        self.loc_region_loss_weight = 0.2
        self.loc_logit_alpha = nn.Parameter(torch.tensor(0.1))
        self.loc_prior_alpha = nn.Parameter(torch.tensor(0.2))
        raw_feat_dim = int(point_encoder_cfg['media'])
        loc_txt_dim = mask_decoder_cfg['d_text']
        inst_dim = self.inst_in_dim
        self.loc_patch_feat_proj = nn.Sequential(nn.Linear(raw_feat_dim, self.loc_patch_hidden), nn.LayerNorm(self.loc_patch_hidden), nn.ReLU(inplace=True))
        self.loc_patch_query_proj = nn.Sequential(nn.Linear(loc_txt_dim + inst_dim + 6, self.loc_patch_hidden), nn.LayerNorm(self.loc_patch_hidden), nn.ReLU(inplace=True))
        self.loc_patch_geo_mlp = nn.Sequential(nn.Linear(10, 64), nn.LayerNorm(64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.loc_patch_score_head = nn.Linear(self.loc_patch_hidden, 1, bias=False)
        self.loc_prior_proj = nn.Sequential(nn.Linear(1, raw_feat_dim), nn.LayerNorm(raw_feat_dim), nn.ReLU(inplace=True))
        self.loc_region_radius_scale = 1.75

    def forward(self, samples):
        try:
            with self.maybe_autocast():
                answer = samples['answer']
                text_input = samples['text_input']
                n_answers = samples['n_answers']
                scan_ids = samples['scan_ids']
                device = samples['feats'].device if torch.is_tensor(samples.get('feats', None)) else torch.device('cuda')
                xyz = samples['coord_float']
                sp_feats_raw, p_feats_raw = self.encoder(samples)
                point_offsets = samples['point_offsets'].to(p_feats_raw.device).long()
                B = int(point_offsets.numel()) - 1
                C = int(p_feats_raw.shape[-1])
                K = getattr(self, 'inst_token_k', 32)
                inst_tokens_fine, inst_atts_fine, inst_tokens_pool, inst_atts, inst_centers, I_max = self._extract_instance_tokens(p_feats_raw, xyz, point_offsets, samples['inst2p_offsets_full'], samples['inst2p_indices_full'])
                samples.update({'inst_tokens_fine': inst_tokens_fine, 'inst_atts_fine': inst_atts_fine, 'inst_tokens': inst_tokens_pool, 'inst_atts': inst_atts, 'inst_centers': inst_centers})
                B = inst_tokens_fine.shape[0]
                padded_bbox = None
                inst_bbox_list = samples.get('inst_bbox_full', None)
                if inst_bbox_list is not None:
                    dummy_embeds = torch.zeros((B, I_max * K, 1408), device=device)
                    _, padded_bbox = self._fuse_bbox_and_vision(dummy_embeds, inst_bbox_list, point_offsets, xyz, B, I_max, K, device, is_training=True)
                pc_embeds, image_atts = self._extract_pyramid_features(p_feats_raw, xyz, point_offsets, inst_tokens_pool, inst_atts, inst_bbox_list, B, device, is_training=True)
            query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(query_embeds=query_tokens, encoder_hidden_states=pc_embeds, encoder_attention_mask=image_atts, return_dict=True)
            inputs_t5 = self.t5_proj(query_output.last_hidden_state)
            atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)
            if self.prompt:
                text_input = [self.prompt.format(question) for question in text_input]
            else:
                text_input = text_input
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                input_tokens = self.t5_tokenizer(text_input, padding='longest', truncation=True, max_length=200, return_tensors='pt').to(pc_embeds.device)
                output_tokens = self.t5_tokenizer(answer, padding='longest', truncation=True, max_length=100, return_tensors='pt').to(pc_embeds.device)
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
                targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)
                inputs_embeds = self.t5_model.encoder.embed_tokens(batch_input_tokens_input_ids)
                inputs_embeds = torch.cat([batch_inputs_t5, inputs_embeds], dim=1)
                outputs = self.t5_model(inputs_embeds=inputs_embeds, attention_mask=encoder_atts, decoder_attention_mask=output_tokens.attention_mask, return_dict=True, labels=targets, output_hidden_states=True)
                vis_len = batch_inputs_t5.shape[1]
                dense_text_raw = outputs.encoder_last_hidden_state[:, vis_len:, :]
                dense_text_proj = self.loc_text_hidden_fcs[0](dense_text_raw)
                text_mask = batch_input_tokens_atts
                seq_out = outputs['decoder_hidden_states'][-1]
                seg_token_index = targets == self.seg_token_idx
                loc_token_index = targets == self.loc_token_idx
                seg_out = seq_out[seg_token_index]
                loc_out = seq_out[loc_token_index]
                seg_features = self.text_hidden_fcs[0](seg_out).unsqueeze(1)
                loc_features = self.loc_text_hidden_fcs[0](loc_out).unsqueeze(1)
                samples['text_features'] = seg_features
                samples['loc_text_features'] = loc_features
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
                    p0, p1 = (int(point_offsets[b].item()), int(point_offsets[b + 1].item()))
                    if p1 <= p0:
                        scene_patch_feats_bank.append(p_feats_raw.new_zeros((0, C)))
                        scene_patch_centers_bank.append(xyz.new_zeros((0, 3)))
                        scene_patch_bboxes_bank.append(xyz.new_zeros((0, 6)))
                        scene_patch_knn_bank.append(torch.zeros((0, 0), dtype=torch.long, device=device))
                        continue
                    xyz_b = xyz[p0:p1]
                    p_feats_b = p_feats_raw[p0:p1]
                    patch_feats, patch_centers, patch_bbox_raw, knn_idx = self._build_patch_tokens(p_feats_b=p_feats_b, xyz_b=xyz_b, num_seed=N_env, k_list=self.patch_ms_ks, k_ref=K_env, random_start=self.training)
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
                    scene_size = (scene_max - scene_min).clamp(min=0.001)
                    env_bboxes[b, :n_act, :3] = (patch_bbox_raw[:, :3] - scene_min) / scene_size
                    env_bboxes[b, :n_act, 3:] = patch_bbox_raw[:, 3:] / scene_size
                    if self.training:
                        env_bboxes[b, :n_act, :] += (torch.rand_like(env_bboxes[b, :n_act, :]) - 0.5) * 0.05
                cmb_tokens_fine = torch.cat([samples['inst_tokens_fine'], env_tokens_fine], dim=1)
                cmb_atts_fine = torch.cat([samples['inst_atts_fine'], env_atts_fine], dim=1)
                cmb_atts = torch.cat([samples['inst_atts'], env_atts], dim=1)
                cmb_centers = torch.cat([samples['inst_centers'], env_centers], dim=1)
                cmb_bboxes = torch.cat([inst_bboxes_norm, env_bboxes], dim=1) if inst_bboxes_norm is not None else None
                inst_logits, aux_outputs = self.inst_selector(seg_features=seg_features, loc_features=loc_features, dense_text_feats=dense_text_proj, text_mask=text_mask, inst_tokens=cmb_tokens_fine, inst_atts=cmb_atts, inst_centers=cmb_centers, inst_atts_fine=cmb_atts_fine, use_token_pooling=True, inst_bboxes=cmb_bboxes, train=True)
                I_max_true = samples['inst_tokens_fine'].shape[1]
                inst_logits = inst_logits[:, :I_max_true]
                if 'base_logits' in aux_outputs:
                    aux_outputs['base_logits'] = aux_outputs['base_logits'][:, :I_max_true]
                if 'inst_feats' in aux_outputs:
                    aux_outputs['inst_feats'] = aux_outputs['inst_feats'][:, :I_max_true]
                if 'corr_logits' in aux_outputs:
                    aux_outputs['corr_logits'] = aux_outputs['corr_logits'][:, :I_max_true]
                base_logits = aux_outputs.get('base_logits', inst_logits)
                inst_mask = samples['inst_atts'] == 0
                inst_logits_masked = inst_logits.masked_fill(inst_mask, -1000000000.0)
                base_logits_masked = base_logits.masked_fill(inst_mask, -1000000000.0)
                samples['inst_logits'] = inst_logits_masked
                inst_feats_final = aux_outputs.get('inst_feats', None)
                txt_feats_final = aux_outputs.get('txt_feats', None)
                B = int(samples['inst_tokens'].shape[0])
                if inst_logits.dim() == 3 and inst_logits.size(-1) == 1:
                    inst_logits = inst_logits.squeeze(-1)
                if inst_logits.shape[0] != B:
                    n_answers = samples['n_answers']
                    if torch.is_tensor(n_answers):
                        n_answers = n_answers.tolist()
                    logits_list = []
                    s = 0
                    for b, n in enumerate(n_answers):
                        n = int(n)
                        if n <= 0:
                            logits_list.append(inst_logits.new_full((inst_logits.shape[1],), -1000000000.0))
                            continue
                        chunk = inst_logits[s:s + n]
                        logits_list.append(chunk.max(dim=0).values)
                        s += n
                    inst_logits = torch.stack(logits_list, dim=0)
                inst_ref_labels_list = samples['inst_ref_labels']
                B = inst_logits.shape[0]
                device = inst_logits.device
                target_indices = torch.full((B,), -100, dtype=torch.long, device=device)
                for b, ref_lbl in enumerate(inst_ref_labels_list):
                    if len(ref_lbl) > 0:
                        pos = (ref_lbl > 0.5).nonzero(as_tuple=True)[0]
                        if len(pos) > 0:
                            target_indices[b] = pos[0].item()
                loss_sem = F.cross_entropy(base_logits_masked, target_indices, ignore_index=-100)
                target_quality = samples['target_quality'].to(inst_logits_masked.device)
                loss_weights = target_quality
                raw_loss = F.cross_entropy(inst_logits_masked, target_indices, reduction='none', ignore_index=-100)
                valid_mask = (target_indices != -100).float()
                loss_sum = (raw_loss * loss_weights * valid_mask).sum()
                weight_sum = (loss_weights * valid_mask).sum() + 1e-06
                loss_final = loss_sum / weight_sum
                loss_cl = torch.tensor(0.0, device=device)
                inst_feats_final = aux_outputs.get('inst_feats', None)
                txt_feats_final = aux_outputs.get('txt_feats', None)
                has_valid_gt = (target_indices != -100).any()
                if inst_feats_final is not None and txt_feats_final is not None and has_valid_gt:
                    import math
                    hard_weight = 10.0
                    cl_loss_sum = 0.0
                    valid_cl_batches = 0
                    inst_class_ids_list = samples.get('inst_class_ids', None)
                    tau = 0.07
                    for b in range(B):
                        tgt_idx = target_indices[b].item()
                        if tgt_idx == -100:
                            continue
                        I_b = len(samples['inst2p_offsets_full'][b]) - 1
                        if tgt_idx >= I_b:
                            continue
                        i_f = F.normalize(inst_feats_final[b, :I_b], p=2, dim=-1)
                        t_f = F.normalize(txt_feats_final[b, 0], p=2, dim=-1)
                        logits = torch.matmul(i_f, t_f) / tau
                        if inst_class_ids_list is not None:
                            cls_ids = inst_class_ids_list[b].to(device)
                            tgt_class = cls_ids[tgt_idx]
                            hard_neg_mask = (cls_ids == tgt_class) & (torch.arange(I_b, device=device) != tgt_idx) & (cls_ids != -1)
                            if hard_neg_mask.any():
                                logits[hard_neg_mask] += math.log(hard_weight)
                        loss_b = F.cross_entropy(logits.unsqueeze(0), torch.tensor([tgt_idx], device=device))
                        cl_loss_sum += loss_b
                        valid_cl_batches += 1
                    if valid_cl_batches > 0:
                        loss_cl = cl_loss_sum / valid_cl_batches
                sim_text = F.cosine_similarity(seg_features.detach(), loc_features, dim=-1).mean()
                loss_orth = torch.abs(sim_text)
                self.global_step += 1
                loss_refine = torch.tensor(0.0, device=device)
                loss_loopback = torch.tensor(0.0, device=device)
                loss_loc_region = torch.tensor(0.0, device=device)
                if inst_feats_final is not None and self.training:
                    loss_refine_list = []
                    loopback_loss_list = []
                    train_iou_list = []
                    pred_iou_list = []
                    loc_region_loss_list = []
                    B_sz, I_sz, D_sz = inst_feats_final.shape
                    point_offsets = samples['point_offsets']
                    xyz = samples['coord_float']
                    all_centers = samples['inst_centers']
                    pred_bboxes_all = self.bbox_head(inst_feats_final)
                    for b in range(B_sz):
                        t_feat = loc_features[b, 0]
                        if hasattr(self, 'text_guide_proj'):
                            t_feat_proj = self.text_guide_proj(t_feat)
                        else:
                            t_feat_proj = t_feat
                        probs_b = inst_logits_masked[b]
                        valid_count = (samples['inst_atts'][b] == 1).sum().item()
                        K_sel = min(3, valid_count)
                        if K_sel <= 0:
                            continue
                        _, topk_idx = probs_b.topk(K_sel)
                        selected_indices = topk_idx.tolist()
                        tgt_idx = target_indices[b].item()
                        has_valid_gt = tgt_idx != -100 and target_quality[b] > 0.25
                        if has_valid_gt and tgt_idx not in selected_indices:
                            selected_indices.append(tgt_idx)
                        K_total = len(selected_indices)
                        sel_idx_tensor = torch.tensor(selected_indices, device=device)
                        sel_inst_feats = inst_feats_final[b, sel_idx_tensor]
                        sel_centers = all_centers[b, sel_idx_tensor]
                        guided_feats = sel_inst_feats * t_feat_proj.unsqueeze(0)
                        pred_box_k = pred_bboxes_all[b, sel_idx_tensor]
                        p0, p1 = (int(point_offsets[b].item()), int(point_offsets[b + 1].item()))
                        p_feats_b = p_feats_raw[p0:p1]
                        xyz_b = xyz[p0:p1]
                        N_b = p1 - p0
                        rel_xyz_k = xyz_b.unsqueeze(0) - sel_centers.unsqueeze(1)
                        abs_dist_sq = (rel_xyz_k ** 2).sum(dim=-1)
                        spatial_mask_refine = abs_dist_sq < self.spatial_radius_sq
                        loc_radius_sq = self.spatial_radius_sq * self.loc_region_radius_scale ** 2
                        spatial_mask_loc = abs_dist_sq < loc_radius_sq
                        box_sizes = pred_box_k[:, 3:6].clamp(min=0.05)
                        rel_xyz_norm = rel_xyz_k / (box_sizes.unsqueeze(1) / 2.0)
                        dense_mask_logits = torch.full((K_total, N_b), -20.0, device=device, dtype=torch.float32)
                        gt_valid_idx = None
                        gt_relative_idx = None
                        if has_valid_gt:
                            gt_relative_idx = selected_indices.index(tgt_idx)
                            gt_valid_idx = torch.where(spatial_mask_refine[gt_relative_idx])[0]
                        for k in range(K_total):
                            refine_idx = torch.where(spatial_mask_refine[k])[0]
                            loc_idx = torch.where(spatial_mask_loc[k])[0]
                            if refine_idx.numel() == 0 or loc_idx.numel() == 0:
                                continue
                            p_feats_local = p_feats_b[refine_idx]
                            rel_xyz_local = rel_xyz_norm[k:k + 1, refine_idx, :]
                            query_local = guided_feats[k:k + 1]
                            region_logits_loc, loc_covered = self._compute_loc_patch_prior_local(loc_feat=t_feat, cand_feat=sel_inst_feats[k], cand_bbox=pred_box_k[k], cand_center=sel_centers[k], patch_feats=scene_patch_feats_bank[b], patch_centers=scene_patch_centers_bank[b], patch_bboxes=scene_patch_bboxes_bank[b], patch_knn_idx=scene_patch_knn_bank[b], valid_idx=loc_idx, N_scene=N_b)
                            scene2loc = torch.full((N_b,), -1, dtype=torch.long, device=device)
                            scene2loc[loc_idx] = torch.arange(loc_idx.numel(), device=device, dtype=torch.long)
                            refine_pos_in_loc = scene2loc[refine_idx]
                            region_logits_refine = region_logits_loc[refine_pos_in_loc]
                            region_prob_refine = torch.sigmoid(region_logits_refine)
                            if has_valid_gt and k == gt_relative_idx and ('gt_pmasks_region' in samples):
                                gt_region_b = samples['gt_pmasks_region'][b].float().to(device)
                                if gt_region_b.shape[0] == N_b and loc_idx.numel() > 0:
                                    gt_region_valid = gt_region_b[loc_idx]
                                    if loc_covered.any():
                                        pred_cov = region_logits_loc[loc_covered]
                                        gt_cov = gt_region_valid[loc_covered]
                                        pred_region_uns = pred_cov.unsqueeze(0)
                                        gt_region_uns = gt_cov.unsqueeze(0)
                                        l_reg_bce = F.binary_cross_entropy_with_logits(pred_cov, gt_cov)
                                        l_reg_dice = dice_loss(pred_region_uns, gt_region_uns, pad_mask=None)
                                        l_reg_lovasz = lovasz_hinge(pred_region_uns, gt_region_uns, pad_mask=None)
                                        loc_region_loss_list.append(l_reg_bce + 3.0 * l_reg_dice + 3.0 * l_reg_lovasz)
                            loc_prior_feat = self.loc_prior_proj(region_prob_refine.unsqueeze(-1))
                            p_feats_local_enh = p_feats_local + self.loc_prior_alpha * loc_prior_feat
                            local_logits = self.iterative_decoder(query=query_local, p_feats=p_feats_local_enh, rel_xyz=rel_xyz_local)
                            local_logits = local_logits + self.loc_logit_alpha * region_logits_refine.unsqueeze(0)
                            dense_mask_logits[k, refine_idx] = local_logits.squeeze(0).to(torch.float32)
                        dense_mask_probs = torch.sigmoid(dense_mask_logits)
                        mask_sum = dense_mask_probs.sum(dim=-1, keepdim=True) + 1e-06
                        repooled_feats = torch.matmul(dense_mask_probs, p_feats_b) / mask_sum
                        t_feat_expanded = t_feat.unsqueeze(0).expand(K_total, -1)
                        loopback_input = torch.cat([repooled_feats, pred_box_k, t_feat_expanded], dim=-1)
                        loopback_logits = self.loopback_fusion(loopback_input).squeeze(-1)
                        if has_valid_gt:
                            gt_relative_idx = selected_indices.index(tgt_idx)
                            match_labels = torch.zeros_like(loopback_logits)
                            match_labels[gt_relative_idx] = 1.0
                            l_match = F.binary_cross_entropy_with_logits(loopback_logits, match_labels)
                            loopback_loss_list.append(l_match)
                            if 'gt_pmasks' in samples and gt_valid_idx is not None:
                                gt_mask_b = samples['gt_pmasks'][b].float().to(device)
                                if gt_mask_b.shape[0] == N_b and gt_valid_idx.numel() > 0:
                                    gt_logits_valid = dense_mask_logits[gt_relative_idx, gt_valid_idx]
                                    gt_mask_valid = gt_mask_b[gt_valid_idx]
                                    pred_logits_uns = gt_logits_valid.unsqueeze(0)
                                    gt_mask_uns = gt_mask_valid.unsqueeze(0)
                                    l_bce = F.binary_cross_entropy_with_logits(gt_logits_valid, gt_mask_valid)
                                    l_dice = dice_loss(pred_logits_uns, gt_mask_uns, pad_mask=None)
                                    l_lovasz = lovasz_hinge(pred_logits_uns, gt_mask_uns, pad_mask=None)
                                    loss_refine_list.append(l_bce + 3.0 * l_dice + 3.0 * l_lovasz)
                                    with torch.no_grad():
                                        pred_m_gt = (gt_logits_valid > 0).float()
                                        inter_gt = (pred_m_gt * gt_mask_valid).sum()
                                        uni_gt = pred_m_gt.sum() + gt_mask_valid.sum() - inter_gt
                                        train_iou_list.append((inter_gt / (uni_gt + 1e-06)).item())
                                        pred_logits_top1 = dense_mask_logits[0]
                                        pred_m_top1 = (pred_logits_top1 > 0).float()
                                        inter_pred = (pred_m_top1 * gt_mask_b).sum()
                                        uni_pred = pred_m_top1.sum() + gt_mask_b.sum() - inter_pred
                                        pred_iou_list.append((inter_pred / (uni_pred + 1e-06)).item())
                    if len(loss_refine_list) > 0:
                        loss_refine = torch.stack(loss_refine_list).mean()
                    if len(loopback_loss_list) > 0:
                        loss_loopback = torch.stack(loopback_loss_list).mean()
                    if len(loc_region_loss_list) > 0:
                        loss_loc_region = torch.stack(loc_region_loss_list).mean()
                loss_bbox = torch.tensor(0.0, device=device)
                if inst_feats_final is not None and 'target_bboxes' in samples and self.training:
                    gt_bboxes = samples['target_bboxes'].to(device)
                    bbox_loss_list = []
                    for b in range(B_sz):
                        tgt_idx = target_indices[b].item()
                        if tgt_idx == -100:
                            continue
                        I_b = inst_feats_final.shape[1]
                        if tgt_idx >= I_b:
                            continue
                        pred_box = pred_bboxes_all[b, tgt_idx]
                        gt_box = gt_bboxes[b]
                        if gt_box.abs().sum() > 1e-06:
                            l1_loss = F.l1_loss(pred_box, gt_box)
                            bbox_loss_list.append(l1_loss)
                    if len(bbox_loss_list) > 0:
                        loss_bbox = torch.stack(bbox_loss_list).mean()
                loss_spatial_aux = torch.tensor(0.0, device=device)
                corr_logits = aux_outputs.get('corr_logits', None)
                if corr_logits is not None:
                    corr_logits_masked = corr_logits.masked_fill(inst_mask, -1000000000.0)
                    raw_spatial_loss = F.cross_entropy(corr_logits_masked, target_indices, reduction='none', ignore_index=-100)
                    loss_spatial_aux = (raw_spatial_loss * loss_weights * valid_mask).sum() / weight_sum
                loss = outputs.loss + 2.0 * loss_sem + 2.0 * loss_final + 2.0 * loss_cl + 0.1 * loss_orth + 1.0 * loss_refine + 2.0 * loss_loopback + 2.0 * loss_bbox + 2.0 * loss_spatial_aux + self.loc_region_loss_weight * loss_loc_region
                if self._is_master() and self.global_step % 100 == 0:
                    with torch.no_grad():
                        print(f'\n[DEBUG Step {self.global_step}] Check Performance >>>>>>>>>>>>>>>>')
                        print('  📊 [Loss Breakdown]:')
                        print(f'    - Sem (2.0)     : {loss_sem.item():.4f}')
                        print(f'    - Final (2.0)   : {loss_final.item():.4f}')
                        print(f'    - CL (2.0)      : {loss_cl.item():.4f}')
                        print(f'    - Orth (0.1)    : {loss_orth.item():.4f}')
                        print(f'    - Refine (2.0)  : {loss_refine.item():.4f}  <-- 决定 mIoU 上限的核心')
                        print(f'    - Loopback (2.0): {loss_loopback.item():.4f}')
                        print(f'    - BBox (5.0)    : {loss_bbox.item():.4f}  <-- 盯住它是不是远大于 Refine')
                        print(f'    - LocRegion(1.0): {loss_loc_region.item():.4f}')
                        print(f'    👉 Total Loss   : {loss.item():.4f}')
                        probs = torch.softmax(inst_logits_masked.float(), dim=1)
                        pred_ids = probs.argmax(dim=1)
                        valid_mask = target_indices != -100
                        num_valid = valid_mask.sum().item()
                        if num_valid > 0:
                            correct_count = (pred_ids[valid_mask] == target_indices[valid_mask]).sum().item()
                            batch_acc = correct_count / num_valid
                            print(f'\n  🎯 Batch Acc  : {batch_acc:.2%} ({correct_count}/{num_valid})')
                        else:
                            print(f'\n  ⚠️ Batch Acc  : N/A (No valid GT in this batch)')
                        if 'train_iou_list' in locals() and len(train_iou_list) > 0:
                            mean_iou = sum(train_iou_list) / len(train_iou_list)
                            mean_pred_iou = sum(pred_iou_list) / len(pred_iou_list)
                            print(f'  😇 Oracle mIoU  : {mean_iou:.2%} (上限能力: 如果全选对能考多少分)')
                            print(f'  🏆 True mIoU    : {mean_pred_iou:.2%} (真实成绩: 包含选错框的综合得分)')
                        else:
                            print(f'  🧩 Train mIoU   : N/A (No valid masks computed in this batch)')
                        print(f'  🚪 Position Gate Alpha: {self.pos_alpha.item():.4f}')
                        unique_preds = torch.unique(pred_ids)
                        n_unique = len(unique_preds)
                        print(f'  🔍 Unique Preds: {n_unique}/{B} (Diversity Check)')
                        if n_unique == 1 and B > 1:
                            print('  🔴 [WARNING] MODEL COLLAPSED! Outputting same ID!')
                        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n')
                return {'loss': loss}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'🔥 Crash at global_step {self.global_step}, saving batch...')
            raise e

    def predict_seg(self, samples, num_beams=5, inference_method='generate', max_len=200, min_len=1, num_ans_candidates=128, answer_list=None, prompt='', length_penalty=-1, repetition_penalty=1.0, **kwargs):
        assert 'full' in ['full', 'wo_rerank', 'wo_remask']
        if not hasattr(self, '_pred_debug_count'):
            self._pred_debug_count = 0
        with torch.cuda.amp.autocast(enabled=self.device != torch.device('cpu'), dtype=torch.bfloat16):
            text_input = samples['text_input']
            sp_feats_raw, p_feats_raw = self.encoder(samples)
            K = getattr(self, 'inst_token_k', 32)
            device = samples['feats'].device if torch.is_tensor(samples.get('feats', None)) else torch.device('cuda')
            point_offsets = samples['point_offsets'].to(device).long()
            inst2p_offsets_list = samples['inst2p_offsets_full']
            inst2p_indices_list = samples['inst2p_indices_full']
            xyz = samples['coord_float']
            B = int(point_offsets.numel()) - 1
            C = int(p_feats_raw.shape[-1])
            K = getattr(self, 'inst_token_k', 32)
            inst_tokens_fine, inst_atts_fine, inst_tokens_pool, inst_atts, inst_centers, I_max = self._extract_instance_tokens(p_feats_raw, xyz, point_offsets, samples['inst2p_offsets_full'], samples['inst2p_indices_full'])
            samples.update({'inst_tokens_fine': inst_tokens_fine, 'inst_atts_fine': inst_atts_fine, 'inst_tokens': inst_tokens_pool, 'inst_atts': inst_atts, 'inst_centers': inst_centers})
            B = inst_tokens_fine.shape[0]
            pc_embeds = inst_tokens_fine.reshape(B, I_max * K, C)
            image_atts = inst_atts_fine.reshape(B, I_max * K)
            pc_embeds = self.pc_adapter(pc_embeds)
            padded_bbox = None
            inst_bbox_list = samples.get('inst_bbox_full', None)
            if inst_bbox_list is not None:
                _, padded_bbox = self._fuse_bbox_and_vision(torch.zeros((B, I_max * K, 1408), device=device), inst_bbox_list, point_offsets, xyz, B, I_max, K, device, is_training=False)
            pc_embeds, image_atts = self._extract_pyramid_features(p_feats_raw, xyz, point_offsets, inst_tokens_pool, inst_atts, inst_bbox_list, B, device, is_training=False)
        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        q_dtype = next(self.Qformer.parameters()).dtype
        pc_embeds = pc_embeds.to(q_dtype)
        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1).to(q_dtype)
        query_output = self.Qformer.bert(query_embeds=query_tokens, encoder_hidden_states=pc_embeds, encoder_attention_mask=image_atts, return_dict=True)
        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(pc_embeds.device)
        if isinstance(text_input, str):
            text_input = [text_input]
        if self.prompt:
            text_input = [prompt.format(question) for question in text_input]
        else:
            text_input = text_input
        input_tokens = self.t5_tokenizer(text_input, padding='longest', return_tensors='pt').to(pc_embeds.device)
        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)
        num_beams = 1
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
        with torch.cuda.amp.autocast(enabled=self.device != torch.device('cpu'), dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)
            encoder_outputs = self.t5_model.encoder(inputs_embeds=inputs_embeds, attention_mask=encoder_atts, output_hidden_states=True, return_dict=True)
            vis_len = inputs_t5.shape[1]
            dense_text_raw = encoder_outputs.hidden_states[-1][:, vis_len:, :]
            dense_text_proj = self.loc_text_hidden_fcs[0](dense_text_raw)
            text_mask = input_tokens.attention_mask
            probe_text = ['[SEG] [LOC]'] * B
            probe_tokens = self.t5_tokenizer(probe_text, padding='longest', truncation=True, max_length=8, add_special_tokens=False, return_tensors='pt').to(pc_embeds.device)
            probe_targets = probe_tokens.input_ids.masked_fill(probe_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100)
            probe_outputs = self.t5_model(encoder_outputs=encoder_outputs, attention_mask=encoder_atts, decoder_attention_mask=probe_tokens.attention_mask, return_dict=True, labels=probe_targets, output_hidden_states=True, use_cache=False)
            probe_seq = probe_outputs['decoder_hidden_states'][-1]
            probe_seg_index = probe_targets == self.seg_token_idx
            probe_loc_index = probe_targets == self.loc_token_idx
            seg_out = probe_seq[probe_seg_index]
            loc_out = probe_seq[probe_loc_index]
            if seg_out.shape[0] != B or loc_out.shape[0] != B:
                if self._is_master():
                    print('[PRED PROBE ERROR] seg_out.shape =', tuple(seg_out.shape), 'loc_out.shape =', tuple(loc_out.shape), 'B =', B)
                raise RuntimeError('Probe extraction for [SEG]/[LOC] failed in predict_seg')
            seg_features = self.text_hidden_fcs[0](seg_out).unsqueeze(1)
            loc_features = self.loc_text_hidden_fcs[0](loc_out).unsqueeze(1)
            samples['text_features'] = seg_features
            samples['loc_text_features'] = loc_features
            inst_bboxes_norm = padded_bbox if 'padded_bbox' in locals() else None
            N_env = 0 if False else 64
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
                p0, p1 = (int(point_offsets[b].item()), int(point_offsets[b + 1].item()))
                if p1 <= p0:
                    scene_patch_feats_bank.append(p_feats_raw.new_zeros((0, C)))
                    scene_patch_centers_bank.append(xyz.new_zeros((0, 3)))
                    scene_patch_bboxes_bank.append(xyz.new_zeros((0, 6)))
                    scene_patch_knn_bank.append(torch.zeros((0, 0), dtype=torch.long, device=device))
                    continue
                xyz_b = xyz[p0:p1]
                p_feats_b = p_feats_raw[p0:p1]
                patch_feats, patch_centers, patch_bbox_raw, knn_idx = self._build_patch_tokens(p_feats_b=p_feats_b, xyz_b=xyz_b, num_seed=N_env, k_list=self.patch_ms_ks, k_ref=K_env, random_start=False)
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
                scene_size = (scene_max - scene_min).clamp(min=0.001)
                env_bboxes[b, :n_act, :3] = (patch_bbox_raw[:, :3] - scene_min) / scene_size
                env_bboxes[b, :n_act, 3:] = patch_bbox_raw[:, 3:] / scene_size
            cmb_tokens_fine = torch.cat([samples['inst_tokens_fine'], env_tokens_fine], dim=1)
            cmb_atts_fine = torch.cat([samples['inst_atts_fine'], env_atts_fine], dim=1)
            cmb_atts = torch.cat([samples['inst_atts'], env_atts], dim=1)
            cmb_centers = torch.cat([samples['inst_centers'], env_centers], dim=1)
            cmb_bboxes = torch.cat([inst_bboxes_norm, env_bboxes], dim=1) if inst_bboxes_norm is not None else None
            inst_logits, aux_outputs = self.inst_selector(seg_features=seg_features, loc_features=loc_features, dense_text_feats=dense_text_proj, text_mask=text_mask, inst_tokens=cmb_tokens_fine, inst_atts=cmb_atts, inst_centers=cmb_centers, inst_atts_fine=cmb_atts_fine, use_token_pooling=True, inst_bboxes=cmb_bboxes, train=False)
            I_max_true = samples['inst_tokens_fine'].shape[1]
            inst_logits = inst_logits[:, :I_max_true]
            if 'base_logits' in aux_outputs:
                aux_outputs['base_logits'] = aux_outputs['base_logits'][:, :I_max_true]
            if 'inst_feats' in aux_outputs:
                aux_outputs['inst_feats'] = aux_outputs['inst_feats'][:, :I_max_true]
            if 'corr_logits' in aux_outputs:
                aux_outputs['corr_logits'] = aux_outputs['corr_logits'][:, :I_max_true]
            if inst_logits.dim() == 3 and inst_logits.size(-1) == 1:
                inst_logits = inst_logits.squeeze(-1)
            inst_logits = inst_logits.masked_fill(samples['inst_atts'] == 0, -1000000000.0)
            probs = torch.softmax(inst_logits.float(), dim=1)
            inst2p_offsets_list = samples['inst2p_offsets_full']
            inst2p_indices_list = samples['inst2p_indices_full']
            point_offsets = samples['point_offsets']
            B = inst_logits.shape[0]
            device = inst_logits.device
            pred_point_masks = []
            pred_scores = []
            selector_topk_diagnostics = []
            inst_feats_final = aux_outputs.get('inst_feats')
            if inst_feats_final is not None:
                pred_bboxes_all = self.bbox_head(inst_feats_final)
            for b in range(B):
                I_b = len(inst2p_offsets_list[b]) - 1
                p0, p1 = (int(point_offsets[b].item()), int(point_offsets[b + 1].item()))
                n_points = p1 - p0
                mask_b = torch.full((n_points,), -20.0, device=device, dtype=torch.float32)
                best_score = 0.0
                if self.save_topk_diagnostics:
                    selector_topk_diagnostics.append(self._compute_selector_topk_diagnostics(probs_b=probs[b, :I_b], inst_offsets=inst2p_offsets_list[b], inst_indices=inst2p_indices_list[b], gt_pmask=samples['gt_pmasks'][b], inst_ref_labels=samples.get('inst_ref_labels', [None] * B)[b], n_points=n_points))
                else:
                    selector_topk_diagnostics.append(None)
                if I_b > 0 and inst_feats_final is not None:
                    probs_b = probs[b, :I_b]
                    K_sel = min(self.eval_topk, I_b)
                    _, topk_idx = probs_b.topk(K_sel)
                    t_feat = loc_features[b, 0]
                    t_feat_proj = self.text_guide_proj(t_feat) if hasattr(self, 'text_guide_proj') else t_feat
                    sel_inst_feats = inst_feats_final[b, topk_idx]
                    sel_centers = samples['inst_centers'][b, topk_idx]
                    pred_box_k = pred_bboxes_all[b, topk_idx]
                    guided_feats = sel_inst_feats * t_feat_proj.unsqueeze(0)
                    p_feats_b = p_feats_raw[p0:p1]
                    xyz_b = xyz[p0:p1]
                    N_b = p1 - p0
                    rel_xyz_k = xyz_b.unsqueeze(0) - sel_centers.unsqueeze(1)
                    abs_dist_sq = (rel_xyz_k ** 2).sum(dim=-1)
                    spatial_mask_refine = abs_dist_sq < self.spatial_radius_sq
                    loc_radius_sq = self.spatial_radius_sq * self.loc_region_radius_scale ** 2
                    spatial_mask_loc = abs_dist_sq < loc_radius_sq
                    box_sizes = pred_box_k[:, 3:6].clamp(min=0.05)
                    rel_xyz_norm = rel_xyz_k / (box_sizes.unsqueeze(1) / 2.0)
                    dense_mask_logits = torch.full((K_sel, N_b), -20.0, device=device, dtype=torch.float32)
                    for k in range(K_sel):
                        refine_idx = torch.where(spatial_mask_refine[k])[0]
                        loc_idx = torch.where(spatial_mask_loc[k])[0]
                        if refine_idx.numel() == 0 or loc_idx.numel() == 0:
                            continue
                        p_feats_local = p_feats_b[refine_idx]
                        rel_xyz_local = rel_xyz_norm[k:k + 1, refine_idx, :]
                        query_local = guided_feats[k:k + 1]
                        region_logits_loc, loc_covered = self._compute_loc_patch_prior_local(loc_feat=t_feat, cand_feat=sel_inst_feats[k], cand_bbox=pred_box_k[k], cand_center=sel_centers[k], patch_feats=scene_patch_feats_bank[b], patch_centers=scene_patch_centers_bank[b], patch_bboxes=scene_patch_bboxes_bank[b], patch_knn_idx=scene_patch_knn_bank[b], valid_idx=loc_idx, N_scene=N_b)
                        scene2loc = torch.full((N_b,), -1, dtype=torch.long, device=device)
                        scene2loc[loc_idx] = torch.arange(loc_idx.numel(), device=device, dtype=torch.long)
                        refine_pos_in_loc = scene2loc[refine_idx]
                        region_logits_refine = region_logits_loc[refine_pos_in_loc]
                        region_prob_refine = torch.sigmoid(region_logits_refine)
                        local_logits_plain = self.iterative_decoder(query=query_local, p_feats=p_feats_local, rel_xyz=rel_xyz_local)
                        loc_prior_feat = self.loc_prior_proj(region_prob_refine.unsqueeze(-1))
                        p_feats_local_enh = p_feats_local + self.loc_prior_alpha * loc_prior_feat
                        local_logits_feat = self.iterative_decoder(query=query_local, p_feats=p_feats_local_enh, rel_xyz=rel_xyz_local)
                        local_logits = local_logits_feat + self.loc_logit_alpha * region_logits_refine.unsqueeze(0)
                        dense_mask_logits[k, refine_idx] = local_logits.squeeze(0).to(torch.float32)
                    p_feats_b = p_feats_raw[p0:p1]
                    dense_mask_probs = torch.sigmoid(dense_mask_logits)
                    mask_sum = dense_mask_probs.sum(dim=-1, keepdim=True) + 1e-06
                    repooled_feats = torch.matmul(dense_mask_probs, p_feats_b) / mask_sum
                    t_feat_expanded = t_feat.unsqueeze(0).expand(K_sel, -1)
                    loopback_input = torch.cat([repooled_feats, pred_box_k, t_feat_expanded], dim=-1)
                    loopback_logits = self.loopback_fusion(loopback_input).squeeze(-1)
                    best_local_idx = loopback_logits.argmax().item()
                    best_score = torch.sigmoid(loopback_logits[best_local_idx]).item()
                    mask_b = dense_mask_logits[best_local_idx].float()
                pred_scores.append(best_score)
                pred_point_masks.append(mask_b)

            def _batch_field_to_list(samples, key, B, default):
                """
                从 samples 里安全取 batch 字段。
                如果字段不存在，返回默认值列表，并标记 ok=False。
                """
                if key not in samples:
                    return ([default for _ in range(B)], False)
                v = samples[key]
                if torch.is_tensor(v):
                    v = v.detach().cpu().view(-1).tolist()
                elif isinstance(v, (list, tuple)):
                    v = list(v)
                else:
                    v = [v]
                if len(v) < B:
                    v = v + [default for _ in range(B - len(v))]
                elif len(v) > B:
                    v = v[:B]
                return (v, True)
            is_unique_sem, ok_unique = _batch_field_to_list(samples, 'is_unique_sem', B, 0)
            is_multiple_sem, ok_multiple = _batch_field_to_list(samples, 'is_multiple_sem', B, 0)
            target_sem_inst_counts, ok_count = _batch_field_to_list(samples, 'target_sem_inst_counts', B, -1)
            target_sem_ids, ok_semid = _batch_field_to_list(samples, 'target_sem_ids', B, -1)
            target_valid_for_sem_count, ok_valid = _batch_field_to_list(samples, 'target_valid_for_sem_count', B, 0)
            has_sem_split = bool(ok_unique and ok_multiple and ok_count and ok_semid and ok_valid)
            sem_split = []
            for b in range(B):
                if not has_sem_split or int(target_valid_for_sem_count[b]) != 1:
                    sem_split.append('unknown')
                elif int(is_unique_sem[b]) == 1:
                    sem_split.append('unique')
                elif int(is_multiple_sem[b]) == 1:
                    sem_split.append('multiple')
                else:
                    sem_split.append('unknown')
            cyl = True
            if cyl:
                result = {'masks': pred_point_masks, 'scores': torch.tensor(pred_scores, device=device), 'is_point_level': True, 'eval_topk': self.eval_topk, 'selector_topk_diagnostics': selector_topk_diagnostics, 'has_sem_split': has_sem_split, 'sem_split': sem_split, 'is_unique_sem': [int(x) for x in is_unique_sem], 'is_multiple_sem': [int(x) for x in is_multiple_sem], 'target_sem_inst_counts': [int(x) for x in target_sem_inst_counts], 'target_sem_ids': [int(x) for x in target_sem_ids], 'target_valid_for_sem_count': [int(x) for x in target_valid_for_sem_count]}
            else:
                result = {'masks': pred_point_masks, 'scores': torch.tensor(pred_scores, device=device), 'is_point_level': True, 'eval_topk': self.eval_topk, 'selector_topk_diagnostics': selector_topk_diagnostics}
            return result

    @torch.no_grad()
    def _compute_selector_topk_diagnostics(self, probs_b, inst_offsets, inst_indices, gt_pmask, inst_ref_labels, n_points):
        """Measure selector retrieval quality before remasking and reranking."""
        device = probs_b.device
        topk_values = self.recall_topk_list
        num_candidates = int(probs_b.numel())
        empty_by_k = {str(k): {'effective_k': 0, 'oracle_iou': 0.0, 'recall_iou_025': 0.0, 'recall_iou_050': 0.0, 'oracle_target_hit': 0.0} for k in topk_values}
        if num_candidates == 0 or int(n_points) <= 0:
            return {'num_candidates': num_candidates, 'requested_eval_topk': self.eval_topk, 'eval_topk': min(self.eval_topk, num_candidates), 'bank_oracle_iou': 0.0, 'bank_recall_iou_025': 0.0, 'bank_recall_iou_050': 0.0, 'by_k': empty_by_k}
        inst_offsets = torch.as_tensor(inst_offsets, dtype=torch.long, device=device)
        inst_indices = torch.as_tensor(inst_indices, dtype=torch.long, device=device)
        gt_mask = torch.as_tensor(gt_pmask, device=device).reshape(-1) > 0.5
        if gt_mask.numel() != int(n_points):
            raise RuntimeError(f'Top-K diagnostics received a GT mask with {gt_mask.numel()} points, expected {n_points}.')
        valid = (inst_indices >= 0) & (inst_indices < int(n_points))
        safe_indices = inst_indices.clamp(min=0, max=max(int(n_points) - 1, 0))
        candidate_sizes = torch_scatter.segment_csr(valid.to(torch.float32), inst_offsets, reduce='sum')
        intersections = torch_scatter.segment_csr((gt_mask[safe_indices] & valid).to(torch.float32), inst_offsets, reduce='sum')
        gt_size = gt_mask.to(torch.float32).sum()
        unions = candidate_sizes + gt_size - intersections
        candidate_ious = torch.where(unions > 0, intersections / unions.clamp_min(1e-06), torch.zeros_like(unions))[:num_candidates]
        if candidate_ious.numel() != num_candidates:
            raise RuntimeError(f'Top-K diagnostics found inconsistent candidate metadata: selector={num_candidates}, CSR={candidate_ious.numel()}.')
        sorted_indices = torch.argsort(probs_b, descending=True)
        ref_labels = None
        if inst_ref_labels is not None:
            ref_labels = torch.as_tensor(inst_ref_labels, dtype=torch.float32, device=device).reshape(-1)[:num_candidates]
        bank_oracle = float(candidate_ious.max().item()) if candidate_ious.numel() else 0.0
        by_k = {}
        for requested_k in topk_values:
            effective_k = min(int(requested_k), num_candidates)
            selected = sorted_indices[:effective_k]
            selected_ious = candidate_ious[selected]
            oracle_iou = float(selected_ious.max().item()) if selected_ious.numel() else 0.0
            oracle_target_hit = 0.0
            if ref_labels is not None and ref_labels.numel() == num_candidates:
                oracle_target_hit = float((ref_labels[selected] > 0.5).any().item())
            by_k[str(requested_k)] = {'effective_k': effective_k, 'oracle_iou': oracle_iou, 'recall_iou_025': float(oracle_iou > 0.25), 'recall_iou_050': float(oracle_iou > 0.5), 'oracle_target_hit': oracle_target_hit}
        return {'num_candidates': num_candidates, 'requested_eval_topk': self.eval_topk, 'eval_topk': min(self.eval_topk, num_candidates), 'bank_oracle_iou': bank_oracle, 'bank_recall_iou_025': float(bank_oracle > 0.25), 'bank_recall_iou_050': float(bank_oracle > 0.5), 'by_k': by_k}

    def _extract_instance_tokens(self, p_feats_raw, xyz, point_offsets, inst2p_offsets_list, inst2p_indices_list):
        B = int(point_offsets.numel()) - 1
        C = int(p_feats_raw.shape[-1])
        K = getattr(self, 'inst_token_k', 32)
        device = p_feats_raw.device
        inst_tokens_fine_list, inst_atts_fine_list = ([], [])
        inst_tokens_pool_list, inst_centers_list = ([], [])
        inst_lens = []
        for b in range(B):
            p0, p1 = (int(point_offsets[b].item()), int(point_offsets[b + 1].item()))
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
            indices, inst_ids = (indices[:M], inst_ids[:M])
            pts_feats = p_feats_b.index_select(0, indices)
            tok_pool = scatter_mean(pts_feats, inst_ids, dim=0, dim_size=I_b)
            current_xyz = xyz_b.index_select(0, indices)
            center_pool = scatter_mean(current_xyz, inst_ids, dim=0, dim_size=I_b)
            for inst_id in range(I_b):
                s, e = (int(offsets[inst_id].item()), int(offsets[inst_id + 1].item()))
                if e <= s:
                    continue
                pts = indices[s:e]
                pts = pts[(pts >= 0) & (pts < N_b)]
                if pts.numel() == 0:
                    continue
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
        I_max = max(inst_lens) if len(inst_lens) > 0 else 0
        inst_tokens_fine = p_feats_raw.new_zeros((B, I_max, K, C))
        inst_atts_fine = torch.zeros((B, I_max, K), dtype=torch.long, device=device)
        inst_tokens_pool = p_feats_raw.new_zeros((B, I_max, C))
        inst_atts = torch.zeros((B, I_max), dtype=torch.long, device=device)
        inst_centers = p_feats_raw.new_zeros((B, I_max, 3))
        for b in range(B):
            I_b = inst_lens[b]
            if I_b <= 0:
                continue
            inst_tokens_fine[b, :I_b] = inst_tokens_fine_list[b]
            inst_atts_fine[b, :I_b] = inst_atts_fine_list[b]
            inst_tokens_pool[b, :I_b] = inst_tokens_pool_list[b]
            inst_atts[b, :I_b] = 1
            inst_centers[b, :I_b] = inst_centers_list[b]
        return (inst_tokens_fine, inst_atts_fine, inst_tokens_pool, inst_atts, inst_centers, I_max)

    def _fuse_bbox_and_vision(self, pc_embeds, inst_bbox_list, point_offsets, xyz, B, I_max, K, device, is_training):
        padded_bbox = pc_embeds.new_zeros((B, I_max, 6))
        for b in range(B):
            I_b = min(I_max, inst_bbox_list[b].shape[0]) if inst_bbox_list[b] is not None else 0
            if I_b > 0:
                p0, p1 = (int(point_offsets[b].item()), int(point_offsets[b + 1].item()))
                xyz_b = xyz[p0:p1]
                scene_min, scene_max = (xyz_b.min(dim=0)[0], xyz_b.max(dim=0)[0])
                scene_size = (scene_max - scene_min).clamp(min=0.001)
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
        return (self.fusion_norm(pc_embeds + self.pos_alpha * bbox_features), padded_bbox)

    def _extract_pyramid_features(self, p_feats_raw, xyz, point_offsets, inst_tokens_pool, inst_atts, inst_bbox_list, B, device, is_training):
        """
        QFormer 主输入:
        region tokens + instance tokens
        """
        pyramid_feats_list = []
        pyramid_bbox_list = []
        pyramid_len_list = []
        target_tokens = self.scene_token_budget
        for b in range(B):
            p0, p1 = (int(point_offsets[b].item()), int(point_offsets[b + 1].item()))
            feat_b = p_feats_raw[p0:p1]
            xyz_b = xyz[p0:p1]
            scene_min = xyz_b.min(dim=0)[0]
            scene_max = xyz_b.max(dim=0)[0]
            scene_size = (scene_max - scene_min).clamp(min=0.001)
            region_feats, region_centers, region_bbox_raw = self._build_region_tokens_voxel(p_feats_b=feat_b, xyz_b=xyz_b, target_tokens=target_tokens, grid_candidates=self.region_grid_candidates)
            region_bbox = torch.zeros((region_bbox_raw.shape[0], 6), device=device, dtype=xyz_b.dtype)
            region_bbox[:, :3] = (region_bbox_raw[:, :3] - scene_min) / scene_size
            region_bbox[:, 3:] = region_bbox_raw[:, 3:] / scene_size
            I_b = int(inst_atts[b].sum().item())
            if I_b > 0 and inst_bbox_list is not None and (inst_bbox_list[b] is not None):
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
            cat_feats = torch.cat([region_feats, inst_feats], dim=0)
            cat_bbox = torch.cat([region_bbox, inst_bbox], dim=0)
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
        return (pc_embeds, image_atts)

    def _lemmatize(self, answers):

        def apply(answer):
            doc = self.lemmatizer(answer)
            words = []
            for token in doc:
                if token.pos_ in ['NOUN', 'VERB']:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = ' '.join(words)
            return answer
        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy
                self._lemmatizer = spacy.load('en_core_web_sm')
            except ImportError:
                logging.error('\n                    Please install spacy and en_core_web_sm model to apply lemmatization.\n                    python -m spacy download en_core_web_sm\n                    OR\n                    import spacy.cli\n                    spacy.cli.download("en_core_web_sm")\n                    ')
                exit(1)
        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        point_encoder_cfg = cfg.get('point_encoder_cfg')
        mask_decoder_cfg = cfg.get('mask_decoder_cfg')
        mask_decoder = cfg.get('mask_decoder')
        seg_criterion_cfg = cfg.get('seg_criterion_cfg')
        pred_confidence = cfg.get('pred_confidence')
        num_query_token = cfg.get('num_query_token')
        t5_model = cfg.get('t5_model')
        prompt = cfg.get('prompt', '')
        apply_lemmatizer = cfg.get('apply_lemmatizer', False)
        eval_topk = cfg.get('eval_topk', 4)
        save_topk_diagnostics = cfg.get('save_topk_diagnostics', False)
        recall_topk_list = cfg.get('recall_topk_list', [1, 2, 3, 4, 5, 8, 10])
        model = cls(num_query_token=num_query_token, t5_model=t5_model, prompt=prompt, apply_lemmatizer=apply_lemmatizer, point_encoder_cfg=point_encoder_cfg, mask_decoder_cfg=mask_decoder_cfg, mask_decoder=mask_decoder, seg_criterion_cfg=seg_criterion_cfg, pred_confidence=pred_confidence, eval_topk=eval_topk, save_topk_diagnostics=save_topk_diagnostics, recall_topk_list=recall_topk_list)
        model.load_checkpoint_from_config(cfg)
        return model

    def maybe_autocast(self, dtype=torch.bfloat16):
        enable_autocast = self.device != torch.device('cpu')
        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, cross_attention_freq=2):
        loacl_dir_of_model = 'bert-base-uncased'
        encoder_config = BertConfig.from_pretrained(loacl_dir_of_model)
        encoder_config.encoder_width = vision_width
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(loacl_dir_of_model, config=encoder_config)
        query_tokens = nn.Parameter(torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return (Qformer, query_tokens)

    def load_from_pretrained(self, url_or_filename):
        print('___________________________________________________')
        print(url_or_filename)
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
            checkpoint = torch.load(cached_file, map_location='cpu')
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location='cpu')
        else:
            raise RuntimeError('checkpoint url or path is invalid')
        state_dict = checkpoint['model']
        model_state_dict = self.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and model_state_dict[k].shape != v.shape:
                print(f'⚠️ [跳过加载] 形状不匹配 {k}: Checkpoint={v.shape}, Model={model_state_dict[k].shape}')
                continue
            filtered_state_dict[k] = v
        msg = self.load_state_dict(filtered_state_dict, strict=False)
        logging.info('load checkpoint from %s' % url_or_filename)
        return msg

    def _is_master(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    def _summarize_tensor(self, x, name, max_items=3):
        try:
            if isinstance(x, list):
                n = len(x)
                if n == 0:
                    return f'{name}: list(len=0)'
                shp0 = tuple(x[0].shape)
                dtype0 = x[0].dtype
                dev0 = x[0].device
                return f'{name}: list(len={n}, first={shp0}, dtype={dtype0}, device={dev0})'
            if torch.is_tensor(x):
                return f'{name}: shape={tuple(x.shape)}, dtype={x.dtype}, device={x.device}'
            return f'{name}: <{type(x)}>'
        except Exception as e:
            return f'{name}: <summary failed: {e}>'

    @torch.no_grad()
    def _stats_tensor(self, x, name, take_sample=False):
        try:
            if not torch.is_tensor(x):
                return f'{name}: <not tensor>'
            if x.numel() == 0:
                return f'{name}: numel=0'
            xs = x.float()
            if take_sample and xs.numel() > 2000000:
                idx = torch.randperm(xs.numel(), device=xs.device)[:1000000]
                xs = xs.view(-1)[idx]
            mn = xs.min().item()
            mx = xs.max().item()
            mean = xs.mean().item()
            std = xs.std().item()
            return f'{name}: min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g}'
        except Exception as e:
            return f'{name}: <stats failed: {e}>'

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
            size = (mx - mn).clamp(min=0.001)
            return torch.cat([center, size], dim=1)
        out = raw_bbox.clone()
        out[:, 3:] = out[:, 3:].clamp(min=0.001)
        return out

    def _build_region_tokens_voxel(self, p_feats_b, xyz_b, target_tokens=2048, grid_candidates=None):
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
            return (p_feats_b.new_zeros((0, C)), xyz_b.new_zeros((0, 3)), xyz_b.new_zeros((0, 6)))
        if grid_candidates is None:
            grid_candidates = self.region_grid_candidates
        scene_min = xyz_b.min(dim=0)[0]
        scene_max = xyz_b.max(dim=0)[0]
        scene_size = (scene_max - scene_min).clamp(min=1e-06)
        xyz_n = (xyz_b - scene_min) / scene_size
        best_diff = 10 ** 18
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
        region_feats = scatter_mean(p_feats_b, inv, dim=0, dim_size=num_regions)
        region_centers = scatter_mean(xyz_b.float(), inv, dim=0, dim_size=num_regions)
        region_min = torch_scatter.scatter_min(xyz_b, inv, dim=0, dim_size=num_regions)[0]
        region_max = torch_scatter.scatter_max(xyz_b, inv, dim=0, dim_size=num_regions)[0]
        region_bbox = torch.zeros((num_regions, 6), device=xyz_b.device, dtype=xyz_b.dtype)
        region_bbox[:, :3] = 0.5 * (region_min + region_max)
        region_bbox[:, 3:] = (region_max - region_min).clamp(min=0.001)
        if num_regions > target_tokens:
            keep_idx = self._fps_select(region_centers, target_tokens)
            region_feats = region_feats[keep_idx]
            region_centers = region_centers[keep_idx]
            region_bbox = region_bbox[keep_idx]
        return (region_feats, region_centers, region_bbox)

    @torch.no_grad()
    def _build_patch_tokens(self, p_feats_b, xyz_b, num_seed, k_list=None, k_ref=None, random_start=False):
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
        k_list = list(sorted(set((int(k) for k in k_list))))
        if k_ref is None:
            k_ref = int(self.patch_ref_k)
        if N_b <= 0 or num_seed <= 0:
            return (p_feats_b.new_zeros((0, C)), xyz_b.new_zeros((0, 3)), xyz_b.new_zeros((0, 6)), torch.zeros((0, 0), dtype=torch.long, device=device))
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
        seed_xyz = xyz_b[seed_idx]
        S = int(seed_xyz.shape[0])
        dist = torch.cdist(seed_xyz.float(), xyz_b.float())
        feats_per_scale = []
        centers_per_scale = []
        bbox_per_scale = []
        knn_dict = {}
        for k in k_list:
            K_eff = min(int(k), N_b)
            knn_idx = dist.topk(k=K_eff, dim=1, largest=False).indices
            knn_dict[int(k)] = knn_idx
            nbr_feats = p_feats_b[knn_idx]
            nbr_xyz = xyz_b[knn_idx]
            feat_mean = nbr_feats.mean(dim=1)
            xyz_mean = nbr_xyz.mean(dim=1)
            xyz_min = nbr_xyz.min(dim=1)[0]
            xyz_max = nbr_xyz.max(dim=1)[0]
            xyz_size = (xyz_max - xyz_min).clamp(min=0.001)
            feats_per_scale.append(feat_mean)
            centers_per_scale.append(xyz_mean)
            bbox_per_scale.append(torch.cat([xyz_mean, xyz_size], dim=-1))
        weights = torch.tensor(self.patch_ms_weights[:len(feats_per_scale)], device=device, dtype=p_feats_b.dtype)
        weights = weights / weights.sum().clamp(min=1e-06)
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
        return (patch_feats, patch_centers, patch_bbox, knn_idx_ref)

    def _compute_loc_patch_prior_local(self, loc_feat, cand_feat, cand_bbox, cand_center, patch_feats, patch_centers, patch_bboxes, patch_knn_idx, valid_idx, N_scene):
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
            return (patch_feats.new_full((N_local,), -20.0), torch.zeros((N_local,), dtype=torch.bool, device=device))
        dist_patch = torch.norm(patch_centers - cand_center.unsqueeze(0), dim=-1)
        K_ctx = min(self.loc_patch_k_ctx, patch_feats.shape[0])
        near_patch_idx = dist_patch.topk(k=K_ctx, largest=False).indices
        sel_patch_feats = patch_feats[near_patch_idx]
        sel_patch_centers = patch_centers[near_patch_idx]
        sel_patch_bboxes = patch_bboxes[near_patch_idx]
        sel_patch_knn = patch_knn_idx[near_patch_idx]
        q_in = torch.cat([loc_feat, cand_feat, cand_bbox_safe], dim=-1).unsqueeze(0)
        q = self.loc_patch_query_proj(q_in)
        p = self.loc_patch_feat_proj(sel_patch_feats)
        rel_center = sel_patch_centers - cand_center.unsqueeze(0)
        rel_size = sel_patch_bboxes[:, 3:6] / cand_bbox_safe[3:6].unsqueeze(0)
        dist = torch.norm(rel_center, dim=-1, keepdim=True)
        direction = rel_center / (dist + 1e-06)
        geo_in = torch.cat([rel_center, rel_size, dist, direction], dim=-1)
        geo_bias = self.loc_patch_geo_mlp(geo_in).squeeze(-1)
        pair_feat = q.expand_as(p) * p
        patch_logits_raw = self.loc_patch_score_head(pair_feat).squeeze(-1)
        patch_logits_raw = 4.0 * torch.tanh(patch_logits_raw)
        geo_bias = 2.0 * torch.tanh(geo_bias)
        patch_logits = patch_logits_raw + geo_bias
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
            region_logits_local[pts_local] = torch.maximum(region_logits_local[pts_local], patch_logits[j].expand_as(region_logits_local[pts_local]))
        return (region_logits_local, covered_local)
