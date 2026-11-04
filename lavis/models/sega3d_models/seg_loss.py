import gorilla
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from typing import Union


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Jaccard loss w.r.t the errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, pad_mask=None):
    """
    Binary Lovasz hinge loss
      logits: [B, N] Logits at each prediction (between -infinity and +infinity)
      labels: [B, N] Binary ground truth labels (0 or 1)
      pad_mask: [B, N] Binary mask, 0 for padding, 1 for non-padding
    """
    if pad_mask is not None:
        # 只保留非 padding 的部分进行计算
        # 注意：Lovasz 损失通常是在每个样本（mask）上单独计算然后平均，或者在整个 batch 上计算
        # 这里为了保持和 dice_loss 维度一致的处理逻辑，我们对每个 mask 单独计算
        
        # 为了效率，这里使用这种扁平化处理可能会对 batch 计算不太友好
        # 但为了准确处理 padding，我们遍历 batch 计算是比较稳妥的方式
        losses = []
        for i in range(logits.shape[0]):
            valid = pad_mask[i].bool()
            if not valid.any():
                continue
            
            single_logit = logits[i][valid]
            single_label = labels[i][valid]
            
            # Lovasz-Hinge calculation
            signs = 2. * single_label.float() - 1.
            errors = (1. - single_logit * signs)
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            perm = perm.data
            gt_sorted = single_label[perm]
            grad = lovasz_grad(gt_sorted)
            loss = torch.dot(F.relu(errors_sorted), grad)
            losses.append(loss)
        
        if len(losses) > 0:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=logits.device)
            
    else:
        # 如果没有 mask，直接对整个 batch 里的每个样本计算
        # 这里为了简化，假设必须有 pad_mask 传入以过滤无效点，
        # 如果确实没有 mask，建议 reshape 成 (-1) 统一计算或者沿用上面的循环
        # 下面是一个简单的 fallback
        losses = []
        for i in range(logits.shape[0]):
            single_logit = logits[i]
            single_label = labels[i]
            signs = 2. * single_label.float() - 1.
            errors = (1. - single_logit * signs)
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            perm = perm.data
            gt_sorted = single_label[perm]
            grad = lovasz_grad(gt_sorted)
            loss = torch.dot(F.relu(errors_sorted), grad)
            losses.append(loss)
        return torch.stack(losses).mean()

def get_iou(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None, pred_confidence=0.5):

    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= pred_confidence)
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score



def get_iou_prob(inputs: torch.Tensor, targets: torch.Tensor, pad_mask: Union[torch.Tensor, None]=None):
    '''
    padding modified
    '''
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    # thresholding
    binarized_inputs = (inputs >= 0.5)#.float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score

@torch.jit.script
def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


#@torch.jit.script
def dice_loss(
    inputs,
    targets,
    pad_mask
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs.sigmoid()*pad_mask
    else:
        inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)  # why+1？
    return loss.mean()

@torch.jit.script
def dice_loss_prob(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pad_mask: Union[torch.Tensor, None]=None
):
    """
    padding modified
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        pad_mask: A float tensor with the same shape as inputs. Stores the binary, 0 for padding, 1 for non-padding.
    """
    if pad_mask is not None:
        inputs = inputs*pad_mask
    else:
        inputs = inputs
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        weights = weights.unsqueeze(-1)
        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


@gorilla.LOSSES.register_module()
class Criterion(nn.Module):

    def __init__(
        self,
        loss_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fun='bce'
        
    ):
        super().__init__()
        self.loss_fun = loss_fun
        loss_weight = torch.tensor(loss_weight)
        self.register_buffer('loss_weight', loss_weight)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, layer, aux_outputs, pad_masks, gt_spmasks, isregin=False):
        loss_out = {}

        pred_scores = aux_outputs['scores'].squeeze()
        pred_masks = aux_outputs['masks'].squeeze()
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)

        # score loss
        with torch.no_grad():
            tgt_scores = get_iou(pred_masks, tgt_padding.float(), pad_masks)
        score_mask = (tgt_scores>0.5)
        if score_mask.sum() > 0:
            score_loss = torch.masked_select(F.mse_loss(pred_scores, tgt_scores, reduction='none'), score_mask).mean()
        else:
            score_loss = torch.tensor(0.0, device=pred_scores.device)

        # mask loss
        mask_bce_raw = F.binary_cross_entropy_with_logits(pred_masks, tgt_padding.float(), reduction='none')  # [B,N]
        mask_bce_loss = (mask_bce_raw*pad_masks).sum(-1) / pad_masks.sum(-1)
        mask_bce_loss = mask_bce_loss.mean()

        

        mask_dice_loss = dice_loss(pred_masks, tgt_padding.float(), pad_masks)
        # --- 新增: Lovasz Loss ---
        # Lovasz hinge 需要 logits 作为输入 (pred_masks 已经是 logits)
        mask_lovasz_loss = lovasz_hinge(pred_masks, tgt_padding.float(), pad_masks)
        if isregin :
            # ============================================================
            # [新增 1] Weighted BCE（偏 Recall）
            # ============================================================
            eps = 1e-6
            valid_cnt = pad_masks.float().sum(-1).clamp_min(1.0)                        # [B]
            pos_cnt = (tgt_padding.float() * pad_masks.float()).sum(-1)                # [B]
            neg_cnt = (valid_cnt - pos_cnt).clamp_min(1.0)                             # [B]
            pos_w = (neg_cnt / pos_cnt.clamp_min(1.0)).clamp(1.0, 20.0)                # [B] clamp 防爆

            w = torch.ones_like(tgt_padding.float())                                   # [B,N]
            w = w + (pos_w[:, None] - 1.0) * (tgt_padding.float() > 0.5).float()       # 只放大正类

            mask_wbce_loss = (mask_bce_raw * w * pad_masks.float()).sum(-1) / valid_cnt
            mask_wbce_loss = mask_wbce_loss.mean()

            # ============================================================
            # [新增 2] Tversky Loss（beta>alpha 更重罚 FN => 提 Recall）
            # ============================================================
            alpha = 0.3
            beta  = 0.7
            p = torch.sigmoid(pred_masks)
            g = tgt_padding.float()
            m = pad_masks.float()

            tp = (p * g * m).sum(-1)
            fp = (p * (1.0 - g) * m).sum(-1)
            fn = ((1.0 - p) * g * m).sum(-1)

            tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
            mask_tversky_loss = (1.0 - tversky).mean()


        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        loss_out['mask_lovasz_loss'] = mask_lovasz_loss # 记录到日志
        if isregin:
            loss_out['mask_wbce_loss'] = mask_wbce_loss
            loss_out['mask_tversky_loss'] = mask_tversky_loss
            


        loss = (
            self.loss_weight[0] * mask_bce_loss + 
            self.loss_weight[1] * mask_dice_loss + 
            self.loss_weight[2] * score_loss +
            self.loss_weight[4] * mask_lovasz_loss # 加入总 loss
        )
        if isregin :
            loss = (
            self.loss_weight[0] * mask_bce_loss + 
            self.loss_weight[1] * mask_dice_loss + 
            self.loss_weight[2] * score_loss +
            self.loss_weight[4] * mask_lovasz_loss+ # 加入总 loss
            self.loss_weight[5] * mask_wbce_loss +
            self.loss_weight[6] * mask_tversky_loss
        )

        loss_out = {f'layer_{layer}_' + k: v for k, v in loss_out.items()}
        return loss, loss_out

    def forward(self, pred, gt_pmasks, gt_spmasks, sp_ref_masks=None, current_epoch=None, isregin=False):
        if isregin :
            return self.loss4reg(pred, gt_pmasks, gt_spmasks, sp_ref_masks, current_epoch)
        else:
            return self.loss4seg(pred, gt_pmasks, gt_spmasks, sp_ref_masks, current_epoch)
        
    def loss4seg(self, pred, gt_pmasks, gt_spmasks, sp_ref_masks=None, current_epoch=None):
        
        loss_out = {}
        pred_scores = pred['scores'].squeeze()
        pred_masks = pred['masks'].squeeze()

        pad_masks = ~pred['batch_mask'].squeeze()
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)
        
        # score loss
        with torch.no_grad():
            tgt_scores = get_iou(pred_masks, tgt_padding.float(), pad_masks)
        score_mask = (tgt_scores>0.5)
        if score_mask.sum() > 0:
            score_loss = torch.masked_select(F.mse_loss(pred_scores, tgt_scores, reduction='none'), score_mask).mean()
        else:
            score_loss = torch.tensor(0.0, device=pred_scores.device)

        # sample loss
        if sp_ref_masks is not None:
            # [B, M]
            ref_padding = pad_sequence(sp_ref_masks, batch_first=True)
            # [B, M]
            ref_scores = pred['ref_scores']

            if self.loss_fun=='focal':
                sample_criterion = SigmoidFocalClassificationLoss()
                cls_weights = pad_masks.float()
                cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
                cls_weights /= torch.clamp(cls_normalizer, min=1.0)
                # focal loss
                sample_loss = sample_criterion(ref_scores.unsqueeze(-1), ref_padding.unsqueeze(-1).float(), weights=cls_weights)
                sample_loss = (sample_loss.squeeze(-1)*pad_masks).sum(-1) # / pad_masks.sum(-1)
                sample_loss = sample_loss.mean()
                
            elif self.loss_fun=='bce':
                # bce loss
                sample_loss = F.binary_cross_entropy_with_logits(ref_scores, ref_padding.float(), reduction='none')
                sample_loss = (sample_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
                sample_loss = sample_loss.mean()
            else:
                raise NotImplementedError
            
        # mask loss
        mask_bce_loss = F.binary_cross_entropy_with_logits(pred_masks, tgt_padding.float(), reduction='none')
        mask_bce_loss = (mask_bce_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
        mask_bce_loss = mask_bce_loss.mean()

        # dice loss
        mask_dice_loss = dice_loss(pred_masks, tgt_padding.float(), pad_masks)

        # --- 新增: Lovasz Loss ---
        mask_lovasz_loss = lovasz_hinge(pred_masks, tgt_padding.float(), pad_masks)

        warmup_epochs = 10  # 你可以根据总 epoch 数调整，通常设为总数的 10%-20%
        
        if current_epoch is not None and current_epoch < warmup_epochs:
            lovasz_w = 0.0  # 前期关掉
        else:
            lovasz_w = self.loss_weight[4]  # 后期使用配置文件的权重 (比如 0.5)
        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        loss_out['mask_lovasz_loss'] = mask_lovasz_loss

        if sp_ref_masks is not None:
            loss_out['sample_loss'] = sample_loss
            loss = (
                self.loss_weight[0] * mask_bce_loss + 
                self.loss_weight[1] * mask_dice_loss +
                self.loss_weight[2] * score_loss + 
                self.loss_weight[3] * sample_loss +
                lovasz_w  * mask_lovasz_loss # 增加 lovasz
            )
        else:   
            loss = (
                self.loss_weight[0] * mask_bce_loss + 
                self.loss_weight[1] * mask_dice_loss +
                self.loss_weight[2] * score_loss +
                lovasz_w  * mask_lovasz_loss # 增加 lovasz (跳过 index 3)
            )
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, pad_masks, gt_spmasks)
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss

        return loss, loss_out

    def loss4reg(self, pred, gt_pmasks, gt_spmasks, sp_ref_masks=None, current_epoch=None):
        
        loss_out = {}
        pred_scores = pred['scores'].squeeze()
        pred_masks = pred['masks'].squeeze()

        pad_masks = ~pred['batch_mask'].squeeze()
        tgt_padding = pad_sequence(gt_spmasks, batch_first=True)
        
        # score loss
        with torch.no_grad():
            tgt_scores = get_iou(pred_masks, tgt_padding.float(), pad_masks)
        score_mask = (tgt_scores>0.5)
        if score_mask.sum() > 0:
            score_loss = torch.masked_select(F.mse_loss(pred_scores, tgt_scores, reduction='none'), score_mask).mean()
        else:
            score_loss = torch.tensor(0.0, device=pred_scores.device)

        # sample loss
        if sp_ref_masks is not None:
            # [B, M]
            ref_padding = pad_sequence(sp_ref_masks, batch_first=True)
            # [B, M]
            ref_scores = pred['ref_scores']

            if self.loss_fun=='focal':
                sample_criterion = SigmoidFocalClassificationLoss()
                cls_weights = pad_masks.float()
                cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
                cls_weights /= torch.clamp(cls_normalizer, min=1.0)
                # focal loss
                sample_loss = sample_criterion(ref_scores.unsqueeze(-1), ref_padding.unsqueeze(-1).float(), weights=cls_weights)
                sample_loss = (sample_loss.squeeze(-1)*pad_masks).sum(-1) # / pad_masks.sum(-1)
                sample_loss = sample_loss.mean()
                
            elif self.loss_fun=='bce':
                # bce loss
                sample_loss = F.binary_cross_entropy_with_logits(ref_scores, ref_padding.float(), reduction='none')
                sample_loss = (sample_loss*pad_masks).sum(-1) / pad_masks.sum(-1)
                sample_loss = sample_loss.mean()
            else:
                raise NotImplementedError
            
        # mask loss
        mask_bce_raw = F.binary_cross_entropy_with_logits(pred_masks, tgt_padding.float(), reduction='none')  # [B,N]
        mask_bce_loss = (mask_bce_raw*pad_masks).sum(-1) / pad_masks.sum(-1)
        mask_bce_loss = mask_bce_loss.mean()

        # dice loss
        mask_dice_loss = dice_loss(pred_masks, tgt_padding.float(), pad_masks)

        # --- 新增: Lovasz Loss ---
        mask_lovasz_loss = lovasz_hinge(pred_masks, tgt_padding.float(), pad_masks)

        # ============================================================
        # [新增 1] Weighted BCE（偏 Recall）
        # ============================================================
        # 计算每个 batch 的正负比例 -> 动态正类权重 pos_w
        # pos_w = neg/pos（上限 clamp 防止正例极少时爆炸）
        eps = 1e-6
        valid_cnt = pad_masks.float().sum(-1).clamp_min(1.0)                      # [B]
        pos_cnt = (tgt_padding.float() * pad_masks.float()).sum(-1)              # [B]
        neg_cnt = (valid_cnt - pos_cnt).clamp_min(1.0)                           # [B]
        pos_w = (neg_cnt / pos_cnt.clamp_min(1.0)).clamp(1.0, 20.0)              # [B]

        # 对正类位置乘以 pos_w（逐 batch）
        w = torch.ones_like(tgt_padding.float())                                 # [B,N]
        w = w + (pos_w[:, None] - 1.0) * (tgt_padding.float() > 0.5).float()     # 只放大正类

        mask_wbce_loss = (mask_bce_raw * w * pad_masks.float()).sum(-1) / valid_cnt
        mask_wbce_loss = mask_wbce_loss.mean()

        # ============================================================
        # [新增 2] Tversky Loss（beta>alpha 更重罚 FN => 提 Recall）
        # ============================================================
        alpha = 0.3   # FP 权重
        beta  = 0.7   # FN 权重（更大 => 更偏 recall）

        p = torch.sigmoid(pred_masks)
        g = tgt_padding.float()
        m = pad_masks.float()

        tp = (p * g * m).sum(-1)                 # [B]
        fp = (p * (1.0 - g) * m).sum(-1)         # [B]
        fn = ((1.0 - p) * g * m).sum(-1)         # [B]

        tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        mask_tversky_loss = (1.0 - tversky).mean()


        warmup_epochs = 10  # 你可以根据总 epoch 数调整，通常设为总数的 10%-20%
        
        if current_epoch is not None and current_epoch < warmup_epochs:
            lovasz_w = 0.0  # 前期关掉
        else:
            lovasz_w = self.loss_weight[4]  # 后期使用配置文件的权重 (比如 0.5)
        loss_out['score_loss'] = score_loss
        loss_out['mask_bce_loss'] = mask_bce_loss
        loss_out['mask_dice_loss'] = mask_dice_loss
        loss_out['mask_lovasz_loss'] = mask_lovasz_loss
        loss_out['mask_wbce_loss'] = mask_wbce_loss
        loss_out['mask_tversky_loss'] = mask_tversky_loss


        if sp_ref_masks is not None:
            loss_out['sample_loss'] = sample_loss
            loss = (
                self.loss_weight[0] * mask_bce_loss + 
                self.loss_weight[1] * mask_dice_loss +
                self.loss_weight[2] * score_loss + 
                self.loss_weight[3] * sample_loss +
                lovasz_w   * mask_lovasz_loss+ # 增加 lovasz
                self.loss_weight[5] * mask_wbce_loss +
                self.loss_weight[6] * mask_tversky_loss
            )
        else:   
            loss = (
                self.loss_weight[0] * 0.2 * mask_bce_loss + 
                self.loss_weight[1] * mask_dice_loss +
                self.loss_weight[2] * score_loss +
                lovasz_w  *  mask_lovasz_loss +# 增加 lovasz (跳过 index 3)
                self.loss_weight[5] * mask_wbce_loss +
                self.loss_weight[6] * mask_tversky_loss
            )
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss_i, loss_out_i = self.get_layer_loss(i, aux_outputs, pad_masks, gt_spmasks)
                loss += loss_i
                loss_out.update(loss_out_i)

        loss_out['loss'] = loss

        return loss, loss_out


def inst_ce_loss(inst_logits, inst_atts, inst_ref_labels_list):
    # inst_logits: [B, I_max]
    # inst_atts:   [B, I_max] 1=valid,0=pad
    device = inst_logits.device
    B, I_max = inst_logits.shape

    # target_idx: [B], 每个样本正例 instance 的 index；若找不到正例则设 -100 忽略
    target_idx = torch.full((B,), -100, dtype=torch.long, device=device)

    for b in range(B):
        y = inst_ref_labels_list[b]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y, device=device)
        else:
            y = y.to(device)
        # y: [I_b] 0/1
        pos = (y > 0.5).nonzero(as_tuple=False).view(-1)
        if pos.numel() > 0:
            target_idx[b] = pos[0].long()  # 你这里 pos_cnt=1，所以直接取第一个

    logits = inst_logits.masked_fill(inst_atts == 0, -1e9)  # 屏蔽 padding
    loss = F.cross_entropy(logits, target_idx, ignore_index=-100)
    return loss

class FocalLoss(nn.Module):
    """
    [SOTA 进阶] 难样本挖掘利器。
    相比普通 CE，它能让模型专注于那些 "明明是负样本，却得分很高" 的干扰项 (Hard Negatives)。
    gamma: 聚焦参数，通常取 2.0。值越大，对难样本的关注度越高。
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha # 平衡正负样本比例
        self.reduction = reduction

    def forward(self, inputs, targets, ignore_index=-100):
        """
        inputs: [B, C] or [B, I] (Logits)
        targets: [B] (Indices)
        """
        # 1. Log Softmax
        log_pt = F.log_softmax(inputs, dim=1)
        
        # 2. Gather p_t (Target 对应的概率)
        # 处理 ignore_index
        valid_mask = (targets != ignore_index)
        targets = targets.clone()
        targets[~valid_mask] = 0 # 临时填充防止 gather 越界
        
        pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        pt = pt.exp()

        # 3. Focal Weight = (1 - pt) ^ gamma
        focal_weight = (1 - pt) ** self.gamma
        
        loss = -focal_weight * log_pt

        # 4. Mask invalid
        if valid_mask.any():
            loss = loss[valid_mask]
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
        else:
            return torch.tensor(0.0, device=inputs.device)

class InfoNCELoss(nn.Module):
    """
    [SOTA 标配] 用于拉近 (Text, Target Instance) 的距离，推远 (Text, Background) 的距离。
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, inst_feats, text_feats, target_indices):
        """
        inst_feats:   [B, I, D]  (Instance Features from Transformer Output)
        text_feats:   [B, 1, D]  (Text Features)
        target_indices: [B]      (GT Index for each batch)
        """
        B, I, D = inst_feats.shape
        
        # 1. 归一化特征 (Cosine Similarity 前置步骤)
        inst_norm = F.normalize(inst_feats, p=2, dim=-1) # [B, I, D]
        text_norm = F.normalize(text_feats, p=2, dim=-1) # [B, 1, D]
        
        # 2. 计算相似度矩阵
        # [B, I, D] * [B, D, 1] -> [B, I, 1] -> [B, I]
        # 这里计算的是每个 batch 内部，文本和所有实例的相似度
        logits = torch.bmm(inst_norm, text_norm.transpose(1, 2)).squeeze(-1)
        logits = logits / self.temperature
        
        # 3. 计算 Cross Entropy
        # 目标是让 target_indices 对应的 logit 最大
        # ignore_index=-100 处理没有 GT 的情况
        loss = F.cross_entropy(logits, target_indices, ignore_index=-100)
        
        return loss