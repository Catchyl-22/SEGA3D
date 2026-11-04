"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import glob
import csv
import json
import os
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from lavis.models.sega3d_models.seg_loss import get_iou
import numpy as np
import torch
import torch.distributed as dist
import datetime
import json
import os

class SEGA3DSegTask(BaseTask):

    def __init__(self, num_beams, max_len, min_len, evaluate, num_ans_candidates, inference_method='rank', prompt='', save_results=False):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt
        self.save_results = save_results
        self.save_dir = 'refer_preds'
        self.count = 0
        self.best_metric = -1.0
        self.best_epoch = -1
        self.best_save_dir = 'eval_history3_best'
        self.answer_list = None
        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        num_beams = run_cfg.get('num_beams', 3)
        max_len = run_cfg.get('max_len', 10)
        min_len = run_cfg.get('min_len', 1)
        evaluate = run_cfg.get('evaluate', False)
        inference_method = run_cfg.get('inference_method', 'rank')
        num_ans_candidates = run_cfg.get('num_ans_candidates', 128)
        prompt = run_cfg.get('prompt', '')
        save_results = run_cfg.get('save_results', False)
        return cls(num_beams=num_beams, max_len=max_len, min_len=min_len, evaluate=evaluate, num_ans_candidates=num_ans_candidates, inference_method=inference_method, prompt=prompt, save_results=save_results)

    def valid_step(self, model, samples):
        pred_result = model.predict_seg(samples=samples, answer_list=None, inference_method=self.inference_method, num_beams=self.num_beams, max_len=self.max_len, min_len=self.min_len, num_ans_candidates=self.num_ans_candidates, prompt=self.prompt)
        assert len(samples['gt_pmasks']) == 1, 'current only support batch size = 1'
        gt_pmask = samples['gt_pmasks'][0]
        is_point_level = pred_result.get('is_point_level', False)
        raw_pred = pred_result['masks'][-1]
        if is_point_level:
            pred_pmask = raw_pred.squeeze()
            spiou = 0.0
        else:
            gt_spmask = samples['gt_spmasks'][0]
            pred_spmask = raw_pred.squeeze()
            spiou = get_iou(pred_spmask, gt_spmask, pred_confidence=model.pred_confidence)
            pred_pmask = pred_spmask[samples['superpoints']]
        piou = get_iou(pred_pmask, gt_pmask, pred_confidence=model.pred_confidence)
        scan_id = samples['scan_ids'][0] if 'scan_ids' in samples else 'unknown'
        obj_id = samples['object_ids'][0] if 'object_ids' in samples else 'unknown'
        ann_id = samples['ann_ids'][0] if 'ann_ids' in samples else 'unknown'
        text_input = samples['text_input'][0] if 'text_input' in samples else 'None'
        obj_cat = samples['object_cats'][0] if 'object_cats' in samples else 'unknown'
        piou_float = float(piou.detach().cpu().item()) if torch.is_tensor(piou) else float(piou)
        region_piou = 0.0
        if 'region_piou' in pred_result:
            val = pred_result['region_piou']
            region_piou = val.item() if torch.is_tensor(val) else val
        region_recall = 0.0
        if 'region_recall' in pred_result:
            val = pred_result['region_recall']
            region_recall = val.item() if torch.is_tensor(val) else val

        def _pick_batch_value(x, idx=0, default=None):
            if x is None:
                return default
            if torch.is_tensor(x):
                if x.numel() == 0:
                    return default
                return x.detach().cpu().view(-1)[idx].item()
            if isinstance(x, (list, tuple)):
                if len(x) <= idx:
                    return default
                return x[idx]
            return x
        has_sem_split = bool(pred_result.get('has_sem_split', False))
        sem_split = _pick_batch_value(pred_result.get('sem_split', None), idx=0, default='unknown')
        target_sem_inst_count = int(_pick_batch_value(pred_result.get('target_sem_inst_counts', None), idx=0, default=-1))
        target_sem_id = int(_pick_batch_value(pred_result.get('target_sem_ids', None), idx=0, default=-1))
        is_unique_sem = int(_pick_batch_value(pred_result.get('is_unique_sem', None), idx=0, default=0))
        is_multiple_sem = int(_pick_batch_value(pred_result.get('is_multiple_sem', None), idx=0, default=0))
        target_valid_for_sem_count = int(_pick_batch_value(pred_result.get('target_valid_for_sem_count', None), idx=0, default=0))
        if not has_sem_split:
            sem_split = 'unknown'
        selector_topk_diagnostic = _pick_batch_value(pred_result.get('selector_topk_diagnostics', None), idx=0, default=None)
        result = dict(scan_id=samples['scan_ids'][0], object_id=samples['object_ids'][0], ann_id=samples['ann_ids'][0], obj_cat=obj_cat, piou=piou, spiou=spiou, region_piou=region_piou, region_recall=region_recall, gt_pmask=gt_pmask, pred_pmask=pred_pmask, has_sem_split=has_sem_split, sem_split=sem_split, target_sem_inst_count=target_sem_inst_count, target_sem_id=target_sem_id, is_unique_sem=is_unique_sem, is_multiple_sem=is_multiple_sem, target_valid_for_sem_count=target_valid_for_sem_count, eval_topk=int(pred_result.get('eval_topk', 0)), selector_topk_diagnostic=selector_topk_diagnostic)
        if self.save_results:
            import pickle
            os.makedirs(self.save_dir, exist_ok=True)
            ann_id = result['ann_id']
            scan_id = result['scan_id']
            gt_pmask = result['gt_pmask'].cpu().numpy()
            pred_pmask = result['pred_pmask'].sigmoid().cpu().numpy()
            text_input = samples['text_input'][0]
            sp_filename = samples['sp_filenames'][0]
            with open(os.path.join(self.save_dir, f'{self.count}.pkl'), 'wb') as f:
                pickle.dump({'scan_id': scan_id, 'gt_pmask': gt_pmask, 'pred_pmask': pred_pmask, 'text_input': text_input, 'sp_filename': sp_filename}, f)
            self.count += 1
        return [{'result': result}]

    def after_evaluation(self, val_result, split_name, epoch):
        pious = []
        spious = []
        region_pious = []
        region_recalls = []
        topk_diagnostic_records = []
        cat2_pious = {}
        cat2_region_pious = {}
        sem_split2_pious = {'unique': [], 'multiple': []}
        sem_split2_region_pious = {'unique': [], 'multiple': []}
        sem_split2_counts = {'unique': 0, 'multiple': 0, 'unknown': 0, 'disabled': 0}
        for i, result in enumerate(val_result):
            res_dict = result['result']
            piou = res_dict['piou']
            spiou = res_dict['spiou']
            pious.append(piou)
            spious.append(spiou)
            topk_diag = res_dict.get('selector_topk_diagnostic', None)
            if isinstance(topk_diag, dict):
                piou_value = float(piou.detach().cpu().item()) if torch.is_tensor(piou) else float(piou)
                topk_diagnostic_records.append({'diagnostic': topk_diag, 'final_iou': piou_value})
            if bool(res_dict.get('has_sem_split', False)):
                sem_split = res_dict.get('sem_split', 'unknown')
                if sem_split in ['unique', 'multiple']:
                    sem_split2_pious[sem_split].append(piou)
                    if 'region_piou' in res_dict:
                        sem_split2_region_pious[sem_split].append(res_dict['region_piou'])
                    sem_split2_counts[sem_split] += 1
                else:
                    sem_split2_counts['unknown'] += 1
            else:
                sem_split2_counts['disabled'] += 1
            if 'region_piou' in res_dict:
                r_piou = res_dict['region_piou']
                if torch.is_tensor(r_piou):
                    r_piou = r_piou.cpu()
                region_pious.append(r_piou)
            else:
                region_pious.append(0.0)
            if 'region_recall' in res_dict:
                region_recalls.append(res_dict['region_recall'])
            else:
                region_recalls.append(0.0)
            obj_cat = res_dict.get('obj_cat', 'unknown')
            if obj_cat not in cat2_pious:
                cat2_pious[obj_cat] = []
                cat2_region_pious[obj_cat] = []
            cat2_pious[obj_cat].append(piou)
            if 'region_piou' in res_dict:
                cat2_region_pious[obj_cat].append(res_dict['region_piou'])
        pious = torch.stack(pious, dim=0).cpu().numpy()

        def _distributed_sum(values):
            use_dist = dist.is_available() and dist.is_initialized()
            if use_dist and str(dist.get_backend()).lower().endswith('nccl'):
                reduce_device = torch.device('cuda', torch.cuda.current_device())
            else:
                reduce_device = torch.device('cpu')
            stats = torch.tensor(values, dtype=torch.float64, device=reduce_device)
            if use_dist:
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            return stats.detach().cpu().tolist()
        fine_stats = _distributed_sum([float(pious.size), float(pious.sum()), float((pious > 0.5).sum()), float((pious > 0.25).sum())])
        global_count = max(int(fine_stats[0]), 1)
        miou = float(fine_stats[1] / global_count)
        precision_half = float(fine_stats[2] / global_count)
        precision_quarter = float(fine_stats[3] / global_count)
        print('\nVal result (FINE): mIoU/Acc50/Acc25 {:.4f}/{:.4f}/{:.4f}'.format(miou, precision_half, precision_quarter))

        def _values_to_numpy(vals):
            arr = []
            for x in vals:
                if torch.is_tensor(x):
                    arr.append(float(x.detach().cpu().item()))
                else:
                    arr.append(float(x))
            if len(arr) == 0:
                return np.array([], dtype=np.float32)
            return np.array(arr, dtype=np.float32)

        def _pearson_from_sums(n, sum_x, sum_y, sum_x2, sum_y2, sum_xy):
            if n < 2:
                return 0.0
            numerator = n * sum_xy - sum_x * sum_y
            denom_x = max(n * sum_x2 - sum_x * sum_x, 0.0)
            denom_y = max(n * sum_y2 - sum_y * sum_y, 0.0)
            denominator = float(np.sqrt(denom_x * denom_y))
            return float(numerator / denominator) if denominator > 1e-12 else 0.0
        topk_summary = {}
        topk_rows = []
        if topk_diagnostic_records:
            first_diag = topk_diagnostic_records[0]['diagnostic']
            requested_eval_topk = int(first_diag.get('requested_eval_topk', 0))
            topk_keys = sorted((int(k) for k in first_diag.get('by_k', {}).keys()))
            for k in topk_keys:
                local_values = []
                for record in topk_diagnostic_records:
                    diag_k = record['diagnostic'].get('by_k', {}).get(str(k), None)
                    if diag_k is None:
                        continue
                    local_values.append((float(diag_k['effective_k']), float(diag_k['oracle_iou']), float(diag_k['recall_iou_025']), float(diag_k['recall_iou_050']), float(diag_k['oracle_target_hit']), float(record['final_iou'])))
                if local_values:
                    arr = np.asarray(local_values, dtype=np.float64)
                else:
                    arr = np.zeros((0, 6), dtype=np.float64)
                x = arr[:, 1] if arr.size else np.zeros((0,))
                y = arr[:, 5] if arr.size else np.zeros((0,))
                sums = _distributed_sum([float(arr.shape[0]), float(arr[:, 0].sum()) if arr.size else 0.0, float(arr[:, 1].sum()) if arr.size else 0.0, float(arr[:, 2].sum()) if arr.size else 0.0, float(arr[:, 3].sum()) if arr.size else 0.0, float(arr[:, 4].sum()) if arr.size else 0.0, float(x.sum()), float(y.sum()), float((x * x).sum()), float((y * y).sum()), float((x * y).sum())])
                count = max(int(sums[0]), 1)
                correlation = _pearson_from_sums(sums[0], sums[6], sums[7], sums[8], sums[9], sums[10])
                topk_rows.append({'K': int(k), 'count': int(sums[0]), 'mean_effective_K': float(sums[1] / count), 'oracle_mIoU_at_K': float(sums[2] / count), 'selector_recall_at_K_iou025': float(sums[3] / count), 'selector_recall_at_K_iou050': float(sums[4] / count), 'oracle_target_hit_at_K': float(sums[5] / count), 'oracle_iou_final_iou_pearson': correlation, 'run_eval_topk': requested_eval_topk, 'final_mIoU': float(miou), 'final_Acc_at_025': float(precision_quarter), 'final_Acc_at_050': float(precision_half)})
            bank_values = np.asarray([[float(record['diagnostic'].get('num_candidates', 0)), float(record['diagnostic'].get('bank_oracle_iou', 0.0)), float(record['diagnostic'].get('bank_recall_iou_025', 0.0)), float(record['diagnostic'].get('bank_recall_iou_050', 0.0)), float(record['final_iou'])] for record in topk_diagnostic_records], dtype=np.float64)
            bank_x = bank_values[:, 1]
            bank_y = bank_values[:, 4]
            bank_sums = _distributed_sum([float(bank_values.shape[0]), float(bank_values[:, 0].sum()), float(bank_values[:, 1].sum()), float(bank_values[:, 2].sum()), float(bank_values[:, 3].sum()), float(bank_x.sum()), float(bank_y.sum()), float((bank_x * bank_x).sum()), float((bank_y * bank_y).sum()), float((bank_x * bank_y).sum())])
            bank_count = max(int(bank_sums[0]), 1)
            bank_summary = {'count': int(bank_sums[0]), 'mean_num_candidates': float(bank_sums[1] / bank_count), 'oracle_mIoU': float(bank_sums[2] / bank_count), 'recall_iou025': float(bank_sums[3] / bank_count), 'recall_iou050': float(bank_sums[4] / bank_count), 'oracle_iou_final_iou_pearson': _pearson_from_sums(bank_sums[0], bank_sums[5], bank_sums[6], bank_sums[7], bank_sums[8], bank_sums[9])}
            topk_summary = {'split': split_name, 'epoch': epoch, 'requested_eval_topk': requested_eval_topk, 'final_metrics': {'mIoU': float(miou), 'Acc@0.25': float(precision_quarter), 'Acc@0.50': float(precision_half), 'count': int(fine_stats[0])}, 'candidate_bank': bank_summary, 'selector_topk': topk_rows}
            is_main_rank = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
            if is_main_rank:
                print('\n==== Selector Top-K recall stats ====')
                print('K | Oracle-mIoU | Recall@0.25 | Recall@0.50 | Oracle-target Hit | Corr(final IoU)')
                for row in topk_rows:
                    print(f"{row['K']:>2d} | {row['oracle_mIoU_at_K']:.4f} | {row['selector_recall_at_K_iou025']:.4f} | {row['selector_recall_at_K_iou050']:.4f} | {row['oracle_target_hit_at_K']:.4f} | {row['oracle_iou_final_iou_pearson']:.4f}")
                print(f"Candidate bank upper bound: Oracle-mIoU={bank_summary['oracle_mIoU']:.4f}, Recall@0.25={bank_summary['recall_iou025']:.4f}, Recall@0.50={bank_summary['recall_iou050']:.4f}")
                print('=====================================\n')
                output_dir = registry.get_path('output_dir')
                metrics_dir = os.path.join(output_dir, 'topk_metrics')
                os.makedirs(metrics_dir, exist_ok=True)
                stem = f'topk_recall_{split_name}_evalK{requested_eval_topk}_epoch{epoch}'
                json_path = os.path.join(metrics_dir, stem + '.json')
                csv_path = os.path.join(metrics_dir, stem + '.csv')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(topk_summary, f, ensure_ascii=False, indent=2)
                if topk_rows:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=list(topk_rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(topk_rows)
                print(f'[TOPK] saved JSON: {json_path}')
                print(f'[TOPK] saved CSV : {csv_path}')
        sem_split_metrics = {}
        if sem_split2_counts['unique'] > 0 or sem_split2_counts['multiple'] > 0:
            print('\n==== Unique / Multiple semantic split stats ====')
            print(f"[SEM-SPLIT] unique={sem_split2_counts['unique']} multiple={sem_split2_counts['multiple']} unknown={sem_split2_counts['unknown']} disabled={sem_split2_counts['disabled']}")
            for split_key in ['unique', 'multiple']:
                arr = _values_to_numpy(sem_split2_pious[split_key])
                if arr.size == 0:
                    print(f'[{split_key.upper()}] N=0')
                    sem_split_metrics[split_key] = {'mIoU': 0.0, 'Acc50': 0.0, 'Acc25': 0.0, 'count': 0}
                    continue
                split_miou = float(arr.mean())
                split_acc50 = float((arr > 0.5).sum() / arr.size)
                split_acc25 = float((arr > 0.25).sum() / arr.size)
                r_arr = _values_to_numpy(sem_split2_region_pious[split_key])
                if r_arr.size > 0:
                    split_region_miou = float(r_arr.mean())
                    split_region_acc50 = float((r_arr > 0.5).sum() / r_arr.size)
                    split_region_acc25 = float((r_arr > 0.25).sum() / r_arr.size)
                else:
                    split_region_miou = 0.0
                    split_region_acc50 = 0.0
                    split_region_acc25 = 0.0
                print(f'[{split_key.upper()}] N={arr.size} Fine mIoU/Acc50/Acc25={split_miou:.4f}/{split_acc50:.4f}/{split_acc25:.4f} | Region mIoU/Acc50/Acc25={split_region_miou:.4f}/{split_region_acc50:.4f}/{split_region_acc25:.4f}')
                sem_split_metrics[split_key] = {'fine': {'mIoU': split_miou, 'Acc50': split_acc50, 'Acc25': split_acc25}, 'region': {'mIoU': split_region_miou, 'Acc50': split_region_acc50, 'Acc25': split_region_acc25}, 'count': int(arr.size)}
            print('===============================================\n')
        else:
            sem_split_metrics = {}
        if len(region_pious) > 0:
            r_list_np = []
            for x in region_pious:
                if torch.is_tensor(x):
                    r_list_np.append(x.item())
                else:
                    r_list_np.append(x)
            r_arr = np.array(r_list_np)
            r_acc50 = (r_arr > 0.5).sum() / r_arr.size
            r_acc25 = (r_arr > 0.25).sum() / r_arr.size
            r_miou = r_arr.mean()
            print('Val result (REGION): mIoU/Acc50/Acc25 {:.4f}/{:.4f}/{:.4f}'.format(r_miou, r_acc50, r_acc25))
        else:
            r_miou, r_acc50, r_acc25 = (0.0, 0.0, 0.0)
        summary_data = {'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'epoch': epoch, 'split': split_name, 'global_metrics_fine': {'mIoU': float(miou), 'Acc50': float(precision_half), 'Acc25': float(precision_quarter), 'count': int(pious.size)}, 'global_metrics_region': {'mIoU': float(r_miou), 'Acc50': float(r_acc50), 'Acc25': float(r_acc25), 'count': len(region_pious)}, 'sem_split_metrics': sem_split_metrics, 'sem_split_counts': sem_split2_counts, 'category_metrics': {}, 'topk_metrics': topk_summary}
        if len(region_recalls) > 0:
            r_rec_arr = np.array(region_recalls)
            mean_recall = r_rec_arr.mean()
            print(f'Val result (REGION): mIoU={r_miou:.4f} | Recall={mean_recall:.4f}')
        print('\n==== Per-category IoU stats ====')
        for cat, p_list in cat2_pious.items():
            if len(p_list) == 0:
                continue
            p_arr = torch.stack(p_list, dim=0).cpu().numpy()
            m_cat = p_arr.mean()
            acc50 = (p_arr > 0.5).sum().astype(float) / p_arr.size
            acc25 = (p_arr > 0.25).sum().astype(float) / p_arr.size
            r_val = 0.0
            if cat in cat2_region_pious and len(cat2_region_pious[cat]) > 0:
                r_sub_list = []
                for x in cat2_region_pious[cat]:
                    if torch.is_tensor(x):
                        r_sub_list.append(x.item())
                    else:
                        r_sub_list.append(x)
                r_val = np.array(r_sub_list).mean()
            print(f'[{cat}] Fine: {m_cat:.4f} | Reg: {r_val:.4f} | N={p_arr.size}')
            summary_data['category_metrics'][cat] = {'fine': {'mIoU': float(m_cat), 'Acc50': float(acc50), 'Acc25': float(acc25)}, 'region': {'mIoU': float(r_val)}, 'count': int(p_arr.size)}
        print('================================\n')
        ret = {'mIoU': float(miou), 'Acc@0.50': float(precision_half), 'Acc@0.25': float(precision_quarter), 'region_mIoU': float(r_miou), 'region_Acc@0.50': float(r_acc50), 'region_Acc@0.25': float(r_acc25), 'agg_metrics': float(miou)}
        if sem_split_metrics:
            for split_key in ['unique', 'multiple']:
                if split_key in sem_split_metrics and sem_split_metrics[split_key].get('count', 0) > 0:
                    ret[f'{split_key}_mIoU'] = float(sem_split_metrics[split_key]['fine']['mIoU'])
                    ret[f'{split_key}_Acc@0.50'] = float(sem_split_metrics[split_key]['fine']['Acc50'])
                    ret[f'{split_key}_Acc@0.25'] = float(sem_split_metrics[split_key]['fine']['Acc25'])
                    ret[f'{split_key}_count'] = int(sem_split_metrics[split_key]['count'])
        if topk_summary:
            eval_k = int(topk_summary['requested_eval_topk'])
            matching_row = next((row for row in topk_rows if int(row['K']) == eval_k), None)
            if matching_row is not None:
                ret[f'SelectorRecall@{eval_k}/IoU0.25'] = float(matching_row['selector_recall_at_K_iou025'])
                ret[f'SelectorRecall@{eval_k}/IoU0.50'] = float(matching_row['selector_recall_at_K_iou050'])
                ret[f'Oracle-mIoU@{eval_k}'] = float(matching_row['oracle_mIoU_at_K'])
        return ret
import numpy as np
import re
import math
import numpy as np
import torch
import os
import json
import os
import json
import textwrap
import numpy as np
import torch
MULTI_VIEWS = [('Iso-45', 25, 45), ('Iso-135', 25, 135), ('Iso-225', 25, 225), ('Iso-315', 25, 315), ('Top', 90, -90), ('Front', 0, -90)]
PROJ_VIEWS = [('XY', 0, 1), ('XZ', 0, 2), ('YZ', 1, 2)]
