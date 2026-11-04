"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import re
import json
import torch
import numpy as np
import os.path as osp
from lavis.common.deferred_import import DeferredModule
pointgroup_ops = DeferredModule("pointgroup_ops")
import math
from PIL import Image
from PIL import ImageFile
import torch_scatter
from typing import Dict, Sequence, Tuple, Union
from torch_scatter import scatter_mean
from lavis.datasets.datasets.base_dataset import BaseDataset
import glob
import random
import scipy.ndimage as ndimage
from scipy import interpolate
from scipy.spatial.distance import cdist # (这个在 `elastic` 里好像没用，但你之前的脚本里有，以防万一)
from torch_scatter import scatter_mean
from lavis.datasets.datasets.scannet200_constants import CLASS_LABELS_200,VALID_CLASS_IDS_200

class SEGA3DScanReferDataset(BaseDataset):
    def __init__(self, text_processor, pts_root, ann_paths, candidate_cfg=None, split=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(text_processor, pts_root, ann_paths)
        candidate_cfg = dict(candidate_cfg or {})
        self.scene_ids = {}
        n = 0
        self.use_xyz = True
        self.mode = 4
        new_annotation = []
        for ann in self.annotation:
            try:
                img_id = ann["scene_id"]
                if img_id not in self.scene_ids.keys():
                    self.scene_ids[img_id] = n
                    n += 1
                new_annotation.append(ann)
            except:
                pass


        if split is None:
            split = "train" if "train" in ann_paths[0] else "val"
        self.training = split == "train"
        self.prefix = "train_sp_refined" if self.training else "val_sp_refined"

        self.with_label = True
        self.annotation = new_annotation
        self.sp_filenames = self.get_sp_filenames()
        
        #=====================================================================
        # (新增) 建立 scene_id -> 文件路径的映射，避免 __getitem__ 里 for 循环
        self.sp_map = {}
        for p in self.sp_filenames:
            # 这里的 scene_id 提取按你文件命名来（与 self.annotation 里的 "scene_id" 对齐）
            # 例如 xxx/{scene_id}_refer.pth
            sid = osp.basename(p).replace('_refer.pth', '')
            self.sp_map[sid] = p

        #=====================================================================
        self.short_question_list = QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.with_elastic = True
        self.aug = True     
        # ===== 新增：instance meta 文件目录与映射 =====
        # 你把 instance 文件放到哪个目录，就把这里改成对应路径
        # 推荐目录类似：pts_root/scannetv2/{prefix}_instmeta/
        self.inst_dir = osp.join(self.pts_root, "scannetv2", f"{self.prefix}_instance")
        print(f"[INFO] 将从此目录加载 instance meta: {self.inst_dir}")

        self.inst_filenames = sorted(glob.glob(osp.join(self.inst_dir, "*.*")))
        assert len(self.inst_filenames) > 0, f"Empty instmeta dir: {self.inst_dir}"

        self.inst_map = {}
        for p in self.inst_filenames:
            base = osp.basename(p)
            m = re.search(r"(scene\d{4}_\d{2})", base)
            if m:
                sid = m.group(1)
            else:
                # fallback：按你命名可能是 {scene_id}_instmeta.pt / {scene_id}_instance.pth
                sid = base.replace("_instmeta.pt", "").replace("_instance.pt", "").replace("_instance.pth", "")
            self.inst_map[sid] = p
        # ============================================
        self.gt_prob = 0.3 if self.training else 0.0
        self.use_prediction = True

        self.copy_paste_prob = 0.5 if self.training else 0.0
        self.tem = 0
        self.tem2 = 0
        # ================================
        # rescue pseudo-instance configs
        # ================================
        self.enable_rescue_candidates = False   # 旧逻辑彻底关掉

        # 新：独立 rescue bank
        self.enable_rescue_bank = True
        self.rescue_anchor_topk = 4
        self.rescue_score_thresh = 0.10

        self.rescue_min_points = 80
        self.rescue_max_points_ratio = 0.12

        self.rescue_dilate_scales = (1.15, 1.35)   # 两个尺度即可
        self.rescue_use_same_class_union = True
        self.rescue_union_topk = 1
        self.rescue_union_max_dist_ratio = 1.25

        self.rescue_min_added_ratio = 0.10
        self.rescue_max_added_ratio = 1.00

        self.rescue_max_bank = 6
        self.dump_dataset_debug_jsonl = False
        self.dump_dataset_debug_path = "outputs/dataset_stats_rank{rank}.jsonl"


        # ===== 实验三：v2 + 200 并集 =====
        self.use_prediction = True

        # 两个候选源
        self.pred_root_v2 = candidate_cfg.get("scannet20", osp.join(pts_root, "candidates", "scannet20"))
        self.pred_root_200 = candidate_cfg.get("scannet200", osp.join(pts_root, "candidates", "scannet200"))

        # 是否启用并集
        self.use_merged_prediction = True

        # 200 侧必须过白名单
        self.pred_whitelist_txt = candidate_cfg.get("whitelist", osp.join(pts_root, "candidates", "whitelist_standard.txt"))
        self.enforce_pred_whitelist = True
        self.strict_pred_whitelist = False
        self._pred_missing_whitelist_printed = False

        # merge 去重参数
        self.pred_merge_dedup_iou = 0.98

        # debug 只打前两次
        self._pred_debug_count = 0
        self._pred_debug_max = 2
        self._pred_whitelist_init_printed = False

        # 读白名单名字
        self.pred_whitelist_names = self._load_pred_whitelist_names(self.pred_whitelist_txt)

        # 这两个在第一次真正读到 pred_classes 时再自动确定
        self.pred_id_space = None          # "contig" or "raw"
        self.pred_label2id = None
        self.pred_whitelist_ids = None

        # 名字别名：解决 txt 名字和 ScanNet200 官方名字不一致
        self.pred_name_alias = {
            "computer_tower": "computer tower",
            "paper_towel_dispenser": "paper towel dispenser",
            "soap_dispenser": "soap dispenser",
            "toilet_paper": "toilet paper",
            "toilet_paper_dispenser": "toilet paper dispenser",
            "storage_bin": "storage bin",

            # 下面这些是近似映射
            "night_lamp": "lamp",
            "organizer_shelf": "storage organizer",
            "shoe_rack": "rack",
            "instrument_case": "guitar case",
            "rug": "mat",
        }


        
    def _normalize_label_name(self, s: str) -> str:
        return s.strip().lower().replace("_", " ")
    
    def _detect_pred_id_space(self, class_ids_np: np.ndarray) -> str:
        u = set(np.unique(class_ids_np).tolist())

        contig_ids = set(range(len(CLASS_LABELS_200)))   # 0..199
        raw_ids = set(VALID_CLASS_IDS_200)               # 官方 raw id

        if len(u) == 0:
            raise ValueError("[WHITELIST] empty pred_classes")

        if u.issubset(contig_ids):
            return "contig"

        if u.issubset(raw_ids):
            return "raw"

        raise ValueError(
            f"[WHITELIST] unknown pred_classes id space: "
            f"min={class_ids_np.min()}, max={class_ids_np.max()}, "
            f"sample={sorted(list(u))[:20]}"
        )
    
    def _build_scannet200_label2id(self, id_space: str):
        names = [self._normalize_label_name(x) for x in CLASS_LABELS_200]

        if id_space == "contig":
            return {name: i for i, name in enumerate(names)}

        if id_space == "raw":
            return {name: raw_id for name, raw_id in zip(names, VALID_CLASS_IDS_200)}

        raise ValueError(f"[WHITELIST] unknown id_space: {id_space}")
    

    
    def get_sp_filenames(self):
        filenames = glob.glob(osp.join(self.pts_root, 'scannetv2', self.prefix, '*' + '_refer.pth'))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        return filenames
        
    def load(self, filename):

        
        data = torch.load(filename)

        # 兼容 tuple / list / dict 三种保存方式
        if isinstance(data, (list, tuple)):
            if self.with_label:
                # 你现在的保存格式： (xyz, rgb, superpoint, sem_labels, instance_labels)
                xyz, rgb, superpoint, sem_labels, instance_labels = data
            else:
                xyz, rgb, superpoint = data
                sem_labels = np.zeros(xyz.shape[0], dtype=np.float32)
                instance_labels = np.zeros(xyz.shape[0], dtype=np.float32)
        elif isinstance(data, dict):
            # 如果后面你想改成 dict 存，这里也可以兼容
            xyz = data["xyz"]
            rgb = data["rgb"]
            superpoint = data["superpoint"]
            sem_labels = data.get("sem_labels", np.zeros(xyz.shape[0], dtype=np.float32))
            instance_labels = data.get("instance_labels", np.zeros(xyz.shape[0], dtype=np.float32))
        else:
            raise TypeError(f"Unrecognized refer file format: {type(data)} in {filename}")

        # 统一转成 numpy，方便后面 data_aug / elastic 继续用 numpy 写法
        if torch.is_tensor(xyz):
            xyz = xyz.cpu().numpy()
        if torch.is_tensor(rgb):
            rgb = rgb.cpu().numpy()
        if torch.is_tensor(superpoint):
            superpoint = superpoint.cpu().numpy()
        if torch.is_tensor(sem_labels):
            sem_labels = sem_labels.cpu().numpy()
        if torch.is_tensor(instance_labels):
            instance_labels = instance_labels.cpu().numpy()
        return xyz, rgb, superpoint, sem_labels, instance_labels

    #备份
    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label, desc=None, camera=None):
        xyz_world = xyz.copy()

        # 方向词检测（你已经有 _is_view_dependent）
        view_dep = self._is_view_dependent(desc) if desc is not None else False

        if self.aug:
            flip_flag = (not view_dep)
            rot_flag  = (not view_dep)
            xyz_middle, m = self.data_aug(xyz, jitter=True, flip=flip_flag, rot=rot_flag, view=True)
        else:
            xyz_middle = xyz.copy()
            m = np.eye(3, dtype=np.float32)

        # === camera 同步到增强坐标系（和 xyz_middle 同坐标）===
        cam_pos = np.zeros((3,), dtype=np.float32)
        cam_lookat = np.zeros((3,), dtype=np.float32)
        cam_dir = np.zeros((3,), dtype=np.float32)
        cam_valid = False

        if camera is not None:
            pos = np.array(camera["position"], dtype=np.float32)
            lookat = np.array(camera["lookat"], dtype=np.float32)

            # 关键：与 xyz 同样右乘 m
            cam_pos = pos @ m
            cam_lookat = lookat @ m

            d = cam_lookat - cam_pos
            cam_dir = d / (np.linalg.norm(d) + 1e-6)
            cam_valid = True

        # === 你原来的后续逻辑不动 ===
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * 50
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)

        valid_idxs = xyz.min(1) >= 0
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]

        sp_valid = superpoint[valid_idxs].astype(np.int64)
        uniq_sp_ids, sp_new = np.unique(sp_valid, return_inverse=True)
        superpoint = sp_new

        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        orig_idx_keep = np.nonzero(valid_idxs)[0].astype(np.int64)

        xyz_world = xyz_world[valid_idxs]

        # 额外把 camera/增强信息返回
        return (xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label,
                uniq_sp_ids, xyz_world, orig_idx_keep,
                cam_pos, cam_lookat, cam_dir, cam_valid, view_dep)

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label, camera=None, desc=None):
        xyz_world = xyz.copy()

        xyz_middle = xyz  # 与 camera 同坐标系（原始 scan 坐标）

        # ====== camera（eval 无增强，直接用原值）======
        cam_pos = np.zeros((3,), np.float32)
        cam_lookat = np.zeros((3,), np.float32)
        cam_dir = np.zeros((3,), np.float32)
        cam_valid = False
        view_dep = self._is_view_dependent(desc) if desc is not None else False

        if camera is not None:
            pos = np.array(camera["position"], dtype=np.float32)
            lookat = np.array(camera["lookat"], dtype=np.float32)
            cam_pos = pos
            cam_lookat = lookat
            d = cam_lookat - cam_pos
            cam_dir = d / (np.linalg.norm(d) + 1e-6)
            cam_valid = True
        # ============================================

        xyz = xyz_middle * 50
        xyz -= xyz.min(0)
        valid_idxs = np.ones(xyz.shape[0], dtype=bool)

        sp_valid = superpoint[valid_idxs].astype(np.int64)
        uniq_sp_ids, sp_new = np.unique(sp_valid, return_inverse=True)
        superpoint = sp_new

        instance_label = self.get_cropped_inst_label(instance_label, valid_idxs)
        xyz_world = xyz_world[valid_idxs]
        orig_idx_keep = np.nonzero(valid_idxs)[0].astype(np.int64)

        return (xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label, uniq_sp_ids, xyz_world, orig_idx_keep,
                cam_pos, cam_lookat, cam_dir, cam_valid, view_dep)


    def data_aug(self, xyz, jitter=False, flip=False, rot=False, view=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        '''
        if rot:
            theta = (np.random.rand() - 0.5) * 2 * np.pi / 18
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        '''
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        xyz2 = np.matmul(xyz, m) 
        if view:
            return (xyz2, m) 
        else:
            return np.matmul(xyz, m) 


    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag

    def get_cropped_inst_label(self, instance_label: np.ndarray, valid_idxs: np.ndarray) -> np.ndarray:
        r"""
        get the instance labels after crop operation and recompact

        Args:
            instance_label (np.ndarray, [N]): instance label ids of point cloud
            valid_idxs (np.ndarray, [N]): boolean valid indices

        Returns:
            np.ndarray: processed instance labels
        """
        instance_label = instance_label[valid_idxs]
        j = 0
        while j < instance_label.max():
            if len(np.where(instance_label == j)[0]) == 0:
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label
    
    def get_ref_mask(self, instance_label, superpoint, object_id):
        ref_lbl = instance_label == object_id
        gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask

    def get_ref_mask_region(self, ref_lbl, superpoint):
        gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask
    

    def __getitem__(self, index: int) -> Tuple:
        
        # -------------------------------------------------------
        # 1. 基础文本与标注信息准备
        # -------------------------------------------------------
        data = self.annotation[index]
        scan_id = data["scene_id"]
        object_id = int(data["object_id"])
        ann_id = int(data["ann_id"])
        camera = data["camera"]        
        # 文本处理 (完全保留原逻辑)
        description = self.text_processor(data["description"])
        question_template = random.choice(self.short_question_list)
        question = question_template.format(description=description)
        obj_cat = data["object_name"]
        
        # 答案生成 (完全保留原逻辑)
        answers = [random.choice(self.answer_list)]

        # -------------------------------------------------------
        # 2. 加载原始点云数据 (作为基准)
        # -------------------------------------------------------
        sp_filename = self.sp_map.get(scan_id)
        if sp_filename is None:
             sp_filename = os.path.join(self.pts_root, "scannetv2", self.prefix, f"{scan_id}_refer.pth")
             if not os.path.exists(sp_filename):
                 raise FileNotFoundError(f"Missing refer file for {scan_id}")

        # data_loaded: (xyz, rgb, superpoint, sem_labels, instance_labels)
        # 注意：这里的 xyz 是原始未裁剪的，RGB 是原始颜色
        data_loaded = self.load(sp_filename)
        
        # -------------------------------------------------------
        # 3. 检查模式：是使用 Pointcept 预测数据，还是使用 GT？
        # -------------------------------------------------------
        use_prediction_global = getattr(self, "use_prediction", False)
        pred_filename_v2 = os.path.join(self.pred_root_v2, f"{scan_id}.pt") if use_prediction_global else None
        pred_filename_200 = os.path.join(self.pred_root_200, f"{scan_id}.pt") if use_prediction_global else None

        has_pred_v2 = (pred_filename_v2 is not None and os.path.exists(pred_filename_v2))
        has_pred_200 = (pred_filename_200 is not None and os.path.exists(pred_filename_200))
        has_any_pred = has_pred_v2 or has_pred_200
        if not self.training and not (has_pred_v2 and has_pred_200):
            missing = [p for p, exists in [(pred_filename_v2, has_pred_v2),
                                          (pred_filename_200, has_pred_200)] if not exists]
            raise FileNotFoundError(f"Missing predicted candidate banks for {scan_id}: {missing}")

        # 默认模式
        mode_switch = "gt"

        if use_prediction_global and has_any_pred:
            if self.training:
                # 训练时：掷骰子
                if random.random() < self.gt_prob:
                    mode_switch = "gt"   # 强制用 GT
                else:
                    mode_switch = "pred" # 用 Pointcept 预测
            else:
                # 测试时：强制用 Pred (模拟真实场景)
                mode_switch = "pred"
        else:
            # 如果没开预测或者没文件，只能用 GT
            mode_switch = "gt"
        is_pred_mode = (mode_switch == "pred")


        # -------------------------------------------------------
        # 4. 执行 Transform (核心步骤：裁剪 + 增强)
        # -------------------------------------------------------
        # 无论哪种模式，都必须经过这一步来获得 xyz_middle (归一化坐标) 和 mask

        if self.training:
            data_trans = self.transform_train(*data_loaded, desc=data["description"], camera=camera)
            (xyz, xyz_middle, feat, superpoint_aug, semantic_label, instance_label, uniq_sp_ids_full, xyz_world, orig_idx_keep,
            cam_pos, cam_lookat, cam_dir, cam_valid, view_dep) = data_trans
        else:
            data_trans = self.transform_test(*data_loaded, camera=camera, desc=data["description"])
            (xyz, xyz_middle, feat, superpoint_aug, semantic_label, instance_label, uniq_sp_ids_full, xyz_world, orig_idx_keep,
            cam_pos, cam_lookat, cam_dir, cam_valid, view_dep) = data_trans
            # 测试阶段如果暂时不用 viewpoint，可置零；或用未增强的原 camera（推荐置零+cam_valid=0）

        # -------------------------------------------------------
        # 4.5 当前 target 语义类的 unique / multiple 统计
        # -------------------------------------------------------
        sem_inst_info = self.get_semantic_instance_info(
            semantic_label=semantic_label,
            instance_label=instance_label,
            object_id=object_id,
        )

        target_sem_id = sem_inst_info["target_sem_id"]
        target_sem_inst_count = sem_inst_info["target_sem_inst_count"]
        is_unique_sem = sem_inst_info["is_unique_sem"]
        is_multiple_sem = sem_inst_info["is_multiple_sem"]
        target_valid_for_sem_count = sem_inst_info["target_valid"]
        # -------------------------------------------------------
        # 5. 准备实例级数据 (CSR / BBox / Ref Labels)
        # -------------------------------------------------------
        # 声明变量
        inst2p_offsets_full = None
        inst2p_indices_full = None
        inst_bbox_full = None
        inst_sem_mode_full = None
        inst_ref_labels = None
        inst_masks = None
        inst_class_ids = None
        target_quality = 0.0
        if is_pred_mode:
            masks_v2 = np.zeros((0, xyz.shape[0]), dtype=np.uint8)
            classes_v2 = np.zeros((0,), dtype=np.int64)
            scores_v2 = np.zeros((0,), dtype=np.float32)

            masks_200 = np.zeros((0, xyz.shape[0]), dtype=np.uint8)
            classes_200 = np.zeros((0,), dtype=np.int64)
            scores_200 = np.zeros((0,), dtype=np.float32)

            if has_pred_v2:
                masks_v2, classes_v2, scores_v2 = self._load_pred_candidates_from_pt(
                    pred_filename_v2,
                    orig_idx_keep,
                    apply_whitelist=False,   # v2 侧先不套 200 白名单
                )

            if has_pred_200:
                masks_200, classes_200, scores_200 = self._load_pred_candidates_from_pt(
                    pred_filename_200,
                    orig_idx_keep,
                    apply_whitelist=True,    # 200 侧必须过白名单
                )

            final_masks, inst_sem_mode_full, inst_scores_full, num_drop_dup = \
                self._merge_v2_and_200_candidates(
                    masks_v2, classes_v2, scores_v2,
                    masks_200, classes_200, scores_200,
                )

            num_rescue_added = 0
            inst_masks_list_np = [final_masks[i].astype(np.uint8) for i in range(final_masks.shape[0])]
            inst_bbox_np = self.compute_bbox_from_mask(final_masks, xyz_middle)
            inst_bbox_list_np = [inst_bbox_np[i].astype(np.float32) for i in range(inst_bbox_np.shape[0])]

            inst_sem_mode_list_np = [int(x) for x in inst_sem_mode_full.tolist()]

            inst_masks_list_np, inst_bbox_list_np, inst_sem_mode_list_np, num_rescue_added = \
                self._append_rescue_candidates_pred(
                    xyz_middle,                 # 不要传 xyz_world，这里必须和 bbox 同坐标系
                    inst_masks_list_np,
                    inst_bbox_list_np,
                    inst_sem_mode_list_np,
                )
            
            if len(inst_masks_list_np) > 0:
                final_masks = np.stack(inst_masks_list_np, axis=0).astype(np.uint8)
                inst_bbox_np = np.stack(inst_bbox_list_np, axis=0).astype(np.float32)
                inst_sem_mode_full = np.array(inst_sem_mode_list_np, dtype=np.int64)
            else:
                final_masks = np.zeros((0, xyz.shape[0]), dtype=np.uint8)
                inst_bbox_np = np.zeros((0, 6), dtype=np.float32)
                inst_sem_mode_full = np.zeros((0,), dtype=np.int64)
                num_rescue_added = 0



            # 4. 生成 CSR (必须在 rescue 之后做)
            inst2p_offsets_full, inst2p_indices_full = self.mask_to_csr_overlapping(final_masks, xyz.shape[0])
            inst2p_offsets_full = torch.from_numpy(inst2p_offsets_full).long()
            inst2p_indices_full = torch.from_numpy(inst2p_indices_full).long()

            # 5. bbox 也用 rescue 后的
            inst_bbox_full = torch.from_numpy(inst_bbox_np).float()

            assert final_masks.shape[1] == xyz_middle.shape[0], \
                f"final_masks N={final_masks.shape[1]} != xyz_middle N={xyz_middle.shape[0]}"

            # 6. 计算 Ref Labels (目标匹配) —— 注意这里已经是 rescue 后的候选池
            num_pred = final_masks.shape[0]
            inst_ref_labels = torch.zeros(num_pred, dtype=torch.float32)

            gt_obj_mask = (instance_label == object_id)  # [N_new]

            if gt_obj_mask.sum() > 0 and num_pred > 0:
                inter = np.logical_and(final_masks, gt_obj_mask[None, :]).sum(axis=1)
                union = np.logical_or(final_masks, gt_obj_mask[None, :]).sum(axis=1)
                ious = inter / (union + 1e-6)

                best_idx = np.argmax(ious)
                best_iou = ious[best_idx]

                target_quality = float(best_iou)
                if best_iou > 0.25:
                    inst_ref_labels[best_idx] = 1.0

            # 7. 准备其他 Collater 需要的数据
            inst_masks = torch.from_numpy(final_masks).float()
            inst_class_ids = torch.from_numpy(inst_sem_mode_full).long()
            inst_sem_mode_full = torch.from_numpy(inst_sem_mode_full).long()

        else:
            # ===========================================
            # 分支 B: 原始 GT 数据 (读取 meta 文件)
            # ===========================================
            inst_filename = self.inst_map.get(scan_id)
            if inst_filename is None:
                raise FileNotFoundError(f"Missing instmeta file for {scan_id}")
            
            # 加载原始 meta
            #off_orig, idx_orig, inst_bbox_full, inst_sem_mode_full = self.load_instmeta(inst_filename)
            off_orig, idx_orig, _, inst_sem_mode_full = self.load_instmeta(inst_filename)
            # 1. Remap CSR (从全量映射到 Crop)
            # 这里的逻辑必须严谨，否则 index out of bound
            # 构建映射表: Orig_Idx -> New_Idx
            max_idx = int(idx_orig.max().item()) if idx_orig.numel() > 0 else -1
            orig2new = torch.full((max_idx + 1,), -1, dtype=torch.long)
            
            orig_idx_keep_t = torch.as_tensor(orig_idx_keep, dtype=torch.long)
            if orig_idx_keep_t.numel() > 0 and max_idx >= 0:
                valid_mask = (orig_idx_keep_t <= max_idx)
                orig2new[orig_idx_keep_t[valid_mask]] = torch.arange(valid_mask.sum(), dtype=torch.long)
            
            # 调用 Remap 工具函数
            inst2p_offsets_full, inst2p_indices_full = self.remap_csr_indices(off_orig, idx_orig, orig2new)
            
            # 2. 生成 GT 相关的 Ref Label
            num_inst = int(instance_label.max().item()) + 1

            inst_ref_labels = torch.zeros(num_inst, dtype=torch.float32)
            if 0 <= object_id < num_inst:
                inst_ref_labels[object_id] = 1.0

            # [关键修复] 现场用当前 crop+aug 后的 xyz_middle 重算 bbox，确保 bbox 和 coord_float 同坐标系
            inst_bbox_full = torch.zeros((num_inst, 6), dtype=torch.float32)
            for i in range(num_inst):
                pts_i = xyz_middle[instance_label == i]  # xyz_middle/instance_label 都是 numpy
                if pts_i.shape[0] > 0:
                    inst_bbox_full[i, :3] = torch.from_numpy(pts_i.min(axis=0)).float()
                    inst_bbox_full[i, 3:] = torch.from_numpy(pts_i.max(axis=0)).float()

            # （可选但强烈建议）把 inst_sem_mode_full 也裁到 num_inst，避免后面维度不一致
            if torch.is_tensor(inst_sem_mode_full) and inst_sem_mode_full.numel() >= num_inst:
                inst_sem_mode_full = inst_sem_mode_full[:num_inst]
                
            # 3. 生成 GT Masks (基于 Superpoint，用于辅助)
            # 注意：GT 模式下 inst_masks 通常是下采样后的，或者全量的，视你原逻辑而定
            # 这里保持你原来的做法：基于 Superpoint 生成
            num_sp = int(superpoint_aug.max().item()) + 1
            inst_masks = torch.zeros(num_inst, num_sp, dtype=torch.float32)
            inst_masks[instance_label, superpoint_aug] = 1.0
            target_quality = 1.0
            # 4. 生成 Class IDs
            inst_class_ids = torch.full((num_inst,), -1, dtype=torch.long)
            for i in range(num_inst):
                m = (instance_label == i)
                if m.any():
                    # 简单取众数
                    s = semantic_label[m]
                    valid_s = s[s>=0]
                    if valid_s.size > 0: 
                        # 下面这行保持不变，这里才转成 tensor
                        inst_class_ids[i] = torch.bincount(torch.from_numpy(valid_s).long()).argmax()

        # -------------------------------------------------------
        # 6. 计算辅助 Loss 需要的 GT Mask (不影响主要流程)
        # -------------------------------------------------------
        # [修正] get_ref_mask 需要 Tensor 输入，这里先转一下
        instance_label_tensor = torch.from_numpy(instance_label).long()
        superpoint_tensor = torch.from_numpy(superpoint_aug).long()
        
        # 这里的 gt_pmask 和 object_id 强绑定
        gt_pmask, gt_spmask = self.get_ref_mask(instance_label_tensor, superpoint_tensor, object_id)
        
        # 区域 Mask (Box 范围内)
        # 注意：xyz_middle 还是 numpy，这里要小心操作
        object_points = xyz_middle[gt_pmask.bool().numpy()] # 转回 numpy 做索引
        
        if object_points.shape[0] > 0:
            min_c = object_points.min(axis=0)
            max_c = object_points.max(axis=0)
            center = (min_c + max_c) / 2

            size = max_c - min_c
            scene_center = xyz_middle.mean(axis=0) 
            # 2. 将物体的绝对中心，转换为相对于当前场景中心的偏移量
            rel_center = center - scene_center 
            # 3. 组合成最终的 [相对 cx, 相对 cy, 相对 cz, w, h, l]
            #target_bbox = torch.from_numpy(np.concatenate([rel_center, size])).float()#改
            target_bbox = torch.from_numpy(np.concatenate([center, size])).float()

            '''
            dist = np.linalg.norm(max_c - min_c) / 2
            # 计算距离
            batch_dist = np.sqrt(((xyz_middle - center)**2).sum(axis=1))
            region_label = (batch_dist <= 2 * dist)
            '''
            # obj_mask_np: [N] bool
            obj_mask_np = gt_pmask.bool().numpy()

            # 用 AABB 自适应扩张生成 region（numpy bool）
            region_label = self._compute_region_aabb_adaptive(
                xyz_middle=xyz_middle,        # numpy [N,3]
                obj_mask=obj_mask_np,         # numpy [N]
                min_c=min_c,
                max_c=max_c,
                frac_min=0.10,
                frac_max=0.40,
                expand_init=0.5,
                max_iter=8,
                hard_max=0.85,
            )

            region_label_tensor = torch.from_numpy(region_label)
            gt_pmask_region, gt_spmask_region = self.get_ref_mask_region(region_label_tensor, superpoint_tensor)
            # region_label 是 numpy bool，转 tensor 传进去
            region_label_tensor = torch.from_numpy(region_label)
            gt_pmask_region, gt_spmask_region = self.get_ref_mask_region(region_label_tensor, superpoint_tensor)
        else:
            gt_pmask_region = torch.zeros_like(gt_pmask)
            gt_spmask_region = torch.zeros_like(gt_spmask)
            target_bbox = torch.zeros(6, dtype=torch.float32)


        return {
            'ann_ids': ann_id,
            'scan_ids': scan_id,
            'coord': torch.from_numpy(xyz).long(),
            'coord_float': torch.from_numpy(xyz_middle).float(),
            'xyz_world': torch.from_numpy(xyz_world).float(),
            'feat': torch.from_numpy(feat).float(),
            'superpoint': superpoint_tensor, # 已经在上面转成 tensor 了
            'object_id': object_id,
            'gt_pmask': gt_pmask,
            'gt_spmask': gt_spmask,
            'sp_ref_mask': None, # 占位
            'lang_tokens': None,
            'answers': answers,
            "text_input": question,
            "sp_filename": sp_filename,
            
            # 注意：原来的顺序里，region 在 filename 后面，obj_cat 在 region 后面
            'gt_pmask_region': gt_pmask_region,
            'gt_spmask_region': gt_spmask_region,
            'obj_cat': obj_cat,

            # ==== 核心实例信息 ====
            'inst_masks': inst_masks,           
            'inst_ref_labels': inst_ref_labels, 
            'inst_class_ids': inst_class_ids,   
            'inst2p_offsets_full': inst2p_offsets_full, 
            'inst2p_indices_full': inst2p_indices_full, 
            'inst_bbox_full': inst_bbox_full,           
            'inst_sem_mode_full': inst_sem_mode_full,
            'inst_filename': inst_filename if not is_pred_mode else "PRED_MODE",
            'target_quality': target_quality,
            'target_bbox': target_bbox,
            'meta_id': f"{scan_id}|{object_id}|{ann_id}",
            'cam_pos': torch.from_numpy(cam_pos).float(),
            'cam_lookat': torch.from_numpy(cam_lookat).float(),
            'cam_dir': torch.from_numpy(cam_dir).float(),
            'cam_valid': torch.tensor(int(cam_valid), dtype=torch.long),
            'view_dep': torch.tensor(int(view_dep), dtype=torch.long),
            'num_rescue_added': int(num_rescue_added) if is_pred_mode else 0,
            'target_sem_id': int(target_sem_id),
            'target_sem_inst_count': int(target_sem_inst_count),
            'is_unique_sem': int(is_unique_sem),
            'is_multiple_sem': int(is_multiple_sem),
            'target_valid_for_sem_count': int(target_valid_for_sem_count),
        }

    def collater(self, batch):
        ann_ids, scan_ids, coords, coords_float,xyz_worlds, feats, superpoints, object_ids, \
        gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, lang_words, \
        answerss, text_input_list, sp_filename_list, gt_pmasks_region, gt_spmasks_region, \
        obj_cats = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[]

        inst_masks_list = []
        inst_ref_labels_list = []
        inst_class_ids_list = []
        batch_offsets = [0]
        inst2p_offsets_full_list = []
        inst2p_indices_full_list = []
        inst_bbox_full_list = []
        inst_sem_mode_full_list = []
        inst_filename_list2 = []
        point_offsets = [0]  
        superpoint_bias = 0
        global_unique_class_ids = set()
        target_quality_list = []    
        target_bboxes_list = []
        n_answers = []
        meta_ids = []
        cam_pos_list, cam_lookat_list, cam_dir_list = [], [], []
        cam_valid_list, view_dep_list = [], []
        no_tgt_flags = []
        best_iou_list = []
        meta_id_list_for_stat = []

        rescue_added_list = []
        target_sem_id_list = []
        target_sem_inst_count_list = []
        is_unique_sem_list = []
        is_multiple_sem_list = []
        target_valid_for_sem_count_list = []

        # =================================

        for i, data in enumerate(batch):
            try:
                ann_id = data['ann_ids']
                scan_id = data['scan_ids']
                coord = data['coord']
                coord_float = data['coord_float']
                xyz_world = data['xyz_world']
                feat = data['feat']
                src_superpoint = data['superpoint']
                object_id = data['object_id']
                gt_pmask = data['gt_pmask']
                gt_spmask = data['gt_spmask']
                sp_ref_mask = data['sp_ref_mask']
                answers = data['answers']
                captions = data['text_input']
                sp_filename = data['sp_filename']
                gt_pmask_region = data['gt_pmask_region']
                gt_spmask_region = data['gt_spmask_region']
                obj_cat = data['obj_cat']

                inst_masks_b = data['inst_masks']
                inst_ref_labels_b = data['inst_ref_labels']
                inst_class_ids_b = data['inst_class_ids']
                inst2p_offsets_full = data['inst2p_offsets_full']
                inst2p_indices_full = data['inst2p_indices_full']
                inst_bbox_full = data['inst_bbox_full']
                inst_sem_mode_full = data['inst_sem_mode_full']
                inst_filename = data['inst_filename']
                t_qual = data['target_quality']
                target_bbox = data['target_bbox']
                # === 新增 camera ===
                meta_id = data.get('meta_id', f"{scan_id}|{object_id}|{ann_id}")
                cam_pos = data.get('cam_pos', torch.zeros(3))
                cam_lookat = data.get('cam_lookat', torch.zeros(3))
                cam_dir = data.get('cam_dir', torch.zeros(3))
                cam_valid = data.get('cam_valid', torch.tensor(0, dtype=torch.long))
                view_dep = data.get('view_dep', torch.tensor(0, dtype=torch.long))
                target_sem_id = int(data.get('target_sem_id', -1))
                target_sem_inst_count = int(data.get('target_sem_inst_count', -1))
                is_unique_sem = int(data.get('is_unique_sem', 0))
                is_multiple_sem = int(data.get('is_multiple_sem', 0))
                target_valid_for_sem_count = int(data.get('target_valid_for_sem_count', 0))
                # =======================================================
                # 统计：当前样本是否没有 tgt_idx
                # 规则：inst_ref_labels 全 0 => 没有 tgt_idx
                # =======================================================
                if torch.is_tensor(inst_ref_labels_b):
                    has_tgt_idx = bool((inst_ref_labels_b > 0.5).any().item())
                else:
                    inst_ref_labels_b_t = torch.as_tensor(inst_ref_labels_b)
                    has_tgt_idx = bool((inst_ref_labels_b_t > 0.5).any().item())

                no_tgt_flags.append(not has_tgt_idx)
                best_iou_list.append(float(t_qual))
                meta_id_list_for_stat.append(meta_id)
                rescue_added_list.append(int(data.get("num_rescue_added", 0)))
            except ValueError:
                print("[FATAL ERROR] collater 和 __getitem__ 返回值数量不匹配!")
                print("data.keys() =", data.keys())
                raise
            
            # 1) superpoint 偏移（点云那条原逻辑）
            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            batch_offsets.append(superpoint_bias)
            meta_ids.append(meta_id)
            cam_pos_list.append(cam_pos)
            cam_lookat_list.append(cam_lookat)
            cam_dir_list.append(cam_dir)
            cam_valid_list.append(cam_valid)
            view_dep_list.append(view_dep)

            ann_ids.append(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            point_offsets.append(point_offsets[-1] + int(coord_float.shape[0]))

            xyz_worlds.append(xyz_world)
            feats.append(feat)
            superpoints.append(superpoint)
            
            object_ids.append(object_id)
            obj_cats.append(obj_cat) 
            
            gt_pmasks.append(gt_pmask)
            gt_pmasks_region.append(gt_pmask_region)
            target_sem_id_list.append(target_sem_id)
            target_sem_inst_count_list.append(target_sem_inst_count)
            is_unique_sem_list.append(is_unique_sem)
            is_multiple_sem_list.append(is_multiple_sem)
            target_valid_for_sem_count_list.append(target_valid_for_sem_count)

            def _is_rank0():
                try:
                    import torch.distributed as dist
                    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                except Exception:
                    return True


            gt_spmasks.append(gt_spmask)
            gt_spmasks_region.append(gt_spmask_region)
            
            sp_ref_masks.append(sp_ref_mask)

            answerss.extend(answers)
            text_input_list.append(captions)
            sp_filename_list.append(sp_filename)
            n_answers.append(len(answers))
            inst_masks_list.append(inst_masks_b)
            inst_ref_labels_list.append(inst_ref_labels_b)
            inst_class_ids_list.append(inst_class_ids_b)

            inst2p_offsets_full_list.append(inst2p_offsets_full)
            inst2p_indices_full_list.append(inst2p_indices_full)
            inst_bbox_full_list.append(inst_bbox_full)
            inst_sem_mode_full_list.append(inst_sem_mode_full)
            inst_filename_list2.append(inst_filename)
            target_quality_list.append(t_qual)

            target_bboxes_list.append(target_bbox)

            # =======================================================
            # >>> [修改] 统计逻辑：使用 inst_sem_mode_full (全量数据)
            # =======================================================
            # inst_sem_mode_full: [I] 包含该场景文件中所有实例的语义类别，不受 crop 影响
            if inst_sem_mode_full is not None:
                # 1. 确保在 CPU 上并转为 numpy (如果它还不是的话)
                if torch.is_tensor(inst_sem_mode_full):
                    full_sem_ids = inst_sem_mode_full.cpu().numpy()
                else:
                    full_sem_ids = inst_sem_mode_full
                
                # 2. 过滤掉无效值 (通常 -1 或 255 代表 ignore)
                # 注意：ScanNet 训练数据通常是 0-19，但也可能有 -1
                valid_ids = full_sem_ids[full_sem_ids >= 0] 
                
                # 3. 更新到全局集合
                global_unique_class_ids.update(valid_ids.tolist())

        

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)
        point_offsets = torch.tensor(point_offsets, dtype=torch.int)
        coords = torch.cat(coords, 0)
        coords_float = torch.cat(coords_float, 0)
        xyz_world = torch.cat(xyz_worlds, 0)
        feats = torch.cat(feats, 0)
        superpoints = torch.cat(superpoints, 0).long()

        sp_centroids = torch_scatter.scatter_mean(coords_float, superpoints, dim=0)

        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)

        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), 128, None)
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)
        # =======================================================
        # [DEBUG] 统计当前 batch 里没有 tgt_idx 的比例
        # =======================================================
        num_batch = len(no_tgt_flags)
        num_no_tgt = int(sum(no_tgt_flags))
        ratio_no_tgt = num_no_tgt / max(num_batch, 1)

        # 同时看看这批 no_tgt 的 best_iou(target_quality) 平均有多低
        no_tgt_best_ious = [best_iou_list[k] for k, flag in enumerate(no_tgt_flags) if flag]
        mean_no_tgt_best_iou = sum(no_tgt_best_ious) / max(len(no_tgt_best_ious), 1)

        def _is_rank0():
            try:
                import torch.distributed as dist
                return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                return True

        if self.dump_dataset_debug_jsonl and _is_rank0():
            rank = 0
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    rank = dist.get_rank()
            except Exception:
                pass

            jsonl_path = self.dump_dataset_debug_path.format(rank=rank)
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

            record = {
                "batch_size": len(no_tgt_flags),
                "num_no_tgt": int(sum(no_tgt_flags)),
                "ratio_no_tgt": float(sum(no_tgt_flags) / max(len(no_tgt_flags), 1)),
                "mean_best_iou_no_tgt": float(
                    np.mean([best_iou_list[k] for k, f in enumerate(no_tgt_flags) if f]) if any(no_tgt_flags) else 0.0
                ),
                "mean_rescue_added": float(np.mean(rescue_added_list) if len(rescue_added_list) > 0 else 0.0),
                "cases": [
                    {
                        "meta_id": meta_id_list_for_stat[k],
                        "no_tgt": bool(no_tgt_flags[k]),
                        "best_iou": float(best_iou_list[k]),
                        "num_rescue_added": int(rescue_added_list[k]),
                    }
                    for k in range(len(no_tgt_flags))
                ]
            }

        out = {
            'ann_ids': ann_ids,
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'object_ids': object_ids,
            'object_cats': obj_cats,
            'gt_pmasks': gt_pmasks,
            'gt_pmasks_region': gt_pmasks_region,
            'sp_ref_masks': sp_ref_masks,
            "answer": answerss,
            "text_input": text_input_list,
            'lang_tokenss': None,
            'lang_masks': None,
            'n_answers': torch.LongTensor(n_answers),
            'sp_filenames': sp_filename_list,
            'coord_float': coords_float,      # 你如果之前没返回，可以顺手加上
            'xyz_world': xyz_world,           # <--- 新增：给后面可视化用
            # ==== 新增：实例级信息（保持 list，不拼成大 tensor） ====
            'inst_masks': inst_masks_list,            # list 长度 B，每个 [N_inst_b, N_sp_b]
            'inst_ref_labels': inst_ref_labels_list,  # list 长度 B，每个 [N_inst_b]
            'inst_class_ids': inst_class_ids_list,     # list 长度 B，每个 [N_inst_b]  <-- 新增
            'inst2p_offsets_full': inst2p_offsets_full_list,
            'inst2p_indices_full': inst2p_indices_full_list,
            'inst_bbox_full': inst_bbox_full_list,
            'inst_sem_mode_full': inst_sem_mode_full_list,
            'instmeta_filenames': inst_filename_list2,
            'point_offsets': point_offsets,
            'sp_centroids': sp_centroids,  
            'target_quality': torch.tensor(target_quality_list, dtype=torch.float32),
            'target_bboxes': torch.stack(target_bboxes_list, dim=0),
            'meta_ids': meta_ids,
            'cam_pos': torch.stack(cam_pos_list,dim=0).float(), 
            'cam_lookat': torch.stack(cam_lookat_list,dim=0).float(),
            'cam_dir': torch.stack(cam_dir_list,dim=0).float(),
            'cam_valid': torch.stack(cam_valid_list,dim=0).long(),
            'view_dep': torch.stack(view_dep_list,dim=0).long(),
            'target_sem_ids': torch.tensor(target_sem_id_list, dtype=torch.long),
            'target_sem_inst_counts': torch.tensor(target_sem_inst_count_list, dtype=torch.long),
            'is_unique_sem': torch.tensor(is_unique_sem_list, dtype=torch.long),
            'is_multiple_sem': torch.tensor(is_multiple_sem_list, dtype=torch.long),
            'target_valid_for_sem_count': torch.tensor(target_valid_for_sem_count_list, dtype=torch.long),
        }
        return out 
    
    def __len__(self):
        return len(self.annotation)

    def load_instmeta(self, filename: str):
        """
        读取离线 instance 文件，返回 (inst2p_offsets, inst2p_indices, inst_bbox, inst_sem_mode)
        兼容 dict 或 tuple/list
        """
        meta = torch.load(filename, map_location="cpu")

        if isinstance(meta, dict):
            inst2p_offsets = meta["inst2p_offsets"]
            inst2p_indices = meta["inst2p_indices"]
            inst_bbox = meta.get("inst_bbox", None)
            inst_sem_mode = meta.get("inst_sem_mode", None)
        elif isinstance(meta, (list, tuple)):
            # 你如果存成 tuple，就按你实际顺序写，这里给一个常见顺序：
            # (inst2p_offsets, inst2p_indices, inst_bbox, inst_sem_mode)
            inst2p_offsets, inst2p_indices = meta[0], meta[1]
            inst_bbox = meta[2] if len(meta) > 2 else None
            inst_sem_mode = meta[3] if len(meta) > 3 else None
        else:
            raise TypeError(f"Unrecognized instmeta format: {type(meta)} in {filename}")

        # 统一成 torch tensor
        if not torch.is_tensor(inst2p_offsets):
            inst2p_offsets = torch.as_tensor(inst2p_offsets, dtype=torch.long)
        else:
            inst2p_offsets = inst2p_offsets.long()

        if not torch.is_tensor(inst2p_indices):
            inst2p_indices = torch.as_tensor(inst2p_indices, dtype=torch.long)
        else:
            inst2p_indices = inst2p_indices.long()

        if inst_bbox is not None and (not torch.is_tensor(inst_bbox)):
            inst_bbox = torch.as_tensor(inst_bbox, dtype=torch.float32)
        elif torch.is_tensor(inst_bbox):
            inst_bbox = inst_bbox.float()

        if inst_sem_mode is not None and (not torch.is_tensor(inst_sem_mode)):
            inst_sem_mode = torch.as_tensor(inst_sem_mode, dtype=torch.long)
        elif torch.is_tensor(inst_sem_mode):
            inst_sem_mode = inst_sem_mode.long()

        return inst2p_offsets, inst2p_indices, inst_bbox, inst_sem_mode



    # =================================================================
    # [新增工具 1] 将重叠 Mask 转换为 CSR 格式 (允许重复索引)
    # =================================================================
    
    def mask_to_csr_overlapping(self, masks, num_points):
        """
        输入: masks [M, N] (Bool or 0/1, CPU numpy)
        输出: offsets [M+1], indices [Sum_of_valid_points]
        说明: 即使点被多个实例共享，scatter_mean 也能正常工作。
        """
        indices_list = []
        offsets = [0]
        
        M = masks.shape[0]
        for i in range(M):
            # 找到第 i 个实例包含的所有点索引
            # masks[i] 是一个布尔数组
            pts_idx = np.nonzero(masks[i])[0] 
            indices_list.append(pts_idx)
            offsets.append(offsets[-1] + len(pts_idx))
            
        if len(indices_list) > 0:
            indices = np.concatenate(indices_list)
        else:
            indices = np.array([], dtype=np.int64)
            
        offsets = np.array(offsets, dtype=np.int64)
        return offsets, indices

    def compute_bbox_from_mask(self, masks: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        """
        masks: [M, N] bool/0-1
        xyz:   [N, 3] (必须与 coord_float 同坐标系)
        return: [M, 6] = [cx, cy, cz, sx, sy, sz]
        """
        M = masks.shape[0]
        bbox = np.zeros((M, 6), dtype=np.float32)
        for i in range(M):
            pts_idx = np.nonzero(masks[i])[0]
            if pts_idx.size == 0:
                continue
            pts = xyz[pts_idx]
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            center = (mn + mx) / 2.0
            size = (mx - mn)
            bbox[i, :3] = center
            bbox[i, 3:] = size
        return bbox
    # =====================================================================
    # [工具 3] GT 模式下的 CSR 索引重映射 (Remap)
    # =====================================================================
    def remap_csr_indices(self, offsets_orig, indices_orig, orig2new_map):
        """
        将全量点云的 CSR 映射到 Crop 后的点云 CSR
        """
        # 转 CPU 计算
        offsets_orig = offsets_orig.cpu().numpy()
        indices_orig = indices_orig.cpu().numpy()
        orig2new_map = orig2new_map.cpu().numpy()
        
        I = offsets_orig.shape[0] - 1
        new_indices_list = []
        new_offsets = [0]
        
        for i in range(I):
            s, e = offsets_orig[i], offsets_orig[i+1]
            pts = indices_orig[s:e] # 原始点索引
            
            # 1. 过滤掉被 Crop 删除的点 (在 map 中为 -1 的)
            valid_pts = pts[(pts >= 0) & (pts < len(orig2new_map))]
            new_pts_idx = orig2new_map[valid_pts] # 获取新索引
            new_pts_idx = new_pts_idx[new_pts_idx >= 0]
            
            new_indices_list.append(new_pts_idx)
            new_offsets.append(new_offsets[-1] + len(new_pts_idx))
            
        if new_indices_list:
            new_indices = np.concatenate(new_indices_list)
        else:
            new_indices = np.array([], dtype=np.int64)
            
        return torch.from_numpy(np.array(new_offsets)).long(), torch.from_numpy(new_indices).long()
    

    def _is_view_dependent(self, desc: str) -> bool:
        if desc is None:
            return False
        return DIR_REGEX.search(desc) is not None

    def _compute_region_aabb_adaptive(
        self,
        xyz_middle: np.ndarray,   # [N,3] numpy
        obj_mask: np.ndarray,     # [N] bool numpy (gt_pmask)
        min_c: np.ndarray,        # [3]
        max_c: np.ndarray,        # [3]
        frac_min: float = 0.05,   # region 至少占 5% 点
        frac_max: float = 0.60,   # region 最多占 60% 点
        expand_init: float = 0.25,
        max_iter: int = 8,
        hard_max: float = 0.85,   # 仍然太大就退化成 obj
    ) -> np.ndarray:
        """
        用“扩张 bbox”的方式构造 region GT，并自适应控制 region 占比，避免 region_pos≈1 或 ≈0。
        """
        assert xyz_middle.ndim == 2 and xyz_middle.shape[1] == 3
        assert obj_mask.ndim == 1 and obj_mask.shape[0] == xyz_middle.shape[0]

        size = (max_c - min_c).astype(np.float32)
        size = np.maximum(size, 1e-6)

        expand = float(expand_init)
        region = None
        frac = None

        for _ in range(max_iter):
            lo = min_c - expand * size
            hi = max_c + expand * size
            region = np.all((xyz_middle >= lo) & (xyz_middle <= hi), axis=1)

            # 保证目标点一定为正（不会 miss_obj）
            region = np.logical_or(region, obj_mask)

            frac = float(region.mean())
            if frac > frac_max:
                expand *= 0.5
            elif frac < frac_min:
                expand *= 1.5
            else:
                break

        # 兜底：若目标本身太大导致 region 仍然接近全场景，则退化为 obj（至少给 LOC 提供区分度）
        if frac is not None and frac > hard_max:
            region = obj_mask.copy()

        return region
    
    def _mask_from_center_size(self, xyz_world_crop, center, size):
        half = size / 2.0
        lo = center - half
        hi = center + half
        inside = np.all((xyz_world_crop >= lo) & (xyz_world_crop <= hi), axis=1)
        return inside
    
    def _mask_to_indices_bbox(self, mask_bool, xyz_world_crop):
        idx = np.nonzero(mask_bool)[0].astype(np.int64)
        if len(idx) == 0:
            return None, None
        pts = xyz_world_crop[idx]
        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)
        center = (pmin + pmax) / 2.0
        size = np.clip(pmax - pmin, 1e-3, None)
        bbox = np.concatenate([center, size], axis=0).astype(np.float32)
        return idx, bbox
    
    def _append_rescue_candidates_pred(
        self,
        xyz_world_crop,
        inst_masks_list,
        inst_bbox_list,
        inst_sem_mode_list,
    ):
        if (not self.enable_rescue_candidates) or len(inst_masks_list) == 0:
            return inst_masks_list, inst_bbox_list, inst_sem_mode_list, 0

        N = xyz_world_crop.shape[0]
        new_masks = []
        new_bboxes = []
        new_sem = []
        num_added = 0

        # centers for nearest-neighbor union
        centers = np.stack([b[:3] for b in inst_bbox_list], axis=0) if len(inst_bbox_list) > 0 else np.zeros((0, 3), dtype=np.float32)

        for i, bbox_i in enumerate(inst_bbox_list):
            if num_added >= len(inst_bbox_list) * self.rescue_max_new_per_inst:
                break

            center_i = bbox_i[:3].copy()
            size_i = bbox_i[3:6].copy()

            # ------------------------
            # A. dilate rescue
            # ------------------------
            if self.rescue_mode in ["dilate", "dilate_union"]:
                size_new = size_i * self.rescue_dilate_scale
                mask_new = self._mask_from_center_size(xyz_world_crop, center_i, size_new)

                if mask_new.sum() >= self.rescue_min_points and mask_new.sum() <= int(N * self.rescue_max_points_ratio):
                    idx_new, bbox_new = self._mask_to_indices_bbox(mask_new, xyz_world_crop)
                    if idx_new is not None:
                        new_masks.append(mask_new.astype(np.uint8))
                        new_bboxes.append(bbox_new)
                        new_sem.append(inst_sem_mode_list[i] if len(inst_sem_mode_list) > i else -1)
                        num_added += 1

            # ------------------------
            # B. nearest union rescue
            # ------------------------
            if self.rescue_mode == "dilate_union" and len(inst_bbox_list) > 1:
                dist = np.linalg.norm(centers - center_i[None, :], axis=1)
                dist[i] = 1e9
                nn_ids = np.argsort(dist)[:self.rescue_neighbor_topk]

                for j in nn_ids:
                    bbox_j = inst_bbox_list[int(j)]
                    c_j, s_j = bbox_j[:3], bbox_j[3:6]

                    lo = np.minimum(center_i - size_i / 2.0, c_j - s_j / 2.0)
                    hi = np.maximum(center_i + size_i / 2.0, c_j + s_j / 2.0)
                    center_u = (lo + hi) / 2.0
                    size_u = np.clip(hi - lo, 1e-3, None)

                    mask_u = self._mask_from_center_size(xyz_world_crop, center_u, size_u)
                    if mask_u.sum() >= self.rescue_min_points and mask_u.sum() <= int(N * self.rescue_max_points_ratio):
                        idx_u, bbox_u = self._mask_to_indices_bbox(mask_u, xyz_world_crop)
                        if idx_u is not None:
                            new_masks.append(mask_u.astype(np.uint8))
                            new_bboxes.append(bbox_u)
                            new_sem.append(inst_sem_mode_list[i] if len(inst_sem_mode_list) > i else -1)
                            num_added += 1
                            break

        # append
        for m, b, s in zip(new_masks, new_bboxes, new_sem):
            inst_masks_list.append(m)
            inst_bbox_list.append(b)
            inst_sem_mode_list.append(s)

        return inst_masks_list, inst_bbox_list, inst_sem_mode_list, num_added
    
    def _mask_iou(self, a: np.ndarray, b: np.ndarray) -> float:
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union + 1e-6)
    
    def _merge_v2_and_200_candidates(
        self,
        masks_v2, classes_v2, scores_v2,
        masks_200, classes_200, scores_200,
    ):
        merged_masks = [m.astype(np.uint8) for m in masks_v2]
        merged_classes = [int(c) for c in classes_v2.tolist()]
        merged_scores = [float(s) for s in scores_v2.tolist()]

        num_drop_dup = 0

        for i in range(masks_200.shape[0]):
            m_new = masks_200[i]
            is_dup = False

            for m_old in merged_masks:
                iou = self._mask_iou(m_new, m_old)
                if iou >= self.pred_merge_dedup_iou:
                    is_dup = True
                    num_drop_dup += 1
                    break

            if not is_dup:
                merged_masks.append(m_new.astype(np.uint8))
                merged_classes.append(int(classes_200[i]))
                merged_scores.append(float(scores_200[i]))

        if len(merged_masks) == 0:
            N = masks_v2.shape[1] if masks_v2.ndim == 2 else (
                masks_200.shape[1] if masks_200.ndim == 2 else 0
            )
            return (
                np.zeros((0, N), dtype=np.uint8),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                num_drop_dup,
            )

        return (
            np.stack(merged_masks, axis=0).astype(np.uint8),
            np.array(merged_classes, dtype=np.int64),
            np.array(merged_scores, dtype=np.float32),
            num_drop_dup,
        )
    
    def _load_pred_candidates_from_pt(
        self,
        pred_filename,
        orig_idx_keep,
        apply_whitelist=False,
    ):
        pred_data = torch.load(pred_filename, map_location="cpu")

        pt_masks = pred_data["pred_masks"]
        if torch.is_tensor(pt_masks):
            pt_masks = pt_masks.cpu().numpy()
        else:
            pt_masks = np.asarray(pt_masks)

        pt_classes = pred_data["pred_classes"]
        if torch.is_tensor(pt_classes):
            pt_classes = pt_classes.cpu().numpy()
        else:
            pt_classes = np.asarray(pt_classes)

        if "pred_scores" in pred_data:
            pt_scores = pred_data["pred_scores"]
            if torch.is_tensor(pt_scores):
                pt_scores = pt_scores.cpu().numpy()
            else:
                pt_scores = np.asarray(pt_scores)
        else:
            pt_scores = np.ones((pt_classes.shape[0],), dtype=np.float32)

        # crop 到当前样本
        cropped_masks = pt_masks[:, orig_idx_keep]

        # 去掉 crop 后空实例
        valid_inst = (cropped_masks.sum(axis=1) > 0)
        masks = cropped_masks[valid_inst]
        classes = pt_classes[valid_inst]
        scores = pt_scores[valid_inst]

        # 只对 200 分支做白名单
        if apply_whitelist:
            self._ensure_pred_whitelist_ready(classes)
            keep = np.isin(classes.astype(np.int64),
                        np.array(sorted(self.pred_whitelist_ids), dtype=np.int64))
            masks = masks[keep]
            classes = classes[keep]
            scores = scores[keep]

        return masks.astype(np.uint8), classes.astype(np.int64), scores.astype(np.float32)

    
    def _load_pred_whitelist_names(self, txt_path):
        if txt_path is None or (not os.path.exists(txt_path)):
            raise FileNotFoundError(f"[WHITELIST] missing txt: {txt_path}")

        names = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                names.append(s)

        if len(names) == 0:
            raise ValueError(f"[WHITELIST] empty txt: {txt_path}")
        return names


    def _ensure_pred_whitelist_ready(self, class_ids_np: np.ndarray):
        if self.pred_whitelist_ids is not None:
            return

        self.pred_id_space = self._detect_pred_id_space(class_ids_np)
        self.pred_label2id = self._build_scannet200_label2id(self.pred_id_space)

        self.pred_whitelist_ids, resolved, missing = self._map_whitelist_names_to_ids(
            self.pred_whitelist_names,
            self.pred_label2id,
            strict=self.strict_pred_whitelist,
        )

        
    
    def _map_whitelist_names_to_ids(self, names, label2id, strict=False):
        ids = []
        resolved = []
        missing = []

        for raw_name in names:
            key = self._normalize_label_name(raw_name)

            # 先直接查
            if key in label2id:
                ids.append(label2id[key])
                resolved.append((raw_name, key))
                continue

            # 再查 alias
            alias = self.pred_name_alias.get(raw_name, None)
            if alias is None:
                alias = self.pred_name_alias.get(key, None)

            if alias is not None:
                alias_key = self._normalize_label_name(alias)
                if alias_key in label2id:
                    ids.append(label2id[alias_key])
                    resolved.append((raw_name, alias_key))
                    continue

            missing.append(raw_name)

        ids = sorted(set(ids))

        # 只有在严格模式下才报错
        if strict and len(missing) > 0:
            raise KeyError(f"[WHITELIST] names not found in CLASS_LABELS_200: {missing}")

        # 真正危险的是一个都映射不出来
        if len(ids) == 0:
            raise ValueError(
                f"[WHITELIST] no valid whitelist ids mapped. missing={missing}"
            )

        return set(ids), resolved, missing
    
    def get_semantic_instance_info(self, semantic_label, instance_label, object_id):
        """
        判断当前 target object 所属语义类，在当前 scene/crop 中有几个 GT instance。

        Args:
            semantic_label: np.ndarray [N], 每个点的语义标签
            instance_label: np.ndarray [N], 每个点的 instance id
            object_id: int, target instance id

        Returns:
            dict:
                target_sem_id: target 的语义类别 id
                target_sem_inst_count: 该语义类别下 GT instance 数量
                is_unique_sem: 是否该语义类只有一个 instance
                is_multiple_sem: 是否该语义类有多个 instance
                target_valid: target mask 是否存在
        """
        sem = np.asarray(semantic_label).astype(np.int64)
        inst = np.asarray(instance_label).astype(np.int64)
        obj_id = int(object_id)

        target_mask = (inst == obj_id)
        if target_mask.sum() == 0:
            return {
                "target_sem_id": -1,
                "target_sem_inst_count": -1,
                "is_unique_sem": 0,
                "is_multiple_sem": 0,
                "target_valid": 0,
            }

        target_sem_vals = sem[target_mask]
        target_sem_vals = target_sem_vals[target_sem_vals >= 0]

        if target_sem_vals.size == 0:
            return {
                "target_sem_id": -1,
                "target_sem_inst_count": -1,
                "is_unique_sem": 0,
                "is_multiple_sem": 0,
                "target_valid": 1,
            }

        vals, cnts = np.unique(target_sem_vals, return_counts=True)
        target_sem_id = int(vals[np.argmax(cnts)])

        # 统计所有 GT instance 的主语义类别
        same_sem_inst_ids = []
        valid_inst_ids = np.unique(inst)
        valid_inst_ids = valid_inst_ids[valid_inst_ids >= 0]

        for iid in valid_inst_ids.tolist():
            m = (inst == int(iid))
            if m.sum() == 0:
                continue

            sem_i = sem[m]
            sem_i = sem_i[sem_i >= 0]
            if sem_i.size == 0:
                continue

            v_i, c_i = np.unique(sem_i, return_counts=True)
            inst_sem_id = int(v_i[np.argmax(c_i)])

            if inst_sem_id == target_sem_id:
                same_sem_inst_ids.append(int(iid))

        target_sem_inst_count = int(len(set(same_sem_inst_ids)))

        return {
            "target_sem_id": int(target_sem_id),
            "target_sem_inst_count": target_sem_inst_count,
            "is_unique_sem": int(target_sem_inst_count == 1),
            "is_multiple_sem": int(target_sem_inst_count > 1),
            "target_valid": 1,
        }

        
    
QUESTION_LIST = [
    "Use local spatial cues to disambiguate the target object, then output the local cue and segmentation mask: {description}",
    "Resolve the target using relative position and nearby reference objects, then output the local cue and segmentation mask: {description}",
]

QUESTION_LIST_VIEW = [
    "Use directional relations such as left, right, front, or behind to disambiguate the target object, then output the local cue and segmentation mask: {description}",
    "Focus on viewpoint-sensitive spatial relations and nearby reference objects to identify the target, then output the local cue and segmentation mask: {description}",
]

ANSWER_LIST = [
    "The local cue is [LOC], and the segmentation mask is [SEG]."
]

DIR_PATTERNS = [
    r"\bleft\b", r"\bright\b",
    r"\bfront\b", r"\bbehind\b", r"\bback\b",
    r"\bfacing\b", r"\bin front of\b", r"\bin the back\b",
    r"\bon the left\b", r"\bon the right\b",
    r"\bleftmost\b", r"\brightmost\b",
]
 
DIR_REGEX = re.compile("|".join(DIR_PATTERNS), flags=re.IGNORECASE)
