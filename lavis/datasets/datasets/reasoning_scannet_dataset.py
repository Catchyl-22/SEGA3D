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

class SEGA3DReasoningDataset(BaseDataset):
    def __init__(self, text_processor, pts_root, ann_paths, candidate_cfg=None, split=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(text_processor, pts_root, ann_paths)
        candidate_cfg = dict(candidate_cfg or {})
        self.scene_ids = {}
        self.source_mode = "scannet"
        n = 0
        self.use_xyz = True
        self.mode = 4
        new_annotation = []
        for ann in self.annotation:
            scene_id = ann.get("scene_id", ann.get("scene_name"))
            if scene_id is None:
                continue

            ann_type = self._infer_ann_type(ann, scene_id)

            if ann_type not in ["scannet", "mp3d"]:
                continue

            if self.source_mode != "all" and ann_type != self.source_mode:
                continue

            if scene_id not in self.scene_ids:
                self.scene_ids[scene_id] = n
                n += 1

            ann["scene_id"] = scene_id
            ann["type"] = ann_type
            new_annotation.append(ann)


        if split is None:
            split = "train" if "train" in ann_paths[0] else "val"
        self.training = split == "train"
        self.prefix = "train" if self.training else "val"

        self.with_label = True
        self.annotation = new_annotation

        # -------------------------------
        # ScanNet 文件
        # -------------------------------
        self.scannet_sp_filenames = sorted(
            glob.glob(osp.join(self.pts_root, "scannetv2", self.prefix, "*_reason.pth"))
        )

        # -------------------------------
        # Matterport / MP3D 文件
        # 参考老 dataset_ol.py
        # -------------------------------
        self.mp3d_root = osp.join(self.pts_root, "matterport")
        self.mp3d_sp_filenames = []
        split_name = "train" if self.training else "val"
        mp3d_scene_txt = osp.join(self.mp3d_root, f"scenes_{split_name}.txt")
        mp3d_data_root = osp.join(self.mp3d_root, "mp3d_data")

        if osp.exists(mp3d_scene_txt) and osp.isdir(mp3d_data_root):
            with open(mp3d_scene_txt, "r") as f:
                mp3d_scene_list = [x.strip() for x in f.readlines() if x.strip()]
            for scene in mp3d_scene_list:
                self.mp3d_sp_filenames += glob.glob(osp.join(mp3d_data_root, scene + "*"))

        self.sp_filenames = sorted(self.scannet_sp_filenames + self.mp3d_sp_filenames)
        assert len(self.sp_filenames) > 0, "Empty dataset."

        # 统一 scene_id -> file path
        self.sp_map = {}
        for p in self.sp_filenames:
            stem = osp.splitext(osp.basename(p))[0]
            if stem.endswith("_reason"):
                sid = stem.replace("_reason", "")
            else:
                sid = stem
            self.sp_map[sid] = p

        self.short_question_list = QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.with_elastic = True
        self.aug = True     


        # ===== ScanNet instance meta =====
        self.inst_map = {}
        self.inst_dir = osp.join(self.pts_root, "scannetv2", f"{self.prefix}_instance")
        if osp.isdir(self.inst_dir):
            print(f"[INFO] ScanNet instance meta dir: {self.inst_dir}")
            self.inst_filenames = sorted(glob.glob(osp.join(self.inst_dir, "*.*")))
            for p in self.inst_filenames:
                base = osp.basename(p)
                m = re.search(r"(scene\d{4}_\d{2})", base)
                if m:
                    sid = m.group(1)
                else:
                    sid = base.replace("_instmeta.pt", "").replace("_instance.pt", "").replace("_instance.pth", "")
                self.inst_map[sid] = p
        else:
            print(f"[WARN] ScanNet instance meta dir not found: {self.inst_dir}")
            self.inst_filenames = []

        # ============================================
        self.gt_prob = 0.0 if self.training else 0.0
        self.use_prediction = True
        self.pred_root = candidate_cfg.get("scannet_semantic", osp.join(pts_root, "candidates", "reasoning", "scannet_semantic"))

        self.copy_paste_prob = 0.5 if self.training else 0.0
        self.tem = 0
        self.tem2 = 0 

        # mp3d room type（可选）
        self.mp3d_room_type = {}
        mp3d_room_type_json = candidate_cfg.get("room_types", osp.join(pts_root, "matterport", "mp3d_room_type.json"))
        if osp.exists(mp3d_room_type_json):
            with open(mp3d_room_type_json, "r") as f:
                self.mp3d_room_type = json.load(f)

        # =========================================================
        # MP3D predicted / exported instance root (预留位置)
        # =========================================================
        self.mp3d_inst_pred_root = candidate_cfg.get("matterport_instances", osp.join(pts_root, "candidates", "reasoning", "matterport_instances"))

        self.mp3d_inst_map = {}
        if osp.isdir(self.mp3d_inst_pred_root):
            mp3d_inst_files = sorted(glob.glob(osp.join(self.mp3d_inst_pred_root, "*.pt")))
            for p in mp3d_inst_files:
                sid = osp.splitext(osp.basename(p))[0]   # e.g. 17DRP5sb8fy_region0
                self.mp3d_inst_map[sid] = p
            print(f"[INFO] MP3D instance pred root: {self.mp3d_inst_pred_root}")
            print(f"[INFO] MP3D instance files: {len(self.mp3d_inst_map)}")
        else:
            print(f"[WARN] MP3D instance pred root not found: {self.mp3d_inst_pred_root}")



    def parse_mp3d_pred_inst(self, meta, xyz_middle=None, orig_idx_keep=None):
        """
        目标：
        返回一个统一 dict，后面 __getitem__ 里直接用。

        当前版本只做预留和兼容：
        1) 如果 meta 已经含有我们需要的字段，就直接取
        2) 如果没有，就先原样返回 raw，等你给样本再补
        """
        out = {
            "raw_meta": meta,
            "inst_masks_full": None,
            "inst_bbox_full": None,
            "inst_sem_mode_full": None,
            "inst_scores_full": None,
            "inst2p_offsets_full": None,
            "inst2p_indices_full": None,
        }

        if isinstance(meta, dict):
            # 这些 key 只是“尝试读取”，不保证一定存在
            for k in ["pred_masks", "inst_masks", "masks"]:
                if k in meta:
                    out["inst_masks_full"] = meta[k]
                    break

            for k in ["pred_boxes", "inst_bbox", "bboxes", "boxes"]:
                if k in meta:
                    out["inst_bbox_full"] = meta[k]
                    break

            for k in ["pred_classes", "inst_sem_mode", "classes", "labels"]:
                if k in meta:
                    out["inst_sem_mode_full"] = meta[k]
                    break

            for k in ["pred_scores", "scores", "conf"]:
                if k in meta:
                    out["inst_scores_full"] = meta[k]
                    break

            for k in ["inst2p_offsets", "offsets"]:
                if k in meta:
                    out["inst2p_offsets_full"] = meta[k]
                    break

            for k in ["inst2p_indices", "indices"]:
                if k in meta:
                    out["inst2p_indices_full"] = meta[k]
                    break

        elif isinstance(meta, (list, tuple)):
            # 先不假设顺序，原样交回去，避免误解析
            out["raw_meta"] = meta

        return out
    
    def load_mp3d_pred_inst(self, filename: str):
        """
        先做 raw loader，不预设内部格式。
        等你给一个样本 payload 后，再精确解析成
        inst2p_offsets / inst2p_indices / inst_bbox / inst_sem_mode / inst_masks。
        """
        meta = torch.load(filename, map_location="cpu")
        return meta
    
    def get_sp_filenames(self):
        filenames = glob.glob(osp.join(self.pts_root, 'scannetv2', self.prefix, '*' + '_reason.pth'))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        return filenames


    def _infer_ann_type(self, ann, scene_id):
        ann_type = str(ann.get("type", "")).lower()
        if ann_type in ["scannet", "mp3d"]:
            return ann_type
        sid = str(scene_id)
        if sid.startswith("scene"):
            return "scannet"
        if "_region" in sid:
            return "mp3d"
        return "scannet"

    def _extract_room_type(self, question_raw, scan_id=None, ann_type=None):
        q = str(question_raw).lower()
        for room_type in ROOM_TYPE_LIST:
            if room_type in q:
                return room_type

        if ann_type == "mp3d" and hasattr(self, "mp3d_room_type"):
            if scan_id in self.mp3d_room_type:
                return self.mp3d_room_type[scan_id]

        return "the room"
      
    def load(self, filename):

        
        data = torch.load(filename)

        # 兼容 tuple / list / dict 三种保存方式
        if isinstance(data, (list, tuple)):
            if len(data) == 5:
                xyz, rgb, superpoint, sem_labels, instance_labels = data
            elif len(data) == 4:
                # mp3d 常见：第四项就是 point-level instance/object labels
                xyz, rgb, superpoint, instance_labels = data
                sem_labels = np.full(xyz.shape[0], -1, dtype=np.int64)
            elif len(data) == 3:
                xyz, rgb, superpoint = data
                sem_labels = np.full(xyz.shape[0], -1, dtype=np.int64)
                instance_labels = np.full(xyz.shape[0], -1, dtype=np.int64)
            else:
                raise ValueError(f"Unsupported tuple length: {len(data)} in {filename}")
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
    


    def get_cropped_inst_label(
        self,
        instance_label: np.ndarray,
        valid_idxs: np.ndarray,
        return_mapping: bool = False,
    ):
        instance_label = instance_label[valid_idxs].astype(np.int64).copy()

        valid_inst_ids = np.unique(instance_label)
        valid_inst_ids = valid_inst_ids[valid_inst_ids >= 0]

        new_instance_label = np.full_like(instance_label, fill_value=-1)
        old2new = {}

        for new_id, old_id in enumerate(valid_inst_ids.tolist()):
            new_instance_label[instance_label == old_id] = new_id
            old2new[int(old_id)] = int(new_id)

        if return_mapping:
            return new_instance_label, old2new
        return new_instance_label
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
        scan_id = data.get("scene_id", data.get("scene_name"))
        ann_type = self._infer_ann_type(data, scan_id)
        object_id = int(data["object_id"])
        ann_id = int(data.get("ann_id", index))
        instance_id = data.get("instance_id", None)

        question_raw = data["question"]
        question_text = self.text_processor(question_raw)

        #room_target = self._extract_room_type(question_raw)   # 从 question 里抽 room type
        question = random.choice(self.short_question_list).format(description=question_text)

        obj_cat = data.get("object_name", "unknown")
        camera = None
        
        # 答案生成 (完全保留原逻辑)
        room_target = self._extract_room_type(question_raw, scan_id=scan_id, ann_type=ann_type)
        answers = [random.choice(self.answer_list).format(location=room_target)]

        # -------------------------------------------------------
        # 2. 加载原始点云数据 (作为基准)
        # -------------------------------------------------------
        sp_filename = self.sp_map.get(scan_id)
        if sp_filename is None:
             sp_filename = os.path.join(self.pts_root, "scannetv2", self.prefix, f"{scan_id}_reason.pth")
             print(f"no no no no :{sp_filename}")
             if not os.path.exists(sp_filename):
                 raise FileNotFoundError(f"Missing refer file for {scan_id}")

        mp3d_inst_filename = None
        if ann_type == "mp3d":
            mp3d_inst_filename = self.mp3d_inst_map.get(scan_id, None)
        # data_loaded: (xyz, rgb, superpoint, sem_labels, instance_labels)
        # 注意：这里的 xyz 是原始未裁剪的，RGB 是原始颜色
        data_loaded = self.load(sp_filename)
        # 保留一份“原始逐点 instance id”，后面只给 mp3d 做 object_id_local 映射
        raw_instance_labels_full = data_loaded[4].copy() if data_loaded[4] is not None else None
        
        # -------------------------------------------------------
        # 3. 检查模式：是使用 Pointcept 预测数据，还是使用 GT？
        # -------------------------------------------------------
        use_prediction_global = getattr(self, "use_prediction", False)
        pred_filename = os.path.join(self.pred_root, f"{scan_id}_pred.npy") if use_prediction_global else None
        #is_pred_mode = (use_prediction and os.path.exists(pred_filename))
        # 默认模式
        mode_switch = "gt"

        if use_prediction_global and os.path.exists(pred_filename):
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

        # 设置标志位，供后面逻辑使用
        is_pred_mode = (ann_type == "scannet") and (mode_switch == "pred")

            


        # -------------------------------------------------------
        # 4. 执行 Transform (核心步骤：裁剪 + 增强)
        # -------------------------------------------------------
        # 无论哪种模式，都必须经过这一步来获得 xyz_middle (归一化坐标) 和 mask

        if self.training:
            data_trans = self.transform_train(*data_loaded, desc=data["question"], camera=camera)
            (xyz, xyz_middle, feat, superpoint_aug, semantic_label, instance_label, uniq_sp_ids_full, xyz_world, orig_idx_keep,
            cam_pos, cam_lookat, cam_dir, cam_valid, view_dep) = data_trans
        else:
            data_trans = self.transform_test(*data_loaded, camera=camera, desc=data["question"])
            (xyz, xyz_middle, feat, superpoint_aug, semantic_label, instance_label, uniq_sp_ids_full, xyz_world, orig_idx_keep,
            cam_pos, cam_lookat, cam_dir, cam_valid, view_dep) = data_trans
        # -------------------------------------------------------
        # 4.0 ScanNet: 直接构建 semantic GT
        # -------------------------------------------------------
        sem_gt_mask_np = None
        gt_pmask = None
        gt_spmask = None

        if ann_type == "scannet":
            sem_gt_mask_np, gt_pmask, gt_spmask = self._build_scannet_sem_gt(
                semantic_label=semantic_label,
                superpoint_aug=superpoint_aug,
                object_id=object_id,
            )
        
        # -------------------------------------------------------
        # 4.1 构造 raw GT mask，并为不同数据源确定 object_id_local
        # -------------------------------------------------------
        object_id_local = int(object_id)
        best_iou_local = -1.0

        raw_inst_crop = None
        raw_gt_mask_np = None

        if raw_instance_labels_full is not None:
            raw_inst_crop = raw_instance_labels_full[orig_idx_keep].astype(np.int64)
            raw_gt_mask_np = (raw_inst_crop == int(object_id))   # [N_new] bool

        if ann_type == "mp3d":
            # MP3D 继续用 old -> new 映射即可
            if raw_inst_crop is not None:
                old2new_inst = {}
                valid_old_ids = np.unique(raw_inst_crop)
                valid_old_ids = valid_old_ids[valid_old_ids >= 0]

                for old_id in valid_old_ids.tolist():
                    new_ids = np.unique(instance_label[raw_inst_crop == old_id])
                    new_ids = new_ids[new_ids >= 0]
                    if new_ids.size > 0:
                        old2new_inst[int(old_id)] = int(new_ids[0])

                object_id_local = old2new_inst.get(int(object_id), -1)

        else:
            object_id_local = -1
            raw_gt_mask_np = None

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
  
        if ann_type == "mp3d":
            # ===========================================
            # 分支 A: MP3D predicted instances
            # ===========================================
            if mp3d_inst_filename is None:
                raise FileNotFoundError(f"Missing MP3D inst file for {scan_id}")

            pred_data = torch.load(mp3d_inst_filename, map_location="cpu")

            pt_masks = pred_data["pred_masks"]
            pt_scores = pred_data["pred_scores"]
            pt_classes = pred_data["pred_classes"]

            if torch.is_tensor(pt_masks):
                pt_masks = pt_masks.cpu().numpy()
            if torch.is_tensor(pt_scores):
                pt_scores = pt_scores.cpu().numpy()
            if torch.is_tensor(pt_classes):
                pt_classes = pt_classes.cpu().numpy()

            pt_masks = (pt_masks > 0).astype(np.bool_)   # [M, N_orig]

            # crop 到当前增强/裁剪后的点
            cropped_masks = pt_masks[:, orig_idx_keep]   # [M, N_new]

            # 过滤 crop 后为空的实例
            valid_inst = (cropped_masks.sum(axis=1) > 0)
            final_masks = cropped_masks[valid_inst]
            final_scores = pt_scores[valid_inst]
            final_classes = pt_classes[valid_inst]

            num_pred = final_masks.shape[0]

            inst2p_offsets_full, inst2p_indices_full = self.mask_to_csr_overlapping(
                final_masks, xyz.shape[0]
            )
            inst2p_offsets_full = torch.from_numpy(inst2p_offsets_full).long()
            inst2p_indices_full = torch.from_numpy(inst2p_indices_full).long()

            inst_bbox_full = torch.from_numpy(
                self.compute_bbox_from_mask(final_masks, xyz_middle)
            ).float()

            inst_sem_mode_full = torch.from_numpy(final_classes).long()
            inst_masks = torch.from_numpy(final_masks).float()
            inst_class_ids = torch.from_numpy(final_classes).long()

            inst_ref_labels = torch.zeros(num_pred, dtype=torch.float32)
            target_quality = 0.0

            gt_obj_mask = (instance_label == object_id_local)

            if object_id_local >= 0 and gt_obj_mask.sum() > 0 and num_pred > 0:
                inter = np.logical_and(final_masks, gt_obj_mask[None, :]).sum(axis=1)
                union = np.logical_or(final_masks, gt_obj_mask[None, :]).sum(axis=1)
                ious = inter / (union + 1e-6)

                best_idx = np.argmax(ious)
                best_iou = float(ious[best_idx])
                target_quality = best_iou

                if best_iou > 0.25:
                    inst_ref_labels[best_idx] = 1.0

            inst_filename = mp3d_inst_filename

        elif is_pred_mode:
            # ===========================================
            # 分支 B: ScanNet semantic prediction candidates
            # 每个出现过的语义类 -> 一个 candidate mask
            # ===========================================
            pred_sem = np.load(pred_filename)   # [N_orig], int

            # crop 到当前增强/裁剪后的点
            pred_sem_crop = pred_sem[orig_idx_keep].astype(np.int64)   # [N_new]

            # 过滤 ignore
            valid_cls = np.unique(pred_sem_crop)
            valid_cls = valid_cls[valid_cls >= 0]

            # 如果一个场景里没有有效类，给空候选
            if valid_cls.size == 0:
                final_masks = np.zeros((0, xyz.shape[0]), dtype=np.bool_)
                inst_sem_mode_full = np.zeros((0,), dtype=np.int64)
            else:
                final_masks = []
                final_classes = []

                for cls in valid_cls.tolist():
                    m = (pred_sem_crop == cls)
                    # 过滤太小的 semantic 区域，避免全是噪声
                    if m.sum() < 20:
                        continue
                    final_masks.append(m)
                    final_classes.append(cls)

                if len(final_masks) == 0:
                    final_masks = np.zeros((0, xyz.shape[0]), dtype=np.bool_)
                    inst_sem_mode_full = np.zeros((0,), dtype=np.int64)
                else:
                    final_masks = np.stack(final_masks, axis=0).astype(np.bool_)   # [M, N_new]
                    inst_sem_mode_full = np.array(final_classes, dtype=np.int64)   # [M]

            inst2p_offsets_full, inst2p_indices_full = self.mask_to_csr_overlapping(
                final_masks, xyz.shape[0]
            )
            inst2p_offsets_full = torch.from_numpy(inst2p_offsets_full).long()
            inst2p_indices_full = torch.from_numpy(inst2p_indices_full).long()

            inst_bbox_full = torch.from_numpy(
                self.compute_bbox_from_mask(final_masks, xyz_middle)
            ).float()

            num_pred = final_masks.shape[0]

            inst_ref_labels, target_quality = self._build_scannet_sem_candidate_labels(
                final_masks=final_masks,
                final_classes=inst_sem_mode_full,
                sem_gt_mask_np=sem_gt_mask_np,
                object_id=int(object_id),
            )

            inst_masks = torch.from_numpy(final_masks).float()
            inst_class_ids = torch.from_numpy(inst_sem_mode_full).long()
            inst_sem_mode_full = torch.from_numpy(inst_sem_mode_full).long()
            inst_filename = pred_filename
        else:
            # ===========================================
            # 分支 C: ScanNet GT 数据
            # ===========================================
            inst_filename = self.inst_map.get(scan_id)
            if inst_filename is None:
                raise FileNotFoundError(f"Missing instmeta file for {scan_id}")

            off_orig, idx_orig, _, inst_sem_mode_full = self.load_instmeta(inst_filename)

            max_idx = int(idx_orig.max().item()) if idx_orig.numel() > 0 else -1
            orig2new = torch.full((max_idx + 1,), -1, dtype=torch.long)

            orig_idx_keep_t = torch.as_tensor(orig_idx_keep, dtype=torch.long)
            if orig_idx_keep_t.numel() > 0 and max_idx >= 0:
                valid_mask = (orig_idx_keep_t <= max_idx)
                orig2new[orig_idx_keep_t[valid_mask]] = torch.arange(valid_mask.sum(), dtype=torch.long)

            inst2p_offsets_full, inst2p_indices_full = self.remap_csr_indices(off_orig, idx_orig, orig2new)

            num_inst = int(instance_label.max().item()) + 1

            inst_ref_labels, inst_class_ids = self._build_scannet_gt_inst_labels(
                instance_label=instance_label,
                semantic_label=semantic_label,
                object_id=object_id,
            )

            # 这里保留你原本的 ScanNet GT 逻辑
            inst_bbox_full = torch.zeros((num_inst, 6), dtype=torch.float32)
            for i in range(num_inst):
                pts_i = xyz_middle[instance_label == i]
                if pts_i.shape[0] > 0:
                    inst_bbox_full[i, :3] = torch.from_numpy(pts_i.min(axis=0)).float()
                    inst_bbox_full[i, 3:] = torch.from_numpy(pts_i.max(axis=0)).float()

            if torch.is_tensor(inst_sem_mode_full) and inst_sem_mode_full.numel() >= num_inst:
                inst_sem_mode_full = inst_sem_mode_full[:num_inst]

            num_sp = int(superpoint_aug.max().item()) + 1
            inst_masks = torch.zeros(num_inst, num_sp, dtype=torch.float32)
            inst_masks[instance_label, superpoint_aug] = 1.0
            if ann_type == "scannet" and raw_gt_mask_np is not None and raw_gt_mask_np.sum() > 0 and object_id_local >= 0:
                cur_mask = (instance_label == object_id_local)
                inter = np.logical_and(cur_mask, raw_gt_mask_np).sum()
                union = np.logical_or(cur_mask, raw_gt_mask_np).sum()
                target_quality = float(inter / (union + 1e-6))
            else:
                target_quality = 1.0

            inst_class_ids = torch.full((num_inst,), -1, dtype=torch.long)
            for i in range(num_inst):
                m = (instance_label == i)
                if m.any():
                    s = semantic_label[m]
                    valid_s = s[s >= 0]
                    if valid_s.size > 0:
                        inst_class_ids[i] = torch.bincount(torch.from_numpy(valid_s).long()).argmax()
        


        # -------------------------------------------------------
        # 6. 计算辅助 Loss 需要的 GT Mask
        # ScanNet: 已在 4.0 用 semantic GT 构造
        # MP3D: 这里补 instance/object GT
        # -------------------------------------------------------
        instance_label_tensor = torch.from_numpy(instance_label).long()
        superpoint_tensor = torch.from_numpy(superpoint_aug).long()

        if ann_type == "mp3d":
            if object_id_local >= 0:
                gt_pmask, gt_spmask = self.get_ref_mask(
                    instance_label_tensor,
                    superpoint_tensor,
                    object_id_local,
                )
            else:
                # 没找到合法 object_id_local，就给空 mask，避免 None 爆炸
                gt_pmask = torch.zeros(instance_label_tensor.shape[0], dtype=torch.float32)
                num_sp = int(superpoint_aug.max()) + 1 if superpoint_aug.size > 0 else 0
                gt_spmask = torch.zeros(num_sp, dtype=torch.float32)

        elif ann_type == "scannet":
            # ScanNet 的 gt_pmask / gt_spmask 已经在 4.0 构造好
            if gt_pmask is None or gt_spmask is None:
                raise RuntimeError(
                    f"[SCANNET-GT-ERROR] semantic gt not built | "
                    f"scan_id={scan_id} object_id={object_id}"
                )

        else:
            raise RuntimeError(f"[GT-ERROR] unknown ann_type={ann_type}")

        if gt_pmask is None or gt_spmask is None:
            raise RuntimeError(
                f"[GT-ERROR] gt_pmask/gt_spmask is None | "
                f"scan_id={scan_id} ann_type={ann_type} object_id={object_id}"
            )
        
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
        
        # -------------------------------------------------------
        # 7. Oracle pseudo viewpoint (reason v1)
        # -------------------------------------------------------
        scene_xyz_t = torch.from_numpy(xyz_middle).float()
        all_inst_centers_t = None
        if inst_bbox_full is not None and torch.is_tensor(inst_bbox_full) and inst_bbox_full.numel() > 0:
            # compute_bbox_from_mask 产出的 bbox 是 [cx, cy, cz, sx, sy, sz]
            # ScanNet GT 这里虽然还是旧格式 [min,max]，但本轮先不动 ScanNet 路径
            if ann_type == "mp3d" or is_pred_mode:
                all_inst_centers_t = inst_bbox_full[:, :3].float()

        scene_xyz_t = torch.from_numpy(xyz_middle).float()

        all_inst_centers_t = None
        if inst_bbox_full is not None and torch.is_tensor(inst_bbox_full) and inst_bbox_full.numel() > 0:
            all_inst_centers_t = inst_bbox_full[:, :3].float()

        # -----------------------------
        # cam 专用目标：默认不用 GT
        # -----------------------------
        cam_target_bbox = None
        cam_source = "none"

        if ann_type == "scannet" and is_pred_mode:
            cam_target_bbox, cam_score = self._select_cam_target_from_candidates(
                object_id=int(object_id),
                inst_bbox_full=inst_bbox_full,
                inst_class_ids=inst_class_ids,
                inst_masks=inst_masks,
                xyz_middle=xyz_middle,
            )
            if cam_target_bbox is not None:
                cam_source = "pred_sem"

        # 可选：如果你以后想保留一个极小概率 GT fallback，再开这个开关
        use_gt_cam_fallback = False
        if cam_target_bbox is None and use_gt_cam_fallback:
            if gt_pmask.sum().item() > 0 and target_bbox.abs().sum().item() > 1e-6:
                cam_target_bbox = target_bbox.float()
                cam_source = "gt"

        if cam_target_bbox is not None:
            cam_pos_t, cam_dir_t, cam_valid_t = build_pseudo_cam_from_box_and_text(
                question=question_raw,
                target_bbox=cam_target_bbox.float(),
                scene_xyz=scene_xyz_t,
                all_inst_centers=all_inst_centers_t,
            )
            cam_lookat_t = cam_target_bbox[:3].float()
        else:
            cam_pos_t = torch.zeros(3, dtype=torch.float32)
            cam_dir_t = torch.zeros(3, dtype=torch.float32)
            cam_lookat_t = torch.zeros(3, dtype=torch.float32)
            cam_valid_t = torch.tensor(0, dtype=torch.long)

        view_dep_t = torch.tensor(int(self._is_view_dependent(question_raw)), dtype=torch.long)
        gt_sem_pmask = None
        if ann_type == "scannet" and gt_pmask is not None:
            gt_sem_pmask = gt_pmask.clone()

        gt_inst_pmask_raw = None
        if ann_type == "scannet" and instance_id is not None and raw_instance_labels_full is not None:
            raw_inst_crop = raw_instance_labels_full[orig_idx_keep].astype(np.int64)
            gt_inst_pmask_raw = torch.from_numpy(
                (raw_inst_crop == int(instance_id)).astype(np.float32)
            )

                
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
            'cam_pos': cam_pos_t.float(),
            'cam_lookat': cam_lookat_t.float(),
            'cam_dir': cam_dir_t.float(),
            'cam_valid': cam_valid_t.long(),
            'view_dep': view_dep_t,
            'gt_sem_pmask': gt_sem_pmask,
            'gt_inst_pmask_raw': gt_inst_pmask_raw,
        }

    
    def collater(self, batch):
        ann_ids, scan_ids, coords, coords_float,xyz_worlds, feats, superpoints, object_ids, \
        gt_pmasks, gt_spmasks, sp_ref_masks, lang_tokenss, lang_masks, lang_words, \
        answerss, text_input_list, sp_filename_list, gt_pmasks_region, gt_spmasks_region, \
        obj_cats = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[]
        gt_vs_sem_stats = []
        sem_dbg_rows = []
        inst_vs_sem_stats = []
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

        nonmatch_rows = []
        upper_tq_rows = []
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

                gt_sem_pmask = data.get('gt_sem_pmask', None)
                gt_inst_pmask_raw = data.get('gt_inst_pmask_raw', None)
                # =======================================================
                # ScanNet semantic debug
                # 说明：
                # 1) gt_pmask 本身就是当前 semantic GT
                # 2) 更有意义的是看：
                #    - 同类候选并集 对 gt_pmask 的覆盖
                #    - 正样本候选并集 对 gt_pmask 的覆盖
                # =======================================================
                if str(scan_id).startswith("scene"):
                    gt_mask_dbg = (gt_pmask > 0.5)

                    gt_ratio = float(gt_mask_dbg.float().mean().item())
                    region_ratio = float((gt_pmask_region > 0.5).float().mean().item())

                    num_cands = 0
                    num_cls_match = 0
                    num_pos = 0

                    cls_cover_gt = 0.0
                    cls_precision = 0.0
                    cls_iou = 0.0
                    cls_ratio = 0.0

                    pos_cover_gt = 0.0
                    pos_precision = 0.0
                    pos_iou = 0.0
                    pos_ratio = 0.0

                    # 只在 point-level candidate mask 时统计
                    # 你的 ScanNet pred sem 分支里 inst_masks_b 是 [M, N_points]
                    if (
                        inst_masks_b is not None
                        and torch.is_tensor(inst_masks_b)
                        and inst_masks_b.ndim == 2
                        and inst_masks_b.shape[1] == gt_mask_dbg.numel()
                    ):
                        num_cands = int(inst_masks_b.shape[0])

                        if inst_class_ids_b is not None and torch.is_tensor(inst_class_ids_b):
                            cls_match_mask = (inst_class_ids_b == int(object_id))
                        else:
                            cls_match_mask = torch.zeros(num_cands, dtype=torch.bool, device=inst_masks_b.device)

                        if inst_ref_labels_b is not None and torch.is_tensor(inst_ref_labels_b):
                            pos_mask = (inst_ref_labels_b > 0.5)
                        else:
                            pos_mask = torch.zeros(num_cands, dtype=torch.bool, device=inst_masks_b.device)

                        num_cls_match = int(cls_match_mask.sum().item())
                        num_pos = int(pos_mask.sum().item())

                        # -------- 同类候选并集 --------
                        if num_cls_match > 0:
                            cls_union = (inst_masks_b[cls_match_mask].sum(dim=0) > 0)
                            inter = (cls_union & gt_mask_dbg).sum().item()
                            union = (cls_union | gt_mask_dbg).sum().item()
                            pred_sum = cls_union.sum().item()
                            gt_sum = gt_mask_dbg.sum().item()

                            cls_cover_gt = inter / (gt_sum + 1e-6)
                            cls_precision = inter / (pred_sum + 1e-6)
                            cls_iou = inter / (union + 1e-6)
                            cls_ratio = pred_sum / (gt_mask_dbg.numel() + 1e-6)

                        # -------- 正样本候选并集 --------
                        if num_pos > 0:
                            pos_union = (inst_masks_b[pos_mask].sum(dim=0) > 0)
                            inter = (pos_union & gt_mask_dbg).sum().item()
                            union = (pos_union | gt_mask_dbg).sum().item()
                            pred_sum = pos_union.sum().item()
                            gt_sum = gt_mask_dbg.sum().item()

                            pos_cover_gt = inter / (gt_sum + 1e-6)
                            pos_precision = inter / (pred_sum + 1e-6)
                            pos_iou = inter / (union + 1e-6)
                            pos_ratio = pred_sum / (gt_mask_dbg.numel() + 1e-6)

                    sem_dbg_rows.append({
                        "scan_id": str(scan_id),
                        "ann_id": int(ann_id),
                        "object_id": int(object_id),
                        "gt_ratio": gt_ratio,
                        "region_ratio": region_ratio,
                        "num_cands": num_cands,
                        "num_cls_match": num_cls_match,
                        "num_pos": num_pos,
                        "cls_cover_gt": cls_cover_gt,
                        "cls_precision": cls_precision,
                        "cls_iou": cls_iou,
                        "cls_ratio": cls_ratio,
                        "pos_cover_gt": pos_cover_gt,
                        "pos_precision": pos_precision,
                        "pos_iou": pos_iou,
                        "pos_ratio": pos_ratio,
                        "target_quality": float(t_qual),
                    })
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

            # -------------------------------------------------------
            # Debug 1: 当前 gt_pmask 和显式保存的 gt_sem_pmask 的覆盖率
            # 说明：按你现在的代码，这个理论上应接近 1
            # -------------------------------------------------------
            if gt_sem_pmask is not None:
                a = (gt_pmask > 0.5)
                b = (gt_sem_pmask > 0.5)

                inter = (a & b).sum().item()
                union = (a | b).sum().item()

                cover_a_by_b = inter / (a.sum().item() + 1e-6)
                cover_b_by_a = inter / (b.sum().item() + 1e-6)
                iou_ab = inter / (union + 1e-6)

                gt_vs_sem_stats.append([cover_a_by_b, cover_b_by_a, iou_ab])


            # -------------------------------------------------------
            # Debug 2: raw instance GT 和 semantic GT 的覆盖率
            # 这个更有诊断价值
            # -------------------------------------------------------
            if gt_inst_pmask_raw is not None and gt_sem_pmask is not None:
                a = (gt_inst_pmask_raw > 0.5)
                b = (gt_sem_pmask > 0.5)

                inter = (a & b).sum().item()
                union = (a | b).sum().item()

                cover_inst_by_sem = inter / (a.sum().item() + 1e-6)
                cover_sem_by_inst = inter / (b.sum().item() + 1e-6)
                iou_inst_sem = inter / (union + 1e-6)

                inst_vs_sem_stats.append([cover_inst_by_sem, cover_sem_by_inst, iou_inst_sem])

            # =======================================================
            # 统计 theoretical upper bound + nonmatch 样本
            # =======================================================
            upper_tq_rows.append(float(t_qual))

            if str(scan_id).startswith("scene"):
                gt_ratio_dbg = float((gt_pmask > 0.5).float().mean().item())
                region_ratio_dbg = float((gt_pmask_region > 0.5).float().mean().item())

                num_cands_dbg = 0
                cls_match_num_dbg = 0

                if inst_masks_b is not None and torch.is_tensor(inst_masks_b):
                    num_cands_dbg = int(inst_masks_b.shape[0])

                if inst_class_ids_b is not None and torch.is_tensor(inst_class_ids_b):
                    cls_match_num_dbg = int((inst_class_ids_b == int(object_id)).sum().item())

                # 没有同类候选
                if cls_match_num_dbg == 0:
                    nonmatch_rows.append({
                        "scan_id": str(scan_id),
                        "ann_id": int(ann_id),
                        "object_id": int(object_id),
                        "obj_cat": str(obj_cat),
                        "is_scannet200_only": bool(int(object_id) > 19),
                        "cand": num_cands_dbg,
                        "gt": gt_ratio_dbg,
                        "region": region_ratio_dbg,
                    })
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

        
        # =======================================================
        # batch-level semantic debug print
        # =======================================================
        def _is_rank0():
            try:
                import torch.distributed as dist
                return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
            except Exception:
                return True

        if _is_rank0() and len(sem_dbg_rows) > 0 and False:
            n = len(sem_dbg_rows)
            '''
            mean_gt_ratio = sum(x["gt_ratio"] for x in sem_dbg_rows) / n
            mean_region_ratio = sum(x["region_ratio"] for x in sem_dbg_rows) / n
            mean_num_cands = sum(x["num_cands"] for x in sem_dbg_rows) / n
            mean_num_cls_match = sum(x["num_cls_match"] for x in sem_dbg_rows) / n
            mean_num_pos = sum(x["num_pos"] for x in sem_dbg_rows) / n

            mean_cls_cover_gt = sum(x["cls_cover_gt"] for x in sem_dbg_rows) / n
            mean_cls_precision = sum(x["cls_precision"] for x in sem_dbg_rows) / n
            mean_cls_iou = sum(x["cls_iou"] for x in sem_dbg_rows) / n
            mean_cls_ratio = sum(x["cls_ratio"] for x in sem_dbg_rows) / n

            mean_pos_cover_gt = sum(x["pos_cover_gt"] for x in sem_dbg_rows) / n
            mean_pos_precision = sum(x["pos_precision"] for x in sem_dbg_rows) / n
            mean_pos_iou = sum(x["pos_iou"] for x in sem_dbg_rows) / n
            mean_pos_ratio = sum(x["pos_ratio"] for x in sem_dbg_rows) / n

            mean_tq = sum(x["target_quality"] for x in sem_dbg_rows) / n
            
            print(
                "[SEM-BATCH] "
                f"bs={n} "
                f"gt={mean_gt_ratio:.3f} region={mean_region_ratio:.3f} "
                f"cand={mean_num_cands:.2f} cls_match={mean_num_cls_match:.2f} pos={mean_num_pos:.2f} "
                f"cls(cov={mean_cls_cover_gt:.3f},prec={mean_cls_precision:.3f},iou={mean_cls_iou:.3f},ratio={mean_cls_ratio:.3f}) "
                f"pos(cov={mean_pos_cover_gt:.3f},prec={mean_pos_precision:.3f},iou={mean_pos_iou:.3f},ratio={mean_pos_ratio:.3f}) "
                f"tq={mean_tq:.3f}"
            )
            '''

            # 额外打印最差的几个样本，方便你抓具体问题
            worst_rows = sorted(sem_dbg_rows, key=lambda x: x["pos_iou"])[: min(3, len(sem_dbg_rows))]
            for x in worst_rows:
                print(
                    "[SEM-WORST] "
                    f"scan={x['scan_id']} ann={x['ann_id']} obj={x['object_id']} "
                    f"gt={x['gt_ratio']:.3f} region={x['region_ratio']:.3f} "
                    f"cand={x['num_cands']} cls_match={x['num_cls_match']} pos={x['num_pos']} "
                    f"cls(cov={x['cls_cover_gt']:.3f},prec={x['cls_precision']:.3f},iou={x['cls_iou']:.3f},ratio={x['cls_ratio']:.3f}) "
                    f"pos(cov={x['pos_cover_gt']:.3f},prec={x['pos_precision']:.3f},iou={x['pos_iou']:.3f},ratio={x['pos_ratio']:.3f}) "
                    f"tq={x['target_quality']:.3f}"
                )

            # =======================================================
            # mismatch / theoretical upper bound debug
            # 放在 for batch 循环结束后，batch_offsets = ... 之前
            # =======================================================
            def _is_rank0():
                try:
                    import torch.distributed as dist
                    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
                except Exception:
                    return True

            if _is_rank0():
                # 这些列表需要在 for 循环前初始化
                # nonmatch_rows = []
                # upper_tq_rows = []

                total_samples = len(scan_ids)
                nonmatch_count = len(nonmatch_rows)
                match_count = total_samples - nonmatch_count
                nonmatch_ratio = nonmatch_count / (total_samples + 1e-6)

                # 当前 batch 理论上限：
                # tq = 当前候选池下“最好可选候选”的 IoU（你当前语义候选设置下基本可这么理解）
                upper_all = 0.0
                upper_matched = 0.0
                if len(upper_tq_rows) > 0:
                    upper_all = float(sum(upper_tq_rows) / len(upper_tq_rows))
                if match_count > 0:
                    matched_tqs = [x for x in upper_tq_rows if x > 0]
                    if len(matched_tqs) > 0:
                        upper_matched = float(sum(matched_tqs) / len(matched_tqs))

                print(
                    "[SEM-UPPER] "
                    f"bs={total_samples} "
                    f"nonmatch={nonmatch_count} "
                    f"nonmatch_ratio={nonmatch_ratio:.4f} "
                    f"upper_all={upper_all:.4f} "
                    f"upper_matched={upper_matched:.4f}"
                )

                if nonmatch_count > 0:
                    print("[SEM-NOMATCH-SAMPLES]")
                    for row in nonmatch_rows:
                        print(
                            "  "
                            f"scan={row['scan_id']} "
                            f"ann={row['ann_id']} "
                            f"obj_id={row['object_id']} "
                            f"obj_cat={row['obj_cat']} "
                            f"is_scannet200_only={row['is_scannet200_only']} "
                            f"cand={row['cand']} "
                            f"gt={row['gt']:.4f} "
                            f"region={row['region']:.4f}"
                        )
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
        }
        if len(gt_vs_sem_stats) > 0:
            out['debug_gt_vs_sem_stats'] = torch.tensor(gt_vs_sem_stats, dtype=torch.float32)

        if len(inst_vs_sem_stats) > 0:
            out['debug_inst_vs_sem_stats'] = torch.tensor(inst_vs_sem_stats, dtype=torch.float32)
        return out 
    
    def __len__(self):
        return len(self.annotation)


    def _select_cam_target_from_candidates(
        self,
        object_id: int,
        inst_bbox_full,
        inst_class_ids,
        inst_masks=None,
        xyz_middle=None,
    ):
        """
        从候选里选一个 bbox 作为 cam 的 pseudo target。
        不使用 gt_pmask / gt bbox。
        适合 ScanNet semantic candidates。
        """
        if inst_bbox_full is None or inst_class_ids is None:
            return None, 0.0

        if torch.is_tensor(inst_bbox_full):
            bbox = inst_bbox_full.detach().cpu().numpy()
        else:
            bbox = np.asarray(inst_bbox_full)

        if torch.is_tensor(inst_class_ids):
            cls_ids = inst_class_ids.detach().cpu().numpy()
        else:
            cls_ids = np.asarray(inst_class_ids)

        if bbox.shape[0] == 0 or cls_ids.shape[0] == 0:
            return None, 0.0

        # 先只看同语义类候选
        cand_idx = np.where(cls_ids == int(object_id))[0]
        if cand_idx.size == 0:
            return None, 0.0

        # 候选面积/体积打分
        if inst_masks is not None:
            if torch.is_tensor(inst_masks):
                area = inst_masks.float().sum(dim=1).detach().cpu().numpy()
            else:
                area = inst_masks.sum(axis=1)
        else:
            size = np.maximum(bbox[:, 3:6], 1e-3)
            area = np.prod(size, axis=1)

        area_score = area[cand_idx] / (area[cand_idx].max() + 1e-6)

        # 再加一个“别太偏场景边缘”的中心性
        if xyz_middle is not None:
            scene_center = xyz_middle.mean(axis=0)
            dist = np.linalg.norm(bbox[cand_idx, :3] - scene_center[None, :], axis=1)
            dist_score = 1.0 - dist / (dist.max() + 1e-6)
        else:
            dist_score = np.ones_like(area_score)

        # 面积优先，中心性次之
        score = 0.7 * area_score + 0.3 * dist_score
        best_idx = cand_idx[int(score.argmax())]

        return torch.from_numpy(bbox[best_idx]).float(), float(score.max())
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

    def load_pointcept_instmeta(self, filename, xyz_coords):
        """
        读取 Pointcept 生成的 .pt 预测文件，并适配分割输入格式。
        注意：预测结果是重叠的，所以不能生成 inst2p_offsets (CSR)，
        而是直接返回 masks 和 现场计算 bbox。
        
        Args:
            filename: .pt 文件路径
            xyz_coords: 对应的原始点云坐标 [N, 3]，用于计算 bbox
        """
        # 1. 加载数据 (CPU)
        data = torch.load(filename, map_location="cpu")
        
        # 2. 提取基础数据
        # feat: [N, C] (Backbone 特征)
        # 这里的 feat 已经是 float16 了，转回 float32 方便计算
        feat = data['feat'].float() if data['feat'] is not None else None
        
        # pred_masks: [M, N] (Bool or Logits) -> 预测的实例掩码
        pred_masks = data['pred_masks']
        if torch.is_tensor(pred_masks):
            pred_masks = pred_masks.bool() # 确保是布尔值
        else:
            pred_masks = torch.from_numpy(pred_masks).bool()

        # pred_classes: [M] -> 语义类别
        inst_sem_mode = data['pred_classes']
        if not torch.is_tensor(inst_sem_mode):
            inst_sem_mode = torch.from_numpy(inst_sem_mode).long()
        
        # pred_scores: [M] -> 置信度 (可按置信度过滤低分 proposal)
        inst_scores = data['pred_scores']
        if not torch.is_tensor(inst_scores):
            inst_scores = torch.from_numpy(inst_scores).float()

        # 3. [关键] 现场计算 BBox (inst_bbox_full)
        # 因为预测结果里没存 BBox，我们需要用 Mask + XYZ 算出来
        # pred_masks: [M, N], xyz: [N, 3]
        num_inst = pred_masks.shape[0]
        inst_bbox = torch.zeros((num_inst, 6), dtype=torch.float32)
        
        # 确保 xyz 是 tensor
        if not torch.is_tensor(xyz_coords):
            xyz_t = torch.from_numpy(xyz_coords).float()
        else:
            xyz_t = xyz_coords.float()

        for i in range(num_inst):
            mask = pred_masks[i] # [N]
            if mask.any():
                pts = xyz_t[mask] # [K, 3]
                mn = pts.min(dim=0)[0]
                mx = pts.max(dim=0)[0]
                center = (mn + mx) / 2.0
                size = (mx - mn).clamp(min=1e-3)
                inst_bbox[i, :3] = center
                inst_bbox[i, 3:] = size
            else:
                # 空实例（虽然不该发生），设为 0
                pass

        # 4. 关于 CSR (Offsets/Indices) 的处理
        # 警告：由于预测是重叠的，我们无法生成标准的 inst2p_offsets。
        # 如果你的下游模型强依赖 offsets，这里会出问题。
        # offsets 用于 Mask Pooling。
        # 这里我们将 offsets 设为 None，强迫下游逻辑改用 Mask 矩阵。
        inst2p_offsets = None 
        inst2p_indices = None

        # 返回字典以便扩展
        return {
            "feat": feat,
            "inst_masks_full": pred_masks,  # [M, N]
            "inst_bbox_full": inst_bbox,    # [M, 6]
            "inst_sem_mode_full": inst_sem_mode, # [M]
            "inst_scores_full": inst_scores # [M]
        }

    def pointcept_mask_to_csr(self, pred_masks, num_points):
        """
        将 [M, N] 的预测 Mask 转换为 CSR 格式 (offsets, indices)。
        策略：Hard Assignment (硬分配)。如果一个点属于多个实例，
        为了适配 CSR，我们这里简化处理：
        1. 简单的 argmax (每个点只属于一个实例) -> 容易丢失重叠信息
        2. 或者：既然 forward 里是用 CSR 来 pooling 特征，
           我们可以只保留置信度高的点。
        
        为了跑通你的 forward，这里采用 "转成 instance_label" 的策略。
        """
        # pred_masks: [M, N] (Bool or 0/1)
        M, N = pred_masks.shape
        
        # 构造一个空的 instance_label, 初始化为 -1
        inst_label = np.full((N,), -1, dtype=np.int64)
        
        # 将 Mask 转为 Label
        # 如果有重叠，后面的覆盖前面的 (或者你可以按 score 排序后覆盖)
        # 假设 pred_masks 已经按 score 从低到高排好序了，或者我们只取 argmax
        
        # 方法 A: 逐个覆盖 (简单)
        # valid_mask_indices = np.nonzero(pred_masks)
        # inst_label[valid_mask_indices[1]] = valid_mask_indices[0] # [1]是点索引, [0]是实例ID
        
        # 方法 B: Argmax (更严谨一点，但还是互斥)
        # 这种方法假设这一层是互斥的。PointGroup 结果往往重叠较多。
        # 如果你的 forward 必须用 scatter_mean (CSR)，那只能接受互斥。
        
        # 让我们用更暴力的 list 方式构建 indices，允许重叠！
        # 既然 offsets 定义的是 "实例 i 拥有哪些点"，那完全可以重叠！
        # 只要 indices 的总长度 > N 没关系！
        
        indices_list = []
        offsets = [0]
        
        for i in range(M):
            # 找到属于实例 i 的所有点索引
            pts_idx = np.nonzero(pred_masks[i])[0]
            indices_list.append(pts_idx)
            offsets.append(offsets[-1] + len(pts_idx))
            
        indices = np.concatenate(indices_list) if indices_list else np.array([], dtype=np.int64)
        offsets = np.array(offsets, dtype=np.int64)
        
        return offsets, indices

    # =================================================================
    # [新增工具 1] 将重叠 Mask 转换为 CSR 格式 (允许重复索引)
    # =================================================================
    
    def _build_scannet_sem_gt(self, semantic_label, superpoint_aug, object_id):
        semantic_label_t = torch.from_numpy(semantic_label).long()
        superpoint_t = torch.from_numpy(superpoint_aug).long()

        gt_pmask, gt_spmask = self.get_ref_mask(semantic_label_t, superpoint_t, int(object_id))
        sem_gt_mask_np = gt_pmask.bool().cpu().numpy()

        return sem_gt_mask_np, gt_pmask, gt_spmask


    def _build_scannet_gt_inst_labels(self, instance_label, semantic_label, object_id):
        num_inst = int(instance_label.max()) + 1 if instance_label.size > 0 else 0

        inst_ref_labels = torch.zeros(num_inst, dtype=torch.float32)
        inst_class_ids = torch.full((num_inst,), -1, dtype=torch.long)

        for i in range(num_inst):
            m = (instance_label == i)
            if not m.any():
                continue

            s = semantic_label[m]
            s = s[s >= 0].astype(np.int64)
            if s.size == 0:
                continue

            cls_i = int(np.bincount(s).argmax())
            inst_class_ids[i] = cls_i

            # 只要这个实例的主语义类别等于 object_id，就算正样本
            if cls_i == int(object_id):
                inst_ref_labels[i] = 1.0

        return inst_ref_labels, inst_class_ids
    
    def _build_scannet_pred_inst_labels(self, final_masks, sem_gt_mask_np):
        num_pred = final_masks.shape[0]
        inst_ref_labels = torch.zeros(num_pred, dtype=torch.float32)

        if num_pred == 0 or sem_gt_mask_np is None or sem_gt_mask_np.sum() == 0:
            return inst_ref_labels, 0.0

        inter = np.logical_and(final_masks, sem_gt_mask_np[None, :]).sum(axis=1)
        pred_area = final_masks.sum(axis=1)
        cover = inter / (pred_area + 1e-6)   # proposal 有多少比例落在 semantic GT 里

        # 多个 proposal 都可以是正样本
        pos_mask = cover > 0.5
        if pos_mask.any():
            inst_ref_labels[pos_mask] = 1.0
        else:
            best_idx = int(np.argmax(cover))
            best_cover = float(cover[best_idx])
            if best_cover > 0.25:
                inst_ref_labels[best_idx] = 1.0

        return inst_ref_labels, float(cover.max())
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

    '''
    # =====================================================================
    # [工具 2] 现场计算 BBox (因为预测结果里没有存 BBox)
    # =====================================================================
    def compute_bbox_from_mask(self, masks, xyz):
        """
        输入: masks [M, N], xyz [N, 3]
        输出: bbox [M, 6]
        """
        M = masks.shape[0]
        bbox = np.zeros((M, 6), dtype=np.float32)
        for i in range(M):
            pts_idx = np.nonzero(masks[i])[0]
            if len(pts_idx) > 0:
                pts = xyz[pts_idx] 
                mn = pts.min(axis=0)
                mx = pts.max(axis=0)
                bbox[i, :3] = mn
                bbox[i, 3:] = mx
        return bbox
    '''
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
    
    def _save_region_debug_png(
        self,
        out_dir: str,
        scan_id: str,
        ann_id: str,
        xyz: np.ndarray,          # [N,3] float32
        gt_obj: np.ndarray,       # [N] bool/0-1
        gt_region: np.ndarray,    # [N] bool/0-1
        max_pts: int = 20000,
    ):
        # 只在服务器保存图：用 Agg 后端
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(out_dir, exist_ok=True)

        xyz = xyz.astype(np.float32)
        gt_obj = gt_obj.astype(bool)
        gt_region = gt_region.astype(bool)

        N = xyz.shape[0]
        if N > max_pts:
            idx = np.random.choice(N, max_pts, replace=False)
            xyz = xyz[idx]
            gt_obj = gt_obj[idx]
            gt_region = gt_region[idx]

        pos_frac = float(gt_region.mean())
        obj_frac = float(gt_obj.mean())

        # 三类：背景0 / region-only1 / obj2（obj 优先）
        cls = np.zeros(xyz.shape[0], dtype=np.int32)
        cls[gt_region] = 1
        cls[gt_obj] = 2

        # 三个投影：XY / XZ / YZ
        proj = [("XY", 0, 1), ("XZ", 0, 2), ("YZ", 1, 2)]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for ax, (name, a, b) in zip(axes, proj):
            # 背景（很淡）
            m0 = (cls == 0)
            ax.scatter(xyz[m0, a], xyz[m0, b], s=1, alpha=0.03)

            # region-only（中等淡）
            m1 = (cls == 1)
            ax.scatter(xyz[m1, a], xyz[m1, b], s=2, alpha=0.15)

            # obj（最明显）
            m2 = (cls == 2)
            ax.scatter(xyz[m2, a], xyz[m2, b], s=3, alpha=0.9)

            ax.set_title(name)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")

        fig.suptitle(
            f"{scan_id} | {ann_id} | region_pos={pos_frac:.3f} obj_pos={obj_frac:.3f} | N={N}",
            fontsize=12
        )

        fn = f"{scan_id}_{ann_id}_region{pos_frac:.3f}_obj{obj_frac:.3f}_N{xyz.shape[0]}.png"
        out_path = os.path.join(out_dir, fn)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close(fig)

        return out_path

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

    def _build_scannet_sem_candidate_labels(
        self,
        final_masks: np.ndarray,      # [M, N] bool
        final_classes: np.ndarray,    # [M]
        sem_gt_mask_np: np.ndarray,   # [N] bool
        object_id: int,
        pos_iou_thr: float = 0.25,
        weak_iou_thr: float = 0.10,
    ):
        """
        给 ScanNet semantic candidates 打标签。

        当前数据流里：
        - final_masks: 每个出现过的 semantic class 对应一个 candidate mask
        - final_classes: 每个 candidate 的 semantic class id
        - sem_gt_mask_np: semantic GT mask（由 semantic_label == object_id 构造）
        - object_id: 当前样本目标的 semantic class id

        返回：
        - inst_ref_labels: [M] float32，候选正负标签
        - target_quality: float，最佳同类候选和 semantic GT 的 IoU
        """
        num_pred = int(final_masks.shape[0])
        inst_ref_labels = torch.zeros(num_pred, dtype=torch.float32)

        if num_pred == 0 or sem_gt_mask_np is None or sem_gt_mask_np.sum() == 0:
            return inst_ref_labels, 0.0

        if torch.is_tensor(final_classes):
            final_classes = final_classes.cpu().numpy()
        else:
            final_classes = np.asarray(final_classes)

        final_classes = final_classes.astype(np.int64)
        sem_gt_mask_np = sem_gt_mask_np.astype(np.bool_)

        # 1) 先只保留语义类别匹配的候选
        class_match = (final_classes == int(object_id))
        if not class_match.any():
            return inst_ref_labels, 0.0

        match_indices = np.nonzero(class_match)[0]
        match_masks = final_masks[match_indices]   # [K, N]

        # 2) 用 IoU 衡量 candidate 与 semantic GT 的质量
        inter = np.logical_and(match_masks, sem_gt_mask_np[None, :]).sum(axis=1)
        union = np.logical_or(match_masks, sem_gt_mask_np[None, :]).sum(axis=1)
        ious = inter / (union + 1e-6)

        best_local = int(np.argmax(ious))
        best_idx = int(match_indices[best_local])
        best_iou = float(ious[best_local])

        # 3) 打正样本
        #    - 若有多个同类 candidate 都足够好，就都可为正
        #    - 否则给最佳候选一个弱正，避免全 0
        strong_pos = (ious >= pos_iou_thr)
        if strong_pos.any():
            pos_global = match_indices[strong_pos]
            inst_ref_labels[pos_global] = 1.0
        elif best_iou >= weak_iou_thr:
            inst_ref_labels[best_idx] = 1.0

        return inst_ref_labels, best_iou


import math
import torch

def safe_normalize(x, eps=1e-6):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)

def infer_view_mode(question: str):
    q = question.lower()

    wall_like = [
        "window", "curtain", "blinds", "mirror", "poster",
        "board", "whiteboard", "blackboard", "sign",
        "door", "doorway", "entrance", "exit"
    ]
    interact_like = [
        "sit", "seating", "relax", "lounge", "watch",
        "work", "study", "type", "computer",
        "cook", "wash", "clean", "open", "place",
        "store", "hold", "use"
    ]

    if any(k in q for k in wall_like):
        return "interior"
    if any(k in q for k in interact_like):
        return "open_side"
    return "mixed"


def build_pseudo_cam_from_box_and_text(
    question,           # str
    target_bbox,        # Tensor [6] = [cx, cy, cz, w, h, l]
    scene_xyz,          # Tensor [N, 3]
    all_inst_centers=None,  # Tensor [I, 3] or None
):
    device = scene_xyz.device
    dtype = scene_xyz.dtype

    if target_bbox is None or torch.isnan(target_bbox).any():
        return (
            torch.zeros(3, device=device, dtype=dtype),
            torch.zeros(3, device=device, dtype=dtype),
            torch.tensor(0, device=device, dtype=torch.long),
        )

    c_t = target_bbox[:3]
    size = target_bbox[3:6].clamp(min=1e-3)

    # scene center
    c_s = scene_xyz.mean(dim=0)

    # inward direction
    u_in = c_s - c_t
    if u_in.norm() < 1e-6:
        u_in = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
    u_in = safe_normalize(u_in.unsqueeze(0)).squeeze(0)

    # candidate radius
    rho = (0.6 * torch.sqrt(size[0] ** 2 + size[1] ** 2) + 0.5).clamp(0.8, 2.0)
    dz = (0.15 * size[2]).clamp(-0.05, 0.25)

    # open-side direction by 8-dir sampling
    thetas = torch.arange(8, device=device, dtype=dtype) * (2.0 * math.pi / 8.0)
    dirs = torch.stack([
        torch.cos(thetas),
        torch.sin(thetas),
        torch.zeros_like(thetas)
    ], dim=-1)  # [8,3]

    cand_pos = c_t.unsqueeze(0) + rho * dirs
    cand_pos[:, 2] = cand_pos[:, 2] + dz

    if all_inst_centers is not None and all_inst_centers.shape[0] > 1:
        dmat = torch.cdist(cand_pos.float(), all_inst_centers.float())  # [8, I]
        free_score = dmat.min(dim=1)[0].to(dtype)
        k = int(free_score.argmax().item())
        u_open = dirs[k]
    else:
        u_open = u_in

    mode = infer_view_mode(question)
    if mode == "interior":
        u = u_in
    elif mode == "open_side":
        u = u_open
    else:
        u = safe_normalize((0.6 * u_in + 0.4 * u_open).unsqueeze(0)).squeeze(0)

    cam_pos = c_t + rho * u
    cam_pos[2] = cam_pos[2] + dz

    cam_dir = safe_normalize((c_t - cam_pos).unsqueeze(0)).squeeze(0)
    cam_valid = torch.tensor(1, device=device, dtype=torch.long)

    return cam_pos, cam_dir, cam_valid


QUESTION_LIST = [
    "Please identify the room first then segment the object according to the given 3D scene and the description: {description}.",
    "Given the 3D scene, provide the room first then segment the object according to the description: {description}.",
    "Respond the room mask first then the segmentation mask of the object: {description}.",
]

ROOM_TYPE_LIST = [
    "bathroom", "bedroom", "closet", "dining room", "entryway", "familyroom",
    "garage", "hallway", "library", "laundryroom", "kitchen", "living room",
    "conference room", "lounge", "office", "porch", "game", "stairs",
    "toilet", "utility room", "tv", "workout", "outdoor areas", "balcony",
    "other room", "bar", "classroom", "dining booth", "spa", "junk"
]

ANSWER_LIST = [
    "The location of {location} is [LOC]. The mask is [SEG].",
    "Sure, the location of {location} is [LOC] and the mask is [SEG].",
    "Sure, the location of {location} is [LOC]. The segmentation result is [SEG].",
]

DIR_PATTERNS = [
    r"\bleft\b", r"\bright\b",
    r"\bfront\b", r"\bbehind\b", r"\bback\b",
    r"\bfacing\b", r"\bin front of\b", r"\bin the back\b",
    r"\bon the left\b", r"\bon the right\b",
    r"\bleftmost\b", r"\brightmost\b",
]
 
DIR_REGEX = re.compile("|".join(DIR_PATTERNS), flags=re.IGNORECASE)


