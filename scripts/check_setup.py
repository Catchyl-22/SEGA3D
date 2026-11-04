"""Check configuration, assets and optional imports without loading model weights."""
import argparse
import importlib
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "eval"], default="eval")
    parser.add_argument("--task", choices=["scanrefer", "reasoning-scannet", "reasoning-matterport"], default="scanrefer")
    parser.add_argument("--model", choices=["nv2", "nv3"], help="Default: nv2 for ScanRefer, nv3 for reasoning segmentation.")
    parser.add_argument("--assets", action="store_true", help="Check actual scene/candidate/checkpoint paths.")
    parser.add_argument("--environment", action="store_true", help="Import runtime dependencies and check CUDA.")
    parser.add_argument("--checkpoint", help="Override the evaluation checkpoint path.")
    args = parser.parse_args()
    model_variant = args.model or ("nv2" if args.task == "scanrefer" else "nv3")
    if args.task == "scanrefer" and model_variant != "nv2":
        parser.error("The ScanRefer workflow uses nv2.")
    os.chdir(ROOT)
    from omegaconf import OmegaConf
    dataset_name = {
        "scanrefer": "sega3d_scanrefer",
        "reasoning-scannet": "sega3d_reasoning_scannet",
        "reasoning-matterport": "sega3d_reasoning_matterport",
    }[args.task]
    defaults_path = {
        "scanrefer": "lavis/configs/datasets/sega3d_scanrefer/defaults.yaml",
        "reasoning-scannet": "lavis/configs/datasets/sega3d_reasoning_scannet/defaults.yaml",
        "reasoning-matterport": "lavis/configs/datasets/sega3d_reasoning_matterport/defaults.yaml",
    }[args.task]
    folder = "train" if args.mode == "train" else "val"
    dataset_suffix = args.task.removeprefix("reasoning-")
    task_path = f"lavis/projects/sega3d/{folder}/sega3d-{dataset_suffix}-{model_variant}.yaml"
    cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml"),
        OmegaConf.load(ROOT / defaults_path),
        OmegaConf.load(ROOT / task_path),
    )
    if args.checkpoint:
        cfg.model.finetuned = args.checkpoint
    errors = []
    def check(ok, message):
        print(f"[{'OK' if ok else 'MISSING/ERROR'}] {message}")
        if not ok:
            errors.append(message)
    check(cfg.model.arch == f"sega3d_t5_{model_variant}", f"{model_variant} model")
    check(list(cfg.datasets) == [dataset_name], f"{args.task} loader: {dataset_name}")
    check(bool(cfg.run.evaluate) == (args.mode == "eval"), f"{args.mode} execution mode")
    check(int(cfg.run.batch_size_eval) == 1, "evaluation batch size = 1")
    if model_variant == "nv2":
        check(int(cfg.model.get("eval_topk", 4)) == 4, "full inference K = 4")
        check(not cfg.model.get("save_topk_diagnostics", False), "normal evaluation; optional Top-K diagnostics disabled")
    if args.assets and args.task != "scanrefer":
        check_reasoning_assets(cfg, dataset_name, args.mode, check)
    if args.assets and args.task == "scanrefer":
        info = cfg.datasets["sega3d_scanrefer"].build_info
        point_root = Path(info.points.storage)
        check(point_root.is_dir(), f"scene root: {point_root}")
        check(Path(cfg.model.point_encoder_cfg.pretrained).is_file(), f"point encoder: {cfg.model.point_encoder_cfg.pretrained}")
        if args.mode == "train":
            check(Path(cfg.model.pretrained).is_file(), f"initialization: {cfg.model.pretrained}")
        else:
            check(Path(cfg.model.finetuned).is_file(), f"trained model: {cfg.model.finetuned}")
        whitelist = Path(info.candidates.whitelist)
        valid_whitelist = whitelist.is_file() and any(line.strip() and not line.lstrip().startswith("#") for line in whitelist.read_text(encoding="utf-8").splitlines())
        check(valid_whitelist, f"nonempty category whitelist: {whitelist}")
        for split in (["train", "val"] if args.mode == "train" else ["val"]):
            annotation = Path(info.annotations[split].storage)
            check(annotation.is_file(), f"{split} annotations: {annotation}")
            if not annotation.is_file():
                continue
            try:
                rows = json.loads(annotation.read_text(encoding="utf-8"))
                if not isinstance(rows, list) or not rows or any(not isinstance(r, dict) or not r.get("scene_id") for r in rows):
                    raise ValueError("Expected a nonempty list of annotations containing scene_id")
                required_fields = {"scene_id", "object_id", "ann_id", "object_name", "description", "camera"}
                bad_rows = [i for i, row in enumerate(rows) if not required_fields.issubset(row)]
                if bad_rows:
                    raise ValueError(f"Use camera-merged nv2 annotations; required fields missing in rows {bad_rows[:5]}")
                for row in rows:
                    camera = row["camera"]
                    if camera is not None and (not isinstance(camera, dict) or any(
                        not isinstance(camera.get(key), (list, tuple)) or len(camera[key]) != 3
                        for key in ("position", "lookat")
                    )):
                        raise ValueError("camera must contain three-component position/lookat vectors, or be null")
                ids = sorted({str(row["scene_id"]) for row in rows})
                if any(Path(scene).name != scene or "/" in scene or "\\" in scene for scene in ids):
                    raise ValueError("scene_id must be a plain scene identifier")
            except (ValueError, OSError) as error:
                check(False, f"{annotation}: {error}")
                continue
            prefix = "train_sp_refined" if split == "train" else "val_sp_refined"
            scene_dir = point_root / "scannetv2" / prefix
            metadata_dir = point_root / "scannetv2" / (prefix+"_instance")
            metadata_files = list(metadata_dir.glob("*")) if metadata_dir.is_dir() else []
            missing_scenes = [scene for scene in ids if not (scene_dir / (scene+"_refer.pth")).is_file()]
            missing_meta = [scene for scene in ids if not any(scene in p.name for p in metadata_files)]
            check(not missing_scenes, f"{split} processed scenes ({len(ids)} scenes; missing {len(missing_scenes)}: {missing_scenes[:5]})")
            check(not missing_meta, f"{split} instance metadata (missing {len(missing_meta)}: {missing_meta[:5]})")
            for source in ["scannet20", "scannet200"]:
                folder = Path(info.candidates[source])
                missing = [scene for scene in ids if not (folder / (scene+".pt")).is_file()]
                check(not missing, f"{split} {source} candidates (missing {len(missing)}: {missing[:5]})")
    if args.environment:
        check(sys.version_info[:2] == (3,10), "Python 3.10")
        for module in ["torch", "torchvision", "torch_scatter", "torch_cluster", "spconv.pytorch", "gorilla", "transformers", "pointgroup_ops", "lavis"]:
            try:
                loaded = importlib.import_module(module)
                check(True, f"import {module} {getattr(loaded, '__version__', '')}".strip())
                if module == "torch":
                    check(loaded.cuda.is_available(), "CUDA GPU available")
                    check(loaded.version.cuda == "11.8", f"PyTorch CUDA build: {loaded.version.cuda}")
            except Exception as error:
                check(False, f"import {module}: {type(error).__name__}: {error}")
    print(f"\n{len(errors)} issue(s). Checks do not load checkpoints or execute the model.")
    return 1 if errors else 0


def check_reasoning_assets(cfg, dataset_name, mode, check):
    info = cfg.datasets[dataset_name].build_info
    point_root = Path(info.points.storage)
    source = "scannet" if dataset_name == "sega3d_reasoning_scannet" else "mp3d"
    check(point_root.is_dir(), f"SEGA3D scene root: {point_root}")
    check(Path(cfg.model.point_encoder_cfg.pretrained).is_file(), "point-encoder checkpoint")
    check(Path(cfg.model.finetuned).is_file(), "ScanRefer initialization" if mode == "train" else "SEGA3D task checkpoint")
    for split in (["train", "val"] if mode == "train" else ["val"]):
        annotation = Path(info.annotations[split].storage)
        check(annotation.is_file(), f"{split} annotations: {annotation}")
        if not annotation.is_file():
            continue
        rows = json.loads(annotation.read_text(encoding="utf-8"))
        check(isinstance(rows, list) and bool(rows), "nonempty annotation list")
        if not isinstance(rows, list):
            continue
        selected = []
        for row in rows:
            if not isinstance(row, dict):
                check(False, "annotation must be an object")
                continue
            scene = str(row.get("scene_id", row.get("scene_name", "")))
            ann_type = str(row.get("type", "")).lower()
            if ann_type not in {"scannet", "mp3d"}:
                ann_type = "mp3d" if "_region" in scene and not scene.startswith("scene") else "scannet"
            if ann_type != source:
                continue
            valid = bool(scene) and Path(scene).name == scene and "/" not in scene and "\\" not in scene
            valid = valid and "object_id" in row and "question" in row
            check(valid, f"{split} annotation fields: {scene}")
            if valid:
                selected.append(scene)
        ids = sorted(set(selected))
        check(bool(ids), f"{split} has records for {source}")
        if source == "scannet":
            scene_dir = point_root / "scannetv2" / split
            missing = [scene for scene in ids if not (scene_dir / (scene+"_reason.pth")).is_file()]
            pred_dir = Path(info.candidates.scannet_semantic)
            missing_predictions = [scene for scene in ids if not (pred_dir / (scene+"_pred.npy")).is_file()]
            check(not missing, f"{split} ScanNet scenes: missing {missing[:5]}")
            check(not missing_predictions, f"{split} ScanNet semantic predictions: missing {missing_predictions[:5]}")
        else:
            scene_dir = point_root / "matterport" / "mp3d_data"
            split_file = point_root / "matterport" / ("scenes_"+split+".txt")
            check(split_file.is_file(), f"{split} Matterport3D building split")
            if split_file.is_file():
                buildings = set(split_file.read_text().splitlines())
                check(all(scene.split("_region")[0] in buildings for scene in ids), "selected buildings match split file")
            check(all((scene_dir / (scene+".pth")).is_file() for scene in ids), f"{split} Matterport3D scene files")
            inst_dir = Path(info.candidates.matterport_instances)
            check(all((inst_dir / (scene+".pt")).is_file() for scene in ids), f"{split} Matterport3D instance predictions")
            def normalize(name):
                stem = name.removesuffix("_pred")
                match = re.match(r"^([A-Za-z0-9]+)_(?:region)?(\d+)$", stem)
                return f"{match[1]}_region{int(match[2])}" if match else stem
            sem_dir = Path(info.candidates.matterport_semantic)
            semantic_ids = {normalize(p.stem) for p in sem_dir.glob("*.npy")}
            check(all(scene in semantic_ids for scene in ids), f"{split} Matterport3D semantic predictions")

if __name__ == "__main__":
    raise SystemExit(main())
