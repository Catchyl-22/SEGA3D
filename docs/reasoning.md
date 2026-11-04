# SEGA3D reasoning segmentation

The second stage uses separate ScanNet and Matterport3D loaders. The nv3 ScanNet and Matterport3D task configurations select `sega3d_t5_nv3` and initialize training from the ScanRefer checkpoint through `model.load_finetuned: true` and `model.finetuned`.

Configuration names specify both dataset and model variant: `sega3d-scannet-nv3.yaml` and `sega3d-matterport-nv3.yaml` select `sega3d_t5_nv3`. Matching files are provided under `lavis/projects/sega3d/train/` and `val/`; the corresponding `nv2` files select `sega3d_t5_nv2`. The inherited `eval_topk` and `save_topk_diagnostics` fields are used by nv2; nv3 retains its original prediction implementation and does not read these fields. Model and dataset registration locations are listed in [code structure](code_structure.md).

## Dataset selection

| Dataset | Builder | Loader | Source selection |
|---|---|---|---|
| ScanNet | `sega3d_reasoning_scannet` | `reasoning_scannet_dataset.py` | `source_mode = "scannet"` |
| Matterport3D | `sega3d_reasoning_matterport` | `reasoning_matterport_dataset.py` | `source_mode = "mp3d"`; one region per sample |

The loaders select matching records from the reasoning annotations. Each record uses `scene_id` (or `scene_name`), `object_id`, and `question`; `ann_id`, `type`, and `object_name` are read when provided. Preserve the original target-ID conventions for each dataset. Use `scene####_##` ScanNet scene IDs and `<building_id>_region<index>` Matterport3D IDs.

## Local layout

```text
data/
  reasoning/
    train.json
    val.json
  scannetv2/
    train/                         # <scene_id>_reason.pth
    val/
    train_instance/                # Matching per-scene instance metadata
    val_instance/
  matterport/
    mp3d_data/                     # <building_id>_region<index>.pth
    scenes_train.txt               # Building IDs for the training split
    scenes_val.txt                 # Building IDs for the validation split
    mp3d_room_type.json             # Room-type metadata when used
  candidates/reasoning/
    scannet_semantic/              # <scene_id>_pred.npy: per-point class predictions
    matterport_instances/          # <building_id>_region<index>.pt
    matterport_semantic/           # Per-point semantic predictions as .npy
checkpoints/
  sega3d_scanrefer.pth
  sega3d_reasoning_scannet_nv3.pth
  sega3d_reasoning_matterport_nv3.pth
```

The ScanNet loader reads per-point semantic predictions from `.npy` files and constructs its candidate representation. The Matterport3D loader combines predicted instance masks (`pred_masks`, `pred_classes`, `pred_scores` in `.pt` files) with semantic candidates from `.npy` files. Prediction point order must match the corresponding scene file. These are distinct input formats from the ScanRefer dual-mask-bank loader.

Set paths under `build_info.points.storage` and `build_info.candidates` in:

- `lavis/configs/datasets/sega3d_reasoning_scannet/defaults.yaml`
- `lavis/configs/datasets/sega3d_reasoning_matterport/defaults.yaml`

The candidate keys are `scannet_semantic`, `matterport_instances`, `matterport_semantic` (Matterport3D loader), and `room_types`. Task YAML files override dataset defaults. Training and evaluation commands are listed in the [README](../README.md#reasoning-segmentation).

## Configuration and input checks

```bash
python scripts/check_setup.py --task reasoning-scannet --mode train
python scripts/check_setup.py --task reasoning-scannet --mode eval --assets
python scripts/check_setup.py --task reasoning-matterport --mode train
python scripts/check_setup.py --task reasoning-matterport --mode eval --assets
```

Reasoning task checks select nv3 by default. Add `--model nv2` to check the retained nv2 configuration.

The asset checks verify annotation selection and scene/prediction paths without loading weights. Use `--checkpoint /path/to/task_checkpoint.pth` to check a checkpoint outside the default directory.

## Language annotations

Download the reasoning annotations from the [upstream annotation archive](https://drive.google.com/file/d/1Z4kzr_4oJxTJgzILaycDuHlRYeZCXOu6/view). Extract the training and validation JSON files and save them as `data/reasoning/train.json` and `data/reasoning/val.json`. Keep their scene IDs, target instance IDs, and language descriptions unchanged:

```text
data/reasoning/
  train.json
  val.json
```

## ScanNet scenes

The upstream preprocessing checkout provides the reasoning-scene preprocessing script. From that checkout:

```bash
cd data/scannetv2
wget -O scannet200_constants.py \
  https://raw.githubusercontent.com/Pointcept/Pointcept/v1.6.1/pointcept/datasets/preprocessing/scannet/meta_data/scannet200_constants.py
python split_data.py
python prepare_data_reason.py --data_split train
python prepare_data_reason.py --data_split val
```

This creates the upstream `*_reason.pth` scene files. Use matching superpoints, instance metadata, and candidate masks when assembling the task inputs. The ScanRefer preprocessing commands and the SEGA3D data structures are described in [data.md](data.md).

## Matterport3D scenes

Obtain the dataset's `download_mp.py` script through the [Matterport3D dataset portal](https://niessner.github.io/Matterport/). The upstream raw-data download command is:

```bash
python2 download_mp.py -o /absolute/path/to/matterport --type region_segmentations
```

Extract the region files into the upstream preprocessing checkout:

```text
upstream-preprocessing/data/matterport/scans/
  <building_id>/region_segmentations/
    region0.ply
    region0.vsegs.json
    region0.semseg.json
```

Then run its scene-preparation script:

```bash
cd upstream-preprocessing/data/matterport
python process_mp3d.py
```

The resulting scene files are written under `mp3d_data/`. Pair scenes with the corresponding reasoning annotations and offline candidate banks, preserving scene identifiers and original point ordering. The base preprocessing workflow and resource links come from [the upstream project](../THIRD_PARTY_NOTICES.md).
