# ScanRefer data and checkpoints

Default paths are relative to the repository root:

```text
data/
  annotations/scanrefer/
    train_with_cameras.json
    val_with_cameras.json
  scannetv2/
    train_sp_refined/               # *_refer.pth processed scene files
    val_sp_refined/
    train_sp_refined_instance/      # Instance metadata used by the loader
    val_sp_refined_instance/
  candidates/
    scannet20/                     # <scene_id>.pt
    scannet200/                    # <scene_id>.pt
    whitelist_standard.txt
checkpoints/
  sp_unet_backbone.pth
  blip2_pretrained_flant5xl.pth
  sega3d_scanrefer.pth
```

Use the camera-merged `train_with_cameras.json` and `val_with_cameras.json` specified by the original nv2 configuration. Each annotation contains `scene_id`, `object_id`, `ann_id`, `object_name`, `description`, and `camera`. Camera dictionaries contain `position` and `lookat`, each a three-component vector. Preserve the annotation records and their scene/object identifiers.

Each offline candidate file contains `pred_masks` with shape [number of candidates, number of original scene points] and `pred_classes` with one label per candidate. `pred_scores` is used when present. Candidate masks must use the same original point ordering as the processed scenes.

The loader combines the ScanNet20 and ScanNet200 banks, applies the configured category whitelist to the ScanNet200 side, and removes near-duplicate masks using the original merge rule. The whitelist is a category set for the candidate source, rather than a query-specific predicted target category.

Configure the candidate directories and whitelist under `build_info.candidates` in `lavis/configs/datasets/sega3d_scanrefer/defaults.yaml`. They can also be overridden under the same dataset key in a task configuration. Both candidate files must exist for each validation scene; a missing bank raises an error.

Base pretrained assets follow the upstream setup linked in [third-party notices](../THIRD_PARTY_NOTICES.md). Keep the original point-encoder and initialization checkpoint files and configure their locations in the model configuration. The point-encoder checkpoint and BLIP2 initialization checkpoint are used for training initialization. Evaluation additionally requires the task-specific trained SEGA3D checkpoint. The underlying T5 model identifier is `google/flan-t5-xl`.

For PTv3 model training and inference, refer to [Pointcept](https://github.com/Pointcept/Pointcept) and [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3). The tuple-based scenes and per-instance candidate banks above are the inputs consumed by SEGA3D.

## Candidate-bank and instance-metadata fields

For every scene, save the two candidate dictionaries under `data/candidates/scannet20/<scene_id>.pt` and `data/candidates/scannet200/<scene_id>.pt`:

```python
{
    "pred_masks": masks,      # Bool/0-1 array or tensor, shape [M, N]
    "pred_classes": labels,   # Integer array or tensor, shape [M]
    "pred_scores": scores,   # Optional float array or tensor, shape [M]
}
```

The loader indexes the mask columns using the original point indices retained by scene cropping. Consequently, keep all original point columns in each saved candidate mask. Set the whitelist file to the category-name set used for the ScanNet200 bank, one category name per line.

Per-scene instance metadata is loaded from the corresponding `*_sp_refined_instance/` directory. Dictionary fields are `inst2p_offsets`, `inst2p_indices`, and optional `inst_bbox` / `inst_sem_mode`. For `I` instances, `inst2p_offsets` has length `I+1`; `inst2p_indices` stores the concatenated point indices. Use the metadata matching each processed scene.

## Preparing base ScanNet scene files

Clone the upstream preprocessing project linked in [third-party notices](../THIRD_PARTY_NOTICES.md) into `upstream-preprocessing/`. Place the ScanNet raw `scans/` directory under its `data/scannetv2/`, then run:

```bash
cd upstream-preprocessing/data/scannetv2
python split_data.py
python prepare_data_refer.py --data_split train
python prepare_data_refer.py --data_split val
```

These utilities require their preprocessing dependencies, including Open3D, plyfile, and the mesh segmentator. An installation example is provided in [installation.md](installation.md#upstream-scene-preprocessing). They generate base `*_refer.pth` tuples in the upstream `train/` and `val/` directories. SEGA3D's default paths select the `train_sp_refined/` and `val_sp_refined/` scene sets with their matching superpoints and instance metadata; keep these assets together in the directory layout above. The upstream also provides downloadable base [training scenes](https://drive.google.com/file/d/1Y41Y6H0To9qB71kUlLISYn4RhvwFe4KZ/view) and [validation scenes](https://drive.google.com/file/d/1y9MSXFGh80W46201bbgCoW95go5DJY7k/view).

## Assets and configuration fields

| Asset | Configuration field |
|---|---|
| Processed data root containing `scannetv2/` | `datasets.sega3d_scanrefer.build_info.points.storage` |
| Training/validation annotations | `datasets.sega3d_scanrefer.build_info.annotations.train.storage` / `.val.storage` |
| ScanNet20/ScanNet200 candidate directories | `datasets.sega3d_scanrefer.build_info.candidates.scannet20` / `.scannet200` |
| Experiment whitelist | `datasets.sega3d_scanrefer.build_info.candidates.whitelist` |
| Point encoder | `model.point_encoder_cfg.pretrained` |
| Initialization checkpoint for training | `model.pretrained` |
| FLAN-T5 model/tokenizer directory or model ID | `model.t5_model` |
| Trained SEGA3D checkpoint for evaluation | `model.finetuned` |

Candidate paths are defined in the dataset defaults and can be overridden in the task YAML. Model initialization defaults are defined in `lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml`. The task YAML has priority over defaults.

Use the original base assets from the [upstream instructions](../THIRD_PARTY_NOTICES.md); the filenames above specify this repository's default local layout. Checkpoints may retain their downloaded names when the configured paths match. Training additionally produces a task-specific SEGA3D checkpoint, which is separate from those base assets.

The implementation initializes BERT using `bert-base-uncased` and FLAN-T5 using the configured model identifier or directory. On an offline machine, populate the Hugging Face cache beforehand and set `HF_HUB_OFFLINE=1`; alternatively, point `model.t5_model` to the existing local FLAN-T5 model directory. BERT config/weights must still be available in the Hugging Face cache. The first initialization with an empty cache requires network access.

Processed scene files are loaded as `(xyz, rgb, superpoint, semantic_label, instance_label)`. Keep consistent coordinate/color conventions and point ordering across the scene files and candidate masks. Configure the experiment category whitelist using `build_info.candidates.whitelist`.
