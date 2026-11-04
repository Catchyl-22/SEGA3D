<div align="center">

# SEGA3D

### Segment and Select: Vision-Language Segmentation in 3D Scenarios

**Yulin Chen, Zhihang Zhong, Yuenan Hou**  
**NeurIPS 2026**

</div>

## Overview

SEGA3D combines offline segmentation candidates, language-conditioned selection, and loopback verification for 3D vision-language segmentation. This repository provides the core model components, ScanRefer training and evaluation, and reasoning segmentation on Reason3D.

Our workflow starts with referring segmentation on **ScanRefer**, followed by reasoning segmentation on **Reason3D**. ScanRefer uses `sega3d_t5_nv2`; reasoning segmentation uses `sega3d_t5_nv3`.

## Installation

Use Linux with an NVIDIA GPU and a CUDA 11.8 toolkit. The dependency versions below follow the original experiment environment.

```bash
git clone https://github.com/Catchyl-22/SEGA3D.git
cd SEGA3D

conda env create -f environment.yml
conda activate sega3d

python -m pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 --only-binary=:all: -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install spconv-cu118==2.3.8
python -m pip install -r requirements.txt
```

Build the point-cloud extension:

```bash
sudo apt-get install -y build-essential libsparsehash-dev
cd lavis/models/sega3d_models/lib
python setup.py develop
cd ../../../..
```

The extension build uses `nvcc` from the local CUDA 11.8 toolkit. If needed, set `CUDA_HOME` to that toolkit and add `$CUDA_HOME/bin` to `PATH`. Use the `lavis` package included in this repository. Further environment checks are described in [installation details](docs/installation.md).

## Data preparation

### ScanRefer and ScanNet

Obtain the ScanNet v2 scenes through the [ScanNet dataset portal](http://www.scan-net.org/). Use the camera-merged ScanRefer annotations `train_with_cameras.json` and `val_with_cameras.json` used by the nv2 configuration. Each annotation contains `scene_id`, `object_id`, `ann_id`, `object_name`, `description`, and `camera`; camera entries use `position` and `lookat`. Organize the processed scenes, instance metadata, and candidate banks as follows:

```text
SEGA3D/
├── data/
│   ├── annotations/scanrefer/
│   │   ├── train_with_cameras.json
│   │   └── val_with_cameras.json
│   ├── scannetv2/
│   │   ├── train_sp_refined/           # <scene_id>_refer.pth
│   │   ├── val_sp_refined/
│   │   ├── train_sp_refined_instance/  # Per-scene instance metadata
│   │   └── val_sp_refined_instance/
│   └── candidates/
│       ├── scannet20/                 # <scene_id>.pt
│       ├── scannet200/                # <scene_id>.pt
│       └── whitelist_standard.txt
└── checkpoints/
    ├── sp_unet_backbone.pth
    ├── blip2_pretrained_flant5xl.pth
    └── sega3d_scanrefer.pth
```

From the SEGA3D root, copy the downloaded annotations into the default locations:

```bash
SCANREFER_ANNOTATIONS=/absolute/path/to/ScanRefer_cam_merged
mkdir -p data/annotations/scanrefer
cp "$SCANREFER_ANNOTATIONS/train_with_cameras.json" data/annotations/scanrefer/train_with_cameras.json
cp "$SCANREFER_ANNOTATIONS/val_with_cameras.json" data/annotations/scanrefer/val_with_cameras.json
```

Scene files contain `(xyz, rgb, superpoint, semantic_label, instance_label)`. Each candidate file contains instance masks `pred_masks` of shape `[M, N]`, class labels `pred_classes` of shape `[M]`, and optional confidence scores `pred_scores`. Here, `N` is the original scene point count and `M` is the number of candidates. Keep the point ordering consistent between scenes and candidate masks. The two candidate sources are fused by the included dataset loader. See [data formats and preprocessing](docs/data.md) for the field definitions.

### Model initialization

Place the point-encoder weight `sp_unet_backbone.pth` under `checkpoints/`; this backbone is distributed through [SPFormer](https://github.com/sunjiahao1999/SPFormer) and [3D-STMN](https://github.com/sosppxo/3D-STMN). Download the BLIP2 initialization weight with:

```bash
mkdir -p checkpoints
wget -O checkpoints/blip2_pretrained_flant5xl.pth \
  https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xl.pth
```

The implementation loads FLAN-T5 and BERT using the model identifiers in the code/configuration and the Hugging Face cache. Set `model.t5_model` to your existing local model directory when using local assets. Evaluation loads the task-specific trained SEGA3D checkpoint through `model.finetuned`.

Set data and model paths in these configuration files:

| Configuration | Contents |
|---|---|
| `lavis/projects/sega3d/train/sega3d-scanrefer-nv2.yaml` | nv2 training; `sega3d_scanrefer`; camera-merged annotations |
| `lavis/projects/sega3d/val/sega3d-scanrefer-nv2.yaml` | nv2 evaluation; same dataset; trained checkpoint |
| `lavis/configs/datasets/sega3d_scanrefer/defaults.yaml` | Processed scenes, candidate banks, and category whitelist |
| `lavis/configs/models/blip2/blip2_pretrain_flant5xl.yaml` | Base model initialization |

For PTv3 model training and inference, refer to the official [Pointcept](https://github.com/Pointcept/Pointcept) and [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) projects. SEGA3D reads the offline candidate banks described above.

## Training

Return to the SEGA3D repository root and activate its environment. Train the ScanRefer model with:

```bash
conda activate sega3d
python -m torch.distributed.run --nproc_per_node=8 --master_port=29501 \
  train.py --cfg-path lavis/projects/sega3d/train/sega3d-scanrefer-nv2.yaml
```

Training and evaluation both use `arch: sega3d_t5_nv2` and the `sega3d_scanrefer` dataset builder. The training configuration uses 100 epochs and batch size 24 per GPU. Outputs are saved under `outputs/scanrefer_train/<job_id>/`.

## Evaluation

Evaluate a trained ScanRefer checkpoint with:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 \
  train.py --cfg-path lavis/projects/sega3d/val/sega3d-scanrefer-nv2.yaml \
  --options model.finetuned=./checkpoints/sega3d_scanrefer.pth
```

Replace the checkpoint path with your trained model. The evaluation configuration sets `evaluate: true` and runs candidate selection, mask refinement, and reranking. It reports mIoU, Acc@0.25, and Acc@0.50; logs are saved under `outputs/scanrefer_eval/<job_id>/`.

## Reasoning segmentation

After ScanRefer training, use the resulting checkpoint to initialize SEGA3D for reasoning segmentation on **Reason3D**.

| Dataset | Dataset builder | Training configuration |
|---|---|---|
| Reason3D | `sega3d_reasoning_scannet` | `lavis/projects/sega3d/train/sega3d-scannet-nv3.yaml` |

Place the Reason3D annotations at `data/reasoning/train.json` and `data/reasoning/val.json`. Prepare the processed scenes and offline semantic predictions as described in the [data guide](docs/reasoning.md#scannet-scenes), and set their paths in `lavis/configs/datasets/sega3d_reasoning_scannet/defaults.yaml`. Train with:

```bash
python -m torch.distributed.run --nproc_per_node=8 --master_port=29501 \
  train.py --cfg-path lavis/projects/sega3d/train/sega3d-scannet-nv3.yaml \
  --options model.finetuned=./checkpoints/sega3d_scanrefer.pth
```

Evaluate the trained Reason3D checkpoint with:

```bash
python -m torch.distributed.run --nproc_per_node=1 --master_port=29501 \
  train.py --cfg-path lavis/projects/sega3d/val/sega3d-scannet-nv3.yaml \
  --options model.finetuned=./checkpoints/sega3d_reasoning_scannet_nv3.pth
```

Replace the checkpoint path with your trained nv3 weights. Training outputs are saved under `outputs/sega3d_reasoning_scannet_train/`; evaluation outputs are saved under `outputs/sega3d_reasoning_scannet_nv3_eval/`.

## Code structure

The central registration table is [lavis/sega3d_registry.py](lavis/sega3d_registry.py). See [component registration and configuration](docs/code_structure.md) for model, dataset, and task names, their registration locations, and the configuration-to-code lookup flow.

## Acknowledgements

Supported by Shanghai Artificial Intelligence Laboratory.

Our implementation builds on the upstream project credited in [third-party notices](THIRD_PARTY_NOTICES.md), [LAVIS](https://github.com/salesforce/LAVIS), and [Pointcept](https://github.com/Pointcept/Pointcept). We thank the authors for sharing their work. See [third-party notices](THIRD_PARTY_NOTICES.md) and [licenses](licenses/) for the included license texts.
