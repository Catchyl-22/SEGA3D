# Installation details


Run the following commands from the repository root on **Linux with an NVIDIA GPU**. The reference versions are taken from the original experiment environment:

| Dependency | Version |
|---|---|
| Python | 3.10 |
| PyTorch / torchvision | 2.1.1 / 0.16.1 |
| CUDA build / local CUDA toolkit | 11.8 |
| torch-scatter / torch-cluster | 2.1.2 / 1.6.3 |
| spconv-cu118 | 2.3.8 |
| Transformers | 4.51.0 |
| NumPy / SciPy | 1.24.0 / 1.15.3 |

Create the environment:

```bash
conda env create -f environment.yml
conda activate sega3d
```

Install PyTorch and matching CUDA wheels:

```bash
python -m pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 --only-binary=:all: -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
python -m pip install spconv-cu118==2.3.8
python -m pip install -r requirements.txt
```

The PyG wheel page named `torch-2.1.0+cu118` serves the PyTorch 2.1 series. Installation references: [PyTorch](https://pytorch.org/get-started/previous-versions/), [PyG](https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html), [spconv](https://github.com/traveller59/spconv).

Build the pointgroup extension on the GPU machine. A local CUDA 11.8 toolkit providing `nvcc`, a compatible C++ compiler, and sparsehash headers are required in addition to the PyTorch wheel. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libsparsehash-dev
nvcc --version
python -m pip install --no-build-isolation -e ./lavis/models/sega3d_models/lib
python -m pip check
python scripts/check_setup.py --environment
python scripts/smoke_test.py
```

The smoke test imports the actual package, checks both nv2 and nv3 exports and registrations, sets up the segmentation task, and parses all ten task configurations. It does not construct a model, load weights or datasets, execute a forward pass, or train.

After compiling the extension, also check its import with:

```bash
python scripts/smoke_test.py --native-ops
```

Sparse-backbone dependencies load when a model is constructed. Pointgroup operations load when first used. These checks do not execute CUDA kernels or replace an evaluation run with actual data and trained weights.

If CUDA is installed outside the compiler search path, set `CUDA_HOME` to the CUDA 11.8 toolkit and add `$CUDA_HOME/bin` to `PATH` before building. Use this repository's `lavis` package directly from the root; a separate LAVIS installation is unnecessary. No PTv3/FlashAttention installation is needed to consume already exported candidate banks.

Base pretrained assets follow the existing [upstream setup](../THIRD_PARTY_NOTICES.md). Their local paths are described in [data.md](data.md).

## Optional launch scripts

The README uses the direct distributed launcher. The equivalent convenience scripts are:

```bash
NPROC_PER_NODE=8 bash scripts/train_scanrefer.sh
NPROC_PER_NODE=1 bash scripts/eval_scanrefer.sh ./checkpoints/sega3d_scanrefer.pth
```

The scripts use `torchrun --standalone`, which selects a rendezvous port automatically. The direct commands use the explicit port `29501`. Both launch the existing `train.py` entry point; the task YAML determines whether training or evaluation runs.

## Setup checks

```bash
python scripts/check_setup.py --mode train
python scripts/check_setup.py --mode eval
python scripts/check_setup.py --mode train --assets
python scripts/check_setup.py --mode eval --assets --checkpoint ./checkpoints/sega3d_scanrefer.pth
```

These check configuration and asset paths; they do not run model inference.

## Common setup issues

| Message or symptom | Action |
|---|---|
| `pointgroup_ops` cannot be imported | Build the included extension in the active CUDA environment. |
| `google/dense_hash_map` header missing | Install `libsparsehash-dev` or provide the sparsehash headers. |
| CUDA extension/wheel mismatch | Check the PyTorch CUDA build, local toolkit, and extension package versions. |
| Model download unavailable | Use cached model assets or a local model directory; see [data.md](data.md). |

## Upstream scene preprocessing

For the upstream scene-preparation commands in [data.md](data.md#preparing-base-scannet-scene-files), use a separate preprocessing environment based on the upstream Python/PyTorch versions:

```bash
conda create -n sega3d-preprocess python=3.8 pytorch=1.11.0 cudatoolkit=11.3 -c pytorch
conda activate sega3d-preprocess
python -m pip install "numpy<2" pandas open3d plyfile "cmake<4"
git clone https://github.com/Karbo123/segmentator.git
cd segmentator
cmake -S csrc -B csrc/build \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
  -DPYTHON_INCLUDE_DIR="$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
  -DPYTHON_LIBRARY="$(python -c 'import os, sysconfig; print(os.path.join(sysconfig.get_config_var("LIBDIR"), sysconfig.get_config_var("LDLIBRARY")))')" \
  -DCMAKE_INSTALL_PREFIX="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
cmake --build csrc/build --parallel 4
cmake --install csrc/build
cd ..
```

These build commands use the CMake interface in the upstream [mesh segmentator](https://github.com/Karbo123/segmentator). Keep that checkout and build directory in place after installation, since it installs a symbolic link. The compiled module supplies mesh superpoints to the upstream scene-preparation utilities. Preserve the point ordering when pairing processed scenes with candidate banks.
