#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  train.py --cfg-path lavis/projects/sega3d/train/sega3d-scanrefer-nv2.yaml "$@"
