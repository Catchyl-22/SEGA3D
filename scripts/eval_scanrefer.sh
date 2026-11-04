#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="${1:-./checkpoints/sega3d_scanrefer.pth}"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then shift; fi
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  train.py --cfg-path lavis/projects/sega3d/val/sega3d-scanrefer-nv2.yaml \
  --options "model.finetuned=$CHECKPOINT" "model.eval_topk=4" "$@"
