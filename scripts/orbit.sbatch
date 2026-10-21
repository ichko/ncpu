#!/bin/bash
# short,unite: take whichever partition frees a slot first. NOT a40 — node1's
# GPU 1 is ECC-dead and GPU 0 is held by another user until 2026-08-08, so an
# a40 allocation can only land on the broken card. Override with
#   sbatch -p a40 --gres=gpu:a40:1 scripts/orbit.sbatch ...
# once that node is healthy again.
#SBATCH -p short,unite
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -J orbit
#SBATCH -o orbit_%j.out
#SBATCH -e orbit_%j.out
#SBATCH --requeue
set -e
cd "$HOME/ncpu"
echo "node: $(hostname)  restart=${SLURM_RESTART_COUNT:-0}  gpu:"; nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# node1 GPU 1 has thousands of uncorrectable ECC errors and cannot run CUDA at
# all. Slurm hands it out whenever GPU 0 is busy, so bounce off it and requeue
# instead of burning the run. Give up after ~4h of retries.
#
# Note: nvidia-smi inside the job lists EVERY GPU on the node, not just the
# allocated one, so parsing its first row reads the wrong card. Probe CUDA
# through torch instead — that tests exactly what the run needs.
nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" \
  --query-gpu=index,uuid,ecc.errors.uncorrected.volatile.total \
  --format=csv,noheader || true
if ! .venv/bin/python -c "import torch; torch.zeros(8, device='cuda').sum().item()"; then
  echo "allocated GPU cannot run CUDA — not training on it"
  if [ "${SLURM_RESTART_COUNT:-0}" -ge 48 ]; then
    echo "FATAL: gave up after ${SLURM_RESTART_COUNT} retries waiting for a healthy a40"
    exit 1
  fi
  sleep 300
  scontrol requeue "$SLURM_JOB_ID"
  exit 1
fi

# .venv/bin/python directly: the login node lacks libGL (cv2), compute nodes have it
.venv/bin/python scripts/train_orbit.py --device cuda --seed 0 "$@"
