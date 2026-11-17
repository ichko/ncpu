#!/bin/bash
# 4-bit adder damage-robustness seed sweep (D4): trains
# scripts/train/train_4bit_adder_damage.py once per seed against the default
# damage mixture (perlin, salt&pepper, occlusion, blur, brightness/contrast,
# quantize, jitter, speckle on the input + state-side damages during rollout).
#
# Submit:
#   sbatch scripts/other/slurm_damage_seed_sweep.sh
#
# Customize via env at submit time:
#   SEEDS="1 2 3" STEPS=30000 \
#     sbatch --array=0-2%3 scripts/other/slurm_damage_seed_sweep.sh
#   DAMAGES=("perlin:60:0.8" "occlusion:3:0.8:block_size=12") \
#     sbatch scripts/other/slurm_damage_seed_sweep.sh   # overrides --damage list
#
# Run dirs: runs/D4_adder4_dmg_s<seed>_<ts>/
#SBATCH --partition=gpu-l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodelist=titan2
#SBATCH --time=06:00:00
#SBATCH --array=0-4%5
#SBATCH --output=damagesweep_%A_%a.out
#SBATCH --error=damagesweep_%A_%a.err

module load uv
cd ncpu

# Per-task venv (ARRAY_TASK_ID in the name: concurrent tasks must not
# share/delete each other's venv).
VENV_DIR="$SLURM_SUBMIT_DIR/venv-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

uv sync --active

SEEDS=(${SEEDS:-1 2 3 4 5})
STEPS="${STEPS:-30000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"

if (( SLURM_ARRAY_TASK_ID >= ${#SEEDS[@]} )); then
    deactivate
    rm -rf "$VENV_DIR"
    exit 0
fi

seed=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== array=$SLURM_ARRAY_TASK_ID seed=$seed steps=$STEPS eval_every=$EVAL_EVERY ==="

# Optional damage overrides, e.g. DAMAGES=("perlin:60:0.8" "occlusion:3:0.8")
DAMAGES=(${DAMAGES:-})
STATE_DAMAGES=(${STATE_DAMAGES:-})

cmd=(uv run python scripts/train/train_4bit_adder_damage.py
    --seed "$seed"
    --steps "$STEPS"
    --device cuda
    --eval-every "$EVAL_EVERY"
    --save-rollout-npz)
for d in "${DAMAGES[@]}"; do
    cmd+=(--damage "$d")
done
for d in "${STATE_DAMAGES[@]}"; do
    cmd+=(--state-damage "$d")
done

"${cmd[@]}"

deactivate
rm -rf "$VENV_DIR"

echo "Completed Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID!"
