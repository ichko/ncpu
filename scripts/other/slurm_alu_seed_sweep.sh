#!/bin/bash
# ALU seed sweep: 5 seeds x 600k steps = 5 jobs on SLURM.
#
# Submit:
#   sbatch scripts/other/slurm_alu_seed_sweep.sh
#
# Runs scripts/train/train_alu.py (E4).
# Run dirs: runs/E4_alu_s<seed>_<ts>/
#
# Customize via env at submit time:
#   SEEDS="1 2 3 4 5"  STEPS=600000 sbatch scripts/other/slurm_alu_seed_sweep.sh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodelist=titan2
#SBATCH --time=03:00:00
#SBATCH --array=0-4%5
#SBATCH --output=alu_sweep_%A_%a.out
#SBATCH --error=alu_sweep_%A_%a.err

module load uv
cd ncpu

VENV_DIR="$SLURM_SUBMIT_DIR/venv-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

uv sync --active

SEEDS=(${SEEDS:-1 2 3 4 5})
STEPS="${STEPS:-600000}"

task_id=$SLURM_ARRAY_TASK_ID
if (( task_id >= ${#SEEDS[@]} )); then
    deactivate
    rm -rf "$VENV_DIR"
    exit 0
fi

seed=${SEEDS[$task_id]}

echo "=== array=$SLURM_ARRAY_TASK_ID seed=$seed steps=$STEPS ==="

uv run python scripts/train/train_alu.py \
    --seed "$seed" \
    --steps "$STEPS" \
    --eval-every 1000 \
    --save-rollout-npz

deactivate
rm -rf "$VENV_DIR"

echo "Completed Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID!"