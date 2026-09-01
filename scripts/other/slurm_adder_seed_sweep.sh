#!/bin/bash
# Adder seed sweep (E2/E3): 4-bit and 8-bit adders x 10 seeds = 20 jobs.
#
# Submit:
#   sbatch scripts/other/slurm_adder_seed_sweep.sh
#
# Runs scripts/train/train_adder.py. 4-bit uses cols1 (E2), 8-bit uses cols2
# (E3; the 2-column input layout is what train_adder.py needs for 8-bit).
# Run dirs: runs/E2_adder4_s<seed>_<ts>/ and runs/E3_adder8_s<seed>_<ts>/.
#
# Customize via env at submit time:
#   BITS="8"  SEEDS="1 2 3"  STEPS=30000 \
#     sbatch --array=0-9%3 scripts/other/slurm_adder_seed_sweep.sh
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodelist=titan2
#SBATCH --time=03:00:00
#SBATCH --array=0-19%5
#SBATCH --output=addersweep_%A_%a.out
#SBATCH --error=addersweep_%A_%a.err

module load uv
cd ncpu

# Per-task venv (see slurm_gate_seed_sweep.sh for why ARRAY_TASK_ID is in the
# name: concurrent tasks must not share/delete each other's venv).
VENV_DIR="$SLURM_SUBMIT_DIR/venv-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

uv sync --active

BITS=(${BITS:-4 8})
SEEDS=(${SEEDS:-1 2 3 4 5 6 7 8 9 10})
STEPS="${STEPS:-50000}"

need=$(( ${#BITS[@]} * ${#SEEDS[@]} ))
if (( SLURM_ARRAY_TASK_ID >= need )); then
    deactivate
    rm -rf "$VENV_DIR"
    exit 0
fi

PER_BIT=${#SEEDS[@]}                        # seeds per bit width
bits=${BITS[$((SLURM_ARRAY_TASK_ID / PER_BIT))]}
seed=${SEEDS[$((SLURM_ARRAY_TASK_ID % PER_BIT))]}

# 8-bit adders need the 2-column input layout (cols2); 4-bit uses cols1.
LAYOUT="cols1"
[[ "$bits" == "8" ]] && LAYOUT="cols2"

echo "=== array=$SLURM_ARRAY_TASK_ID bits=$bits seed=$seed steps=$STEPS layout=$LAYOUT ==="

uv run python scripts/train/train_adder.py \
    --bits "$bits" \
    --seed "$seed" \
    --steps "$STEPS" \
    --layout "$LAYOUT" \
    --device cuda

deactivate
rm -rf "$VENV_DIR"

echo "Completed Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID!"