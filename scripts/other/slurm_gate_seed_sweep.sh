#!/bin/bash
# Gate seed sweep (E1): all 8 gates x 10 seeds = 80 jobs on SLURM.
#
# Submit:
#   sbatch scripts/other/slurm_gate_seed_sweep.sh
#
# Runs the unified scripts/train/train_gate.py (this replaced the old
# train_gates_noise_robust.py). With --gaussian-noise / fire-rate left at 0 the
# script trains the plain E1 recipe (kernel 5, alive 0.0) and periodically
# evaluates robustness, so run dirs are tagged "_robust" (e.g.
# runs/E1_XOR_ni2_robust_s1_<ts>/).
#
# Customize via env at submit time (must adjust --array / %N to match):
#   GATES="AND XOR"  SEEDS="1 2 3"  STEPS=30000 \
#     sbatch --array=0-5%3 scripts/other/slurm_gate_seed_sweep.sh
#
# NOTE: the flags are --kernel_size / --alive_threshold (underscores) in the
# new script; kernel 7 / alive 0.1 (the old noise-robust recipe) would be:
#   ... train_gate.py ... --kernel_size 7 --alive_threshold 0.1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodelist=titan2
#SBATCH --time=03:00:00
#SBATCH --array=0-79%5
#SBATCH --output=sweep_%A_%a.out
#SBATCH --error=sweep_%A_%a.err

module load uv
cd ncpu

# Per-task venv: include ARRAY_TASK_ID so concurrent tasks don't race on (or
# delete) each other's venv mid-run (the template's venv-$JOB_ID was shared).
VENV_DIR="$SLURM_SUBMIT_DIR/venv-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

uv sync --active

GATES=(${GATES:-AND OR XOR NAND NOR XNOR half_adder majority3})
SEEDS=(${SEEDS:-1 2 3 4 5 6 7 8 9 10})
STEPS="${STEPS:-20000}"

need=$(( ${#GATES[@]} * ${#SEEDS[@]} ))
if (( SLURM_ARRAY_TASK_ID >= need )); then
    deactivate
    rm -rf "$VENV_DIR"
    exit 0
fi

PER_GATE=${#SEEDS[@]}                        # seeds per gate
gate=${GATES[$((SLURM_ARRAY_TASK_ID / PER_GATE))]}
seed=${SEEDS[$((SLURM_ARRAY_TASK_ID % PER_GATE))]}

echo "=== array=$SLURM_ARRAY_TASK_ID gate=$gate seed=$seed steps=$STEPS ==="

uv run python scripts/train/train_gate.py \
    --gate "$gate" \
    --seed "$seed" \
    --steps "$STEPS" \
    --eval-every 1000

deactivate
rm -rf "$VENV_DIR"

echo "Completed Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID!"