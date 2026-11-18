#!/bin/bash
# Gate x damage x seed sweep (D5): can the NCA learn every gate while being
# harassed by each noise damage separately?
#
# One SLURM array task = one (gate, damage profile, seed) running
# scripts/train/train_gate_damage.py. Each damage kind gets its own profile
# (single-damage ablation), plus "clean" (no damage) and "mix" (full default
# damage set, one sampled per step).
#
# Run dirs: runs/D5_<gate>_ni2_<damage-slug>_s<seed>_<ts>/
# Analyze afterwards with:
#   uv run python scripts/analyze/analyze_damage_sweep.py
#
# Submit:
#   sbatch scripts/other/slurm_gate_damage_sweep.sh
#
# Customize via env at submit time (adjust --array / %N to match
# len(GATES) * len(PROFILES) * len(SEEDS)):
#   GATES="AND XOR" PROFILES="clean perlin" SEEDS="1 2 3" STEPS=30000 \
#     sbatch --array=0-11%3 scripts/other/slurm_gate_damage_sweep.sh
#SBATCH --partition=gpu-m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --nodelist=h32
#SBATCH --time=06:00:00
#SBATCH --array=0-449%10
#SBATCH --output=gatedmgsweep_%A_%a.out
#SBATCH --error=gatedmgsweep_%A_%a.err

module load uv
cd ncpu

# Per-task venv (ARRAY_TASK_ID in the name: concurrent tasks must not
# share/delete each other's venv).
VENV_DIR="$SLURM_SUBMIT_DIR/venv-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

uv sync --active

GATES=(${GATES:-AND OR XOR NAND NOR XNOR})
SEEDS=(${SEEDS:-1 2 3 4 5})
STEPS="${STEPS:-30000}"
EVAL_EVERY="${EVAL_EVERY:-1000}"
ALL_PROFILES="clean gaussian perlin salt_pepper occlusion blur brightness_contrast quantize jitter speckle state_gaussian state_perlin channel_dropout block_dropout mix"
PROFILES=(${PROFILES:-$ALL_PROFILES})

# ---- build the job list: "gate profile seed" per array task ----
JOBS=()
for g in "${GATES[@]}"; do
    for p in "${PROFILES[@]}"; do
        for s in "${SEEDS[@]}"; do
            JOBS+=("$g $p $s")
        done
    done
done

if (( SLURM_ARRAY_TASK_ID >= ${#JOBS[@]} )); then
    deactivate
    rm -rf "$VENV_DIR"
    exit 0
fi

read -r gate profile seed <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

# Each profile = exactly one damage (input side via --damage, state side via
# --state-damage; "none" disables the other side). "clean" = no damages,
# "mix" = script defaults (full damage set, sampled per step).
args=(--gate "$gate" --seed "$seed" --steps "$STEPS" --device cuda
      --eval-every "$EVAL_EVERY" --save-rollout-npz)
case "$profile" in
    clean)              args+=(--damage none --state-damage none) ;;
    gaussian)           args+=(--damage "gaussian:20:0.8" --state-damage none) ;;
    perlin)             args+=(--damage "perlin:60:0.8" --state-damage none) ;;
    salt_pepper)        args+=(--damage "salt_pepper:0.1:1.0" --state-damage none) ;;
    occlusion)          args+=(--damage "occlusion:3:0.8:block_size=12,fill=0" --state-damage none) ;;
    blur)               args+=(--damage "blur:2.0:0.8" --state-damage none) ;;
    brightness_contrast) args+=(--damage "brightness_contrast:0.3:0.8" --state-damage none) ;;
    quantize)           args+=(--damage "quantize:1:0.8:levels=4" --state-damage none) ;;
    jitter)             args+=(--damage "jitter:2:0.8" --state-damage none) ;;
    speckle)            args+=(--damage "speckle:0.3:0.8" --state-damage none) ;;
    state_gaussian)     args+=(--damage none --state-damage "state_gaussian:0.8:0.8") ;;
    state_perlin)       args+=(--damage none --state-damage "state_perlin:1.0:0.8") ;;
    channel_dropout)    args+=(--damage none --state-damage "channel_dropout:0.2:0.8") ;;
    block_dropout)      args+=(--damage none --state-damage "block_dropout:2:0.8:block_size=8") ;;
    mix)                ;;
    *) echo "unknown profile: $profile"; deactivate; rm -rf "$VENV_DIR"; exit 1 ;;
esac

echo "=== array=$SLURM_ARRAY_TASK_ID gate=$gate profile=$profile seed=$seed steps=$STEPS ==="

uv run python scripts/train/train_gate_damage.py "${args[@]}"

deactivate
rm -rf "$VENV_DIR"

echo "Completed Job $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID!"
