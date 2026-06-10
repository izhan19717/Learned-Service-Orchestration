#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="/usr/bin:/bin:/usr/local/bin:$HOME/.local/bin"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

LOG_DIR="$ROOT/logs/training"
mkdir -p "$LOG_DIR"

latest_checkpoint() {
  local label="$1"
  local ckpt_dir="$ROOT/results/checkpoints/$label/load_0.7"
  if [[ ! -d "$ckpt_dir" ]]; then
    return 0
  fi
  find "$ckpt_dir" -maxdepth 1 -type f -name 'policy_iter_*.pt' \
    | sed -E 's/.*policy_iter_([0-9]+)\.pt$/\1 &/' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

run_condition() {
  local visible_slots="$1"
  local label="experiment_c_m${visible_slots}_author_source"
  local log_file="$LOG_DIR/experiment_c_m${visible_slots}_training.out"
  local final_ckpt="$ROOT/results/checkpoints/$label/load_0.7/policy_final.pt"

  {
    echo "START_TS=$(date --iso-8601=seconds)"
    echo "RUN_LABEL=$label"
    echo "VISIBLE_SLOTS=$visible_slots"
    echo "SCIENTIFIC_PROTOCOL=Experiment C action-space ablation, author-source DeepRM settings"
  } >> "$log_file"

  if [[ -f "$final_ckpt" ]]; then
    {
      echo "FINAL_CHECKPOINT_EXISTS=$final_ckpt"
      echo "SKIP_TS=$(date --iso-8601=seconds)"
      echo "EXIT_CODE=0"
    } >> "$log_file"
    return 0
  fi

  local resume_args=()
  local resume_ckpt
  resume_ckpt="$(latest_checkpoint "$label" || true)"
  if [[ -n "$resume_ckpt" ]]; then
    resume_args=(--resume-from "$resume_ckpt")
    echo "RESUME_FROM=${resume_ckpt#$ROOT/}" >> "$log_file"
  else
    echo "RESUME_FROM=none" >> "$log_file"
  fi

  set +e
  {
    echo "COMMAND=cisose-deeprm train --author-source --visible-slots $visible_slots --load 0.7 --iterations 1000 --num-jobsets 100 --rollouts-per-jobset 20 --episode-horizon 200 --max-episode-steps 800 --checkpoint-interval 10 --eval-interval 10 --train-end all-done --rollout-workers 8 --run-label $label ${resume_args[*]}"
    /usr/bin/time -p /usr/bin/env \
      CUDA_VISIBLE_DEVICES= \
      .venv/bin/cisose-deeprm train \
      --author-source \
      --visible-slots "$visible_slots" \
      --load 0.7 \
      --iterations 1000 \
      --num-jobsets 100 \
      --rollouts-per-jobset 20 \
      --episode-horizon 200 \
      --max-episode-steps 800 \
      --checkpoint-interval 10 \
      --eval-interval 10 \
      --train-end all-done \
      --rollout-workers 8 \
      --run-label "$label" \
      --run-name "experiment-c-m${visible_slots}-author-source-training" \
      "${resume_args[@]}"
    status=$?
    echo "END_TS=$(date --iso-8601=seconds)"
    echo "EXIT_CODE=$status"
  } >> "$log_file" 2>&1
  set -e
  return "$status"
}

run_condition 3
run_condition 1
