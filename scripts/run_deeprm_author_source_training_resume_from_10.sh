#!/usr/bin/env bash
set -u

echo "START_TS=$(date --iso-8601=seconds)"
echo "RUN_LABEL=author_source_aligned"
echo "RESUME_FROM=results/checkpoints/author_source_aligned/load_0.7/policy_iter_10.pt"
echo "COMMAND=cisose-deeprm train --author-source --load 0.7 --iterations 1000 --num-jobsets 100 --rollouts-per-jobset 20 --episode-horizon 200 --max-episode-steps 800 --checkpoint-interval 10 --eval-interval 10 --train-end all-done --rollout-workers 8 --resume-from results/checkpoints/author_source_aligned/load_0.7/policy_iter_10.pt"

/usr/bin/time -p /usr/bin/env \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES= \
  .venv/bin/cisose-deeprm train \
    --author-source \
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
    --resume-from results/checkpoints/author_source_aligned/load_0.7/policy_iter_10.pt \
    --run-label author_source_aligned \
    --run-name deeprm-author-source-lambda-0.7-aligned-resume-from-10

code=$?
echo "END_TS=$(date --iso-8601=seconds)"
echo "EXIT_CODE=${code}"
exit "${code}"
