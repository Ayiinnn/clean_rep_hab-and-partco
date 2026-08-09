#!/usr/bin/env bash
set -e

HAB_ROOT=/data/projects/clean_projects/hab
cd "$HAB_ROOT"

CUDA_VISIBLE_DEVICES=${1:-0} /data/miniconda3/envs/hypcd/bin/python -u \
  -m train.train_hab_v2 \
  --dataset_name cub \
  --model_name v2 \
  --weight_decay 5e-5 \
  --memax_weight 1.0 \
  --c 0.1 \
  --cr 1.2 \
  --hyper_temp_scale 0.4 \
  --deterministic \
  --part_unsup_scale 0.0625 \
  --exp_root "$HAB_ROOT/dev_outputs" \
  --exp_name S006