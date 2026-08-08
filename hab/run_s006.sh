#!/usr/bin/env bash
set -e

cd /data/projects/HypCD
GPU=${1:-0}

CUDA_VISIBLE_DEVICES=$GPU /data/miniconda3/envs/hypcd/bin/python -u -m train.train_hab_v2 \
  --dataset_name cub --model_name v2 --batch_size 128 --grad_from_block 11 \
  --epochs 200 --num_workers 8 --use_ssb_splits --sup_weight 0.35 \
  --weight_decay 5e-5 --transform imagenet --lr 0.1 --eval_funcs v2 v2b \
  --warmup_teacher_temp 0.07 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 \
  --memax_weight 1.0 --c 0.1 --cr 1.2 --hyper_temp_scale 0.4 \
  --hyper_start_epoch 0 --hyper_end_epoch 200 --hyper_max_weight 1.0 \
  --deterministic --seed 0 --part_unsup_scale 0.0625 \
  --exp_name hab-S006-cub-200e-seed0
