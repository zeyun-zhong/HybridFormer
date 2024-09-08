#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main_ddp.py \
--config config/nturgbd120-cross-subject/joint.yaml \
--work-dir work_dir/ntu120/csub/base \
--base-lr 0.05 \
--batch-size 64 \
--print-log False \
--max_norm 3. \
--use_mixup True \
--mixup_prob 0. \
--sync_bn False \
--start_eval 120