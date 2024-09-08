#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main_ddp.py \
--config config/nturgbd120-cross-subject/joint.yaml \
--work-dir YOUR_WORK_DIR \
--sync_bn False \
--phase test \
--save-score True \
--weights YOUR_CHECKPOINT_PATH