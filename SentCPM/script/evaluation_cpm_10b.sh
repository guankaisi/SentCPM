#!/bin/bash
CUDA_VISIBLE_DEVICES=5,6 torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1 --rdzv_endpoint=localhost:12365 evaluation_cpm.py \
--model_name cpm-bee-10b-ckpt.pt \
--config_name config/cpm-bee-10b.json \
--delta_name cpm_finetune/cpm-bee-10b-delta.pt \
--pooler avg \
--task_set transfer \
--mode test \