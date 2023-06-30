#!/bin/bash
CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=1 --rdzv_id=1 --rdzv_endpoint=localhost:12345 evaluation_cpm.py \
--model_name cpm-bee-1b-ckpt.pt \
--config_name config/cpm-bee-1b.json \
--delta_name cpm_finetune/cpm-bee-1b-delta4.pt \
--task_set transfer \
--mode test \