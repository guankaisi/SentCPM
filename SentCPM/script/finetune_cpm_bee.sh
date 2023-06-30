#!/bin/bash

# CUDA_VISIBLE_DEVICES= 0,7
# torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=1  --rdzv_endpoint=localhost:12345 finetune_cpm.py \
python finetune_cpm.py \
--model-config /run/user/guankaisi/config/cpm-bee-1b.json \
--load /run/user/guankaisi/cpm-bee-1b-ckpt.pt \
--dataset data/nli_for_simcse.csv \
--epoch 2 \
--batch-size 16 \
--lr 1e-5 \
--save /run/user/guankaisi/cpm_finetune/ \
--save-name cpm-bee-1b \
--use-delta