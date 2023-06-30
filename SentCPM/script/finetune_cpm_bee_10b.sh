#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=6 --rdzv_id=1  --rdzv_endpoint=localhost:12346 finetune_cpm_10b.py \
--model-config /run/user/guankaisi/config/cpm-bee-10b.json \
--load /run/user/guankaisi/cpm-bee-10b-ckpt.pt \
--dataset data/nli_for_simcse.csv \
--epoch 2 \
--batch-size 16 \
--lr 1e-3 \
--save /run/user/guankaisi/cpm_finetune/ \
--save-name cpm-bee-10b_2 \
--use-delta