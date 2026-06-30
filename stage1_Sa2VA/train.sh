#!/usr/bin/env bash
export NCCL_P2P_DISABLE=1
set -x
PORT=$((28505 + $RANDOM % 2000))

echo "Using launch mode."
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.5" \
  --master_port=${PORT} \
  --nproc_per_node=8 \
  tools/train.py projects/llava_sam2/configs/sa2va_4b.py \
  --launcher pytorch \
  --deepspeed deepspeed_zero2

