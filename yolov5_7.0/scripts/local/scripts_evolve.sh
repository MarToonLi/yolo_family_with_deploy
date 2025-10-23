#!/bin/bash
# ================ local ====================
# Single-GPU
python ../train.py --epochs 10 --evolve

# Multi-GPU with delay
#for i in {0..1}; do
#  sleep $((30 * i)) # 30-second delay (optional)
#  echo "Starting GPU $i..."
#  nohup python train.py --epochs 10 --cache --device $i --evolve > "evolve_gpu_$i.log" &
#done