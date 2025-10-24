#!/bin/bash
# ================ remote ====================
# =============== Single-GPU ================
python ../../train.py       --device 0 --batch 24 --imgsz 1120 --epochs 10  --evolve
#nohup python ../../train.py --device 0 --batch 24 --imgsz 1120 --epochs 10  --evolve > output_evolve.log 2>&1 &
# 远程时似乎不适合使用cache，会占用大量内存，导致训练失败

# ============== Multi-GPU with delay ================
#for i in {0..1}; do
#  sleep $((30 * i)) # 30-second delay (optional)
#  echo "Starting GPU $i..."
#  nohup python train.py --epochs 10 --cache --device $i --evolve > "evolve_gpu_$i.log" &
#done