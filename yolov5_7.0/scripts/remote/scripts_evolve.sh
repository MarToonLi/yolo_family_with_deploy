#!/bin/bash
# ================ remote ====================

# =============== Single-GPU ================
python ../../train.py       --device 0 --batch 24 --cache --imgsz 1120 --epochs 10  --evolve
#nohup python ../../train.py --device 0 --batch 24 --cache --imgsz 1120 --epochs 10  --evolve > output_train.log 2>&1 &


# ============== Multi-GPU with delay ================
#for i in {0..1}; do
#  sleep $((30 * i)) # 30-second delay (optional)
#  echo "Starting GPU $i..."
#  nohup python train.py --epochs 10 --cache --device $i --evolve > "evolve_gpu_$i.log" &
#done