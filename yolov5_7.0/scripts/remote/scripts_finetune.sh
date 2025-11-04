#!/bin/bash
# ================ remote ====================
# =============== Single-GPU ================ 
# 1) train configuration + freeze10 + low lr + stay warmup
# 2) train configuration + low lr + stay warmup
# nohup python ../../train_little.py --device 0 --batch 24 --epochs 80 --cache --imgsz 1120 --cos-lr  --image-weights  --freeze 10  > output_finetune.log 2>&1 &

# ============== Multi-GPU with delay ================
# nohup python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1 --batch 96  > output_2.log 2>&1 &
# nohup python -m torch.distributed.run --nproc_per_node 2 train.py --device 0 --batch -1 --cache --imgsz 640 > output_1.log 2>&1 &
# python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1 --batch 32 --imgsz 1120 > output_2.log 2>&1 &