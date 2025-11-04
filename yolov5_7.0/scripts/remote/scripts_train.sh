#!/bin/bash
# ================ remote ====================
# =============== Single-GPU ================
# nohup python ../../train.py --device 0 --batch 24 --epochs 800 --cache --imgsz 1120 --cos-lr --image-weights > output_train.log 2>&1 &

# nohup python ../../train_little.py --device 0 --batch 24 --epochs 100 --cache --imgsz 1120 --cos-lr --image-weights --hyp '/root/lanyun-tmp/projects/yolo_family_with_deploy/yolov5_7.0/data/hyps/apple_3_7_hyp_little.yaml' > output_train.log 2>&1 &
# pid=$!   # 上一个在后台运行（&）的命令的进程 ID（PID）
# wait $pid
# echo "train_little.py - ori 已经执行完毕"

# ============== Multi-GPU with delay ================
# nohup python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1 --batch 96  > output_2.log 2>&1 &
# nohup python -m torch.distributed.run --nproc_per_node 2 train.py --device 0 --batch -1 --cache --imgsz 640 > output_1.log 2>&1 &
# python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1 --batch 32 --imgsz 1120 > output_2.log 2>&1 &