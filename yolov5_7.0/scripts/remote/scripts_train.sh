#!/bin/bash
# ================ remote ====================
# =============== Single-GPU ================
# python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve
# python train.py  --batch-size 64 --data coco.yaml --weights yolov5s.pt --device 0
# python ../../train.py       --device 0 --batch 24 --cache --imgsz 1120 
# nohup python ../../train.py --device 0 --batch 24 --epochs 800 --cache --imgsz 1120 --cos-lr --image-weights > output_train.log 2>&1 &

# 20251029
# 测试使用exp42的最佳模型，在原数据集上继续训练100个epoch，修改learning_rate和warmup看是否能提升
# nohup python ../../train_little.py --device 0 --batch 24 --epochs 100 --cache --imgsz 1120 --cos-lr --image-weights  --hyp '/root/lanyun-tmp/projects/yolo_family_with_deploy/yolov5_7.0/data/hyps/apple_3_7_hyp_evolve_20251024_1126.yaml'  > output_train.log 2>&1 &
# nohup python ../../train_little.py --device 0 --batch 24 --epochs 100 --cache --imgsz 1120 --cos-lr --image-weights  > output_train.log 2>&1 &
# nohup python ../../train_little.py --device 0 --batch 24 --epochs 100 --cache --imgsz 1120 --cos-lr --image-weights --optimizer AdamW --hyp '/root/lanyun-tmp/projects/yolo_family_with_deploy/yolov5_7.0/data/hyps/apple_3_7_hyp_evolve_20251024_1126.yaml'  > output_train.log 2>&1 &
# nohup python ../../train_little.py --device 0 --batch 24 --epochs 100 --cache --imgsz 1120 --cos-lr --image-weights --hyp '/root/lanyun-tmp/projects/yolo_family_with_deploy/yolov5_7.0/data/hyps/apple_3_7_hyp_little.yaml' > output_train.log 2>&1 &
# pid=$!   # 上一个在后台运行（&）的命令的进程 ID（PID）
# wait $pid
# echo "train_little.py - ori 已经执行完毕"

nohup python ../../train_little.py --device 0 --batch 24 --epochs 500 --cache --imgsz 1120 --cos-lr --image-weights --hyp '/root/lanyun-tmp/projects/yolo_family_with_deploy/yolov5_7.0/data/hyps/apple_3_7_hyp_evolve_20251031_0240.yaml' > output_train.log 2>&1 &
# pid=$!   # 上一个在后台运行（&）的命令的进程 ID（PID）
# wait $pid
# echo "train_little.py - lr 已经执行完毕"

# echo "train_little.py - warmup starting ..."
# nohup python ../../train_little.py --device 0 --batch 24 --epochs 100 --cache --imgsz 1120 --cos-lr --image-weights --hyp '/root/lanyun-tmp/projects/yolo_family_with_deploy/yolov5_7.0/data/hyps/apple_3_7_hyp_little_warmup0.yaml' > output_train_warmup0.log 2>&1 &
# pid=$!   # 上一个在后台运行（&）的命令的进程 ID（PID）
# wait $pid
# echo "train_little.py - warmup 已经执行完毕"

# nohup python ../../train_little.py --device 0 --batch 24 --epochs 500 --cache --imgsz 1120 --cos-lr --image-weights > output_train.log 2>&1 &

# ============== Multi-GPU with delay ================
# nohup python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1 --batch 96  > output_2.log 2>&1 &
# nohup python -m torch.distributed.run --nproc_per_node 2 train.py --device 0 --batch -1 --cache --imgsz 640 > output_1.log 2>&1 &
# python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1 --batch 32 --imgsz 1120 > output_2.log 2>&1 &