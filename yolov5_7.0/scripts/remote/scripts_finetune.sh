#!/bin/bash
# ================ remote ====================
# =============== Single-GPU ================
# python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve
# python train.py  --batch-size 64 --data coco.yaml --weights yolov5s.pt --device 0
# python ../../train.py       --device 0 --batch 24 --cache --imgsz 1120 
# nohup python ../../train_sgdm.py --device 0 --batch 24 --epochs 200 --cache --imgsz 1120 --cos-lr --image-weights --optimizer SGD  > output_train.log 2>&1 &
nohup python ../../train_little.py --device 0 --batch 24 --epochs 80 --cache --imgsz 1120 --cos-lr  --image-weights  --freeze 10  > output_finetune.log 2>&1 &

# ============== Multi-GPU with delay ================
# nohup python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1 --batch 96  > output_2.log 2>&1 &
# nohup python -m torch.distributed.run --nproc_per_node 2 train.py --device 0 --batch -1 --cache --imgsz 640 > output_1.log 2>&1 &
# python -m torch.distributed.run --nproc_per_node 2 train.py --device 0,1 --batch 32 --imgsz 1120 > output_2.log 2>&1 &