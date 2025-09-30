#!/bin/bash
# Single-GPU
# python train.py --epochs 10 --data coco128.yaml --weights yolov5s.pt --cache --evolve
# python train.py  --batch-size 64 --data coco.yaml --weights yolov5s.pt --device 0

# Multi-GPU with delay
nohup python detect_train_remote.py > output_1.log 2>&1 &