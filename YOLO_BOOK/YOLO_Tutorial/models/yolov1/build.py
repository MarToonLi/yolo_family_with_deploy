#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

from .loss import build_criterion
from .yolov1 import YOLOv1


# 构建 YOLOv1 网络
def build_yolov1(args, cfg, device, num_classes=80, trainable=False, deploy=False):
    print('==============================')
    print('Build {} ...'.format(args.model.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    # -------------- 构建YOLOv1 --------------
    model = YOLOv1(
        cfg = cfg,
        device = device,
        img_size = args.img_size,
        num_classes = num_classes,
        conf_thresh = args.conf_thresh,
        nms_thresh = args.nms_thresh,
        trainable = trainable,
        deploy = deploy
        )

    # -------------- 初始化YOLOv1的pred层参数 --------------
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))       #? torch.log(99)是什么意思
    
    # obj pred
    b = model.obj_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # cls pred
    b = model.cls_pred.bias.view(1, -1)
    b.data.fill_(bias_value.item())
    model.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # reg pred
    b = model.reg_pred.bias.view(-1, )
    b.data.fill_(1.0)
    model.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    w = model.reg_pred.weight
    w.data.fill_(0.)
    model.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    # -------------- 构建用于计算标签分配和计算损失的Criterion类 --------------
    criterion = None
    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion



"""
总结：
1. 整体看，bias_value偏差值用于初始化obj置信度和分类置信度的预测头
2. 


"""