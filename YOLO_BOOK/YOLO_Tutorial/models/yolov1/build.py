#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

# from .loss import build_criterion
from yolov1.loss import build_criterion
from .yolov1 import YOLOv1
from config import build_model_config, build_trans_config


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


def test_parse_args():
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='dataset/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='dataset/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/demos/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('-vt', '--vis_thresh', default=0.4, type=float,
                        help='Final confidence threshold for visualization')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show visualization')
    parser.add_argument('--gif', action='store_true', default=False, 
                        help='generate gif.')

    # model
    parser.add_argument('-m', '--model', default='yolov1', type=str,
                        help='build yolo')
    parser.add_argument('-nc', '--num_classes', default=80, type=int,
                        help='number of classes.')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--deploy", action="store_true", default=False,
                        help="deploy mode or not")
    parser.add_argument('--fuse_repconv', action='store_true', default=False,
                        help='fuse RepConv')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse Conv & BN')

    return parser.parse_args()


if __name__ == "__main__":
    
    
    args = test_parse_args()
    # 如果args.cuda为True，则使用GPU来推理，否则使用CPU来训练（可接受）
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 构建测试所用到的 Model & Transform相关的config变量
    model_cfg = build_model_config(args)
    trans_cfg = build_trans_config(model_cfg['trans_type'])

    # 构建YOLO模型
    model = build_yolov1(args, model_cfg, device, args.num_classes, False)

"""
总结：
1. 整体看，bias_value偏差值用于初始化obj置信度和分类置信度的预测头
2. 


"""