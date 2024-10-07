import torch
import torch.nn as nn
import numpy as np

from utils.misc import multiclass_nms


from yolov1.yolov1_backbone import build_backbone
from yolov1.yolov1_neck import build_neck
from yolov1.yolov1_head import build_head

# YOLOV1
class YOLOV1(nn.Module):
    def __init__(self,
                 cfg, 
                 device,
                 img_size=None,
                 num_classes=20,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 trainable=False,
                 deploy=False):
        super(YOLOV1, self).__init()
        
        # ---------------------- 基础参数 --------------------
        # 非模型可调参数
        self.cfg = cfg  # 模型配置文件
        self.device = device  # cuda或者是cpu
        self.num_classes = num_classes  # 类别的数量
        self.trainalble = trainable  # 训练的标记
        self.stride = 32 #? 模型的最大步长！
        self.deploy = deploy  #? 

        # 模型可调超级参数
        self.img_size = img_size  # 输入图像大小
        self.conf_thresh = conf_thresh  # 得分阈值
        self.nms_thresh = nms_thresh  # NMS阈值
        
        
        # --------------------- 网络结构 ---------------------
        ## 主干网络
        self.backbone, feat_dim = build_backbone(
            cfg["backbone"], trainable & cfg["pretrained"]
        )  # 不可训练意味着使用已有的权重进行模型参数初始化
        
        ## 颈部网络
        self.neck = build_neck(cfg, feat_dim, out_dim=512)
        head_dim = self.neck.out_dim
        
        ## 检测头
        self.head = build_head(cfg, head_dim, head_dim, num_classes)
        
        ## 预测头
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)
        
    def create_grid(self, fmp_size):
        """
            用于生成G矩阵，其中每个元素都是特征图上的像素坐标
        """
        
        # t而征途的宽和高
        ws, hs = fmp_size
        
        # 生成网络的x坐标和y坐标
        #! 由于输出张量的行数受meshgrid第一个参数长度的影响，因此第一个参数必须是高！
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        # torch.meshgrid 函数输入两个类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数。
        # 第一个输出张量填充为第一个输入张量的元素，各行元素相同；
        # 第二个输出张量填充为第二个输入张量的元素，各列元素相同；
        # https://blog.csdn.net/weixin_39504171/article/details/106356977#:~:text=torch.me
        
        
        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        
        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        
        # G矩阵本身就是 网格数量 * 2；一组表示预测框中心坐标的X坐标；一组表示Y坐标；
        return grid_xy
    
    def decode_boxes(self, pred_reg, fmp_size):
        """
            将yolo预测的(tx, ty), (tw, th) 转换为bbox的左上角坐标(x1, y1)和右小角坐标(x2, y2)。
            输入：
            pred_reg: (torch.Tensor) -> [B, H*W, 4] OR [H*W, 4], 网络预测的txtttwth
            fmp_size: (List[int, int]), 包含输出特征图的宽度和高度两个参数
            输出：
            pred_box: (torch.Tensor) -> [B, H*W, 4] or [H*W, 4], 解算出的边界框坐标。
        """
        
        # 生成网络坐标矩阵
        grid_cell = self.create_grid(fmp_size=fmp_size)
        
        # 计算预测边界框的中心点坐标和宽高
        #! 预测值是相对于特征图的尺寸；
        #! exp输入范围负无穷到正无穷；输出范围是大于0；这两点设计使得模型预测出的数据是合乎物理世界的；
        #? 需要观察之后的YOLO版本是否存在改进；
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * self.stride
        pred_wh = torch.exp(pred_reg[..., 2:]) * self.stride
        
        #  将所有bbox的中心点坐标和款到换算成x1y1x2y2形式
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
        
        #! torch.stack 会保存序列信息和张量信息，因此会增加一个新的维度；维度总数会发生变化
        #! torch.cat是将torch.stack后的张量序列沿着某个维度拼接；维度的总数不会变化；
        #! https://blog.csdn.net/qq_40507857/article/details/119854085#:~:text=torch.ca
        
        return pred_box
        
        