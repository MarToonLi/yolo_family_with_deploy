# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):     # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  #! target-subset of predictions, corresponding to the positive samples.

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio      #! 真实框的物体存在置信度(标签)是IOU值!

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """
        Summary: 为所有的GT选择相应的anchor正样本
                筛选条件：GT和anchor的宽比和高比，大于一定阈值的就是负样本，反之为正样本; 
                #! 指标宽比和高比的计算不需要真实框的中心点数据和模型的预测框数据，只需要真实框的宽高和先验框的宽高。
                而筛选得到的tcls, tbox, indices, anch信息将传入call函数，根据indices筛选出pred中每个grid预测得到的信息，获取对应gridcell上的正样本。
        Following Comments's assumption:
            当前batch的图像中的目标实例有72个，其他配置是yolov5的基本配置

        Args:
            p (list<torch.Tensor>): p is output of model (train), list of 3 layers; this func uses p's each layer's shape (fm's w and h) to build gain;
                [22, 3, 80, 80, 10]; 
                [22, 3, 40, 40, 10]; 
                [22, 3, 20, 20, 10]; 
            targets (torch.Tensor) -> [nt, 7]: targets is normalized label (image_index, class, x, y, w, h);
                                                targets: [72, 6] -> [72, 7] -> [3, 72, 7] -> t[X1, 7] -> t[5*X1, 7] -> t[X2, 7]

        Returns: (四个列表包含nl个张量元素)
            tcls    -> list<torch.Tensor[M, 1]>[3]: 各正样本anchor对应真实框的class_index;
            tbox    -> list<torch.Tensor[M, 4]>[3]: wywh 各样本anchor对应真实框的box;
                                                    其中 1. xy是这个target对当前grid_cell左上角的偏移量; 2. xywh的单位尺度不是01而是anchor所属的金字塔level的特征图尺度
            indices -> list<torch.Tensor[M, 1]>[3]: b: 各正样本anchor对应真实框所属图像的image_index; 
                                                    a: 使用的anchor_index; 
                                                    gj: 这个网格的左上角y坐标; 
                                                    gi: 表示这个网格的左上角x坐标;
            anch    -> list<torch.Tensor[M, 2]>[3]: 各正样本anchor尺寸(原始尺寸除以对应特征图层的步长);
        """
        # Build targets for compute_loss(), input targets(image_index, class, x, y, w, h)
        na, nt = self.na, targets.shape[0]          # number of anchors, number of targets [X, 6] (e.g. na = 3, nt = 72)
        tcls, tbox, indices, anch = [], [], [], []  
        gain = torch.ones(7, device=self.device)    #! 负责将归一化的XYXY值的尺度从01尺度转换到feature map的宽高尺度；
        # ai is matrix: [[0,0,...,0], [1,1,...,1], [2,2,...,2]], ai.shape = (na, nt) # ai means anchor_index, 元素取值范围是(0,1,2);
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)   #! [3, 72]      
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)            # append anchor indices            #! [3, 72, 7]    #! repeat: tensor在各个维度的重复次数

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets  #TODO (超级参数) 这里能够控制考虑周围区域多大的范围


        for i in range(self.nl):  # number of layers
            anchors, shape = self.anchors[i], p[i].shape    # self.anchors [3, 3, 2]       #! self.anchors: 是model.yaml中尺寸宽高除以特征图的尺度(8/16/32), 该操作在DetectionModel中
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]   # xyxy gain                    #! torch.tensor(shape)得到torch类型的列表；2,3索引指的是HW； 
                                                            # e.g. if shape == [22, 3, 80, 80, 10]; gain = [1, 1, 80, 80, 80, 80, 1];  
            

            # Match targets to anchors          #!!! targets
            t = targets * gain  # shape(3,n,7)  #! 将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度  #!!! label.txt文件中回归标签都是归一化值；
            if nt:
                #! S: 过滤掉的是每一特征层下，相对于真实框过长或者过高的anchor；
                # 这一步骤是不局限于真实框所在网格位置的！
                # Matches
                r = t[..., 4:6] / anchors[:, None]                          # wh ratio   # [3, 72, 2] / [3,1,2] --> [3, 72, 2] 
                                                                            #! 各GT真实框与各个anchor的比例值, 作为过滤条件; #! t[4:6]和anchors的值都是在对应特征图的尺度上！
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']    # compare  #! 小于阈值的被留下; anchor_t默认是4;
                                                                            # 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                                                                            # tensor.max(dim): 指定维度上张量的最大值 第一项是values，第二项是indices；[3, 72]
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]                                                    # filter    j[3, 72] --> t[150, 7];  #? 如何变得平坦的呢？—— bool索引筛选后，会实现降维！

                #! S: 从周围的十字范围内选择最可能的两个anchor网格作为正样本，因为GT中心点的附近cnchor也可能存在高质量的预测框
                # Offsets
                gxy = t[:, 2:4]           # grid xy  [150, 2]                     左上角为中心
                gxi = gain[[2, 3]] - gxy  # inverse  [2] - [150, 2] --> [150, 2]
                j, k = ((gxy % 1 < g) & (gxy > 1)).T   # [150, 2] --> [2, 150] --> [150], [150];  
                l, m = ((gxi % 1 < g) & (gxi > 1)).T   # [150, 2] --> [2, 150] --> [150], [150];
                j = torch.stack((torch.ones_like(j), j, k, l, m))   # 5 * [150] --> [5, 150];
                t = t.repeat((5, 1, 1))[j]   # t.repeat((5, 1, 1)): [150, 7] --> [5, 150, 7] --> [449, 7]  #! [j] 会引起张量降低维度
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # torch.zeros_like(gxy)[None]: [1, 150, 2]; off[:, None]: [5, 1, 2]
                # (torch.zeros_like(gxy)[None] + off[:, None]): [5, 150, 2]; --> offsets: [449, 2]: 我最多懂它能实现什么效果，但是为什么这样编写可以导致这样的结果以及变通我是不懂的；
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)             # (image, class), grid xy, grid wh, anchors  #! chunk沿某个维度均匀切分成N份；t [68, 2];
            a, (b, c) = a.long().view(-1), bc.long().T  # anchor(取值为0,1,2), image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image_id, anchor_id, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors: 
            tcls.append(c)           # class

        return tcls, tbox, indices, anch

"""
1. 所谓的正样本和负样本是针对anchor_id的；
2. 正样本和负样本所构成的集合是[3, 72, 7], 而不是[72, 7];
3. yolov5中正样本的class和box是多少就是多少，但是正样本的物体存在置信度是根据p和t在当前网格的iou值而定的；
4. 损失函数计算时，回归损失和分类损失只考虑正样本，而存在置信度的损失是考虑正负样本，其中负样本就是特征图中除了正样本所在网格以外的其他网格。
"""