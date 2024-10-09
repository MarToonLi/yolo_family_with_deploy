import numpy as np
import torch



def multiclass_nms(scores, labels, bboxes, nms_thresh, num_class, class_agnostic=False):
    """_summary_

    Args:
        scores (torch.Tensor): [M, 1]
        labels (torch.Tensor): [M, 1]
        bboxes (torch.Tensor): [M, 4]
        nms_thresh (Int): _description_
        num_class (_type_): if clss_agnostic == False, num_class need to be -1;
        class_agnostic (bool, optional): 类别(数目)不可知. Defaults to False.
    """
    
    if class_agnostic:
        return multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh)
    else:
        return multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_class)
    

# class-agnostic NMS
def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh):
    # 类外nms
    keep = nms(bboxes, scores, nms_thresh)
    
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]
    
    return scores, labels, bboxes


#! 类内NMS和类间NMS的区别是什么？
#! 1. NMS极大值抑制：去除冗余预测框的操作；在模型预测的多个重叠预测框中选择出置信度最高的框；
#! 2. NMS极大值抑制操作步骤: 1) 根据置信度分数对所有框进行排序；2) 选择得分最高的框，并通过设定的iou阈值抑制与其重叠度过高的预测框；
#! 3. 整体NMS方式适用于: 目标类别之间不会没有明显重叠；优点，仅执行一次nms，但是回丢失某些类别的高分框!
#! 4. 类内NMS方式适用于：目标类别之间会存在明显重叠:   特点：各种类需要执行一次nms，会保留同类的高分框！
# class-aware NMS
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, num_class):
    # 类内nms
    keep = np.zeros(len(bboxes), dtype=np.int32)
    
    for i in range(num_class):
        inds = np.where(labels == i) [0]
        if len(inds) == 0: continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1
        
    keep = np.where(keep > 0)  #?! keep是一维数组，但是为什么不用加[0]?  
                               #! np.where返回了包含一个数组元素的元组。直接放到scores中进行条件筛选是允许的。
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]
    
    return scores, labels, bboxes
    
def nms(bboxes, scores: np.array, nms_thresh):
    """pure Python NMS

    Args:
        bboxes (np.array): [M, 4]
        scores (np.array): [M, 1]
        nms_thresh (Int): _description_
    """
    x1 = bboxes[:, 0]  # xmin
    y1 = bboxes[:, 1]  # ymin
    x2 = bboxes[:, 2]  # xmax
    y2 = bboxes[:, 3]  # ymax
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  #! 实现倒序: 1. numpy.argsort + [::-1]; 2. torch.argsort(decending=True);
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(1e-10, xx2 - xx1)    #! 1e-10: 1乘以e的10次方
        h = np.maximum(1e-10, yy2 - yy1)    
        
        inter = w * h
        outer = areas[i] + areas[order[1:]] - inter + 1e-14
        iou = inter / outer        
        
        #! 注意iou < nms_thresh, 这使得所有与当前主框NMS值超过阈值的次框都被抑制/过滤了；
        #! 1. np.where(condition, x, y), 满足条件(condition)，输出x(len(x)不一定等于len(iou))，不满足输出y。
        #!! 2. 只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标, 这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组
        inds = np.where(iou < nms_thresh)[0]  #! np.where返回满足条件的索引值, 需要是[0]
        order = order[inds + 1]               #! inds与真实的order会相差1，因为order是考虑了主框位置，而iou是不包含主框数据的.
                                              #! order的元素值表示对应score在scores中的索引值；inds的元素值是相对于order，而不是scores，表示哪些预测框被未被抑制！
        
    return keep



def nms_test(bboxes, scores: np.array, nms_thresh):
    """pure Python NMS
    s1: 置信度排序；
    s2: 选择当前循环最大置信度的预测框(索引)；
    s3: 对重叠度较大的预测框进行过滤，筛选出与当前主框无较大重叠的预测框；
    Args:
        bboxes (np.array): [M, 4]
        scores (np.array): [M, 1]
        nms_thresh (Int): _description_
    """
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    
    
    # 计算面积
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    # 全程维护的
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)                      # 选出当前的最大可能性的预测框
        
        w= 0; h =0
        
        inter = w * h
        outer = areas[i] + areas[order[1:]] - inter + 1e-14
        iou = inter / outer       
        
        inds = np.where(iou < nms_thresh)[0] # 抑制相对于当前最大可能性的预测框的其他次框。
        order = order[inds + 1] 

    return keep