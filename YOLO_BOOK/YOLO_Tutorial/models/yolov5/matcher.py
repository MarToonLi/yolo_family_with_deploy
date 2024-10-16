import numpy as np
import torch


class Yolov5Matcher(object):
    def __init__(self, num_classes, num_anchors, anchor_size, anchor_theshold):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.anchor_theshold = anchor_theshold
        # [KA, 2] : wh
        self.anchor_sizes = np.array([[anchor[0], anchor[1]]
                                      for anchor in anchor_size])
        # [KA, 4] : xywh
        self.anchor_boxes = np.array([[0., 0., anchor[0], anchor[1]]
                                      for anchor in anchor_size])     # anchor_size == 9 (default)

    def compute_iou(self, anchor_boxes, gt_box):
        """
            anchor_boxes : ndarray -> [KA, 4] (cx, cy, bw, bh).
            gt_box : ndarray -> [1, 4] (cx, cy, bw, bh).
        """
        
        # anchors: [KA, 4] (xywh->xyxy)
        anchors = np.zeros_like(anchor_boxes)
        anchors[..., :2] = anchor_boxes[..., :2] - anchor_boxes[..., 2:] * 0.5  # x1y1
        anchors[..., 2:] = anchor_boxes[..., :2] + anchor_boxes[..., 2:] * 0.5  # x2y2
        anchors_area = anchor_boxes[..., 2] * anchor_boxes[..., 3]
        
        # gt_box: [1, 4] -> [KA, 4] (xywh->xyxy)
        gt_box = np.array(gt_box).reshape(-1, 4)
        gt_box = np.repeat(gt_box, anchors.shape[0], axis=0)
        gt_box_ = np.zeros_like(gt_box)
        gt_box_[..., :2] = gt_box[..., :2] - gt_box[..., 2:] * 0.5  # x1y1
        gt_box_[..., 2:] = gt_box[..., :2] + gt_box[..., 2:] * 0.5  # x2y2
        gt_box_area = np.prod(gt_box[..., 2:] - gt_box[..., :2], axis=1)

        # intersection
        inter_w = np.minimum(anchors[:, 2], gt_box_[:, 2]) - \
                  np.maximum(anchors[:, 0], gt_box_[:, 0])
        inter_h = np.minimum(anchors[:, 3], gt_box_[:, 3]) - \
                  np.maximum(anchors[:, 1], gt_box_[:, 1])
        inter_area = inter_w * inter_h
        
        # union
        union_area = anchors_area + gt_box_area - inter_area

        # iou
        iou = inter_area / union_area
        iou = np.clip(iou, a_min=1e-10, a_max=1.0)
        
        return iou


    def iou_assignment(self, ctr_points, gt_box, fpn_strides):
        """面向单个的gt_box，ctr_points是

        Args:
            ctr_points (_type_): _description_
            gt_box (_type_): _description_
            fpn_strides (_type_): _description_

        Returns:
            _type_: _description_
        """
        # compute IoU
        iou = self.compute_iou(self.anchor_boxes, gt_box)
        iou_mask = (iou > 0.5)                                  #! 筛选: 筛选出与gt_box(单个)重叠度较高的预测框

        label_assignment_results = []
        if iou_mask.sum() == 0:
            # We assign the anchor box with highest IoU score.
            iou_ind = np.argmax(iou)                            #? 一定是9个预测框

            level = iou_ind // self.num_anchors              # pyramid level: 每个fm的anchors数目是3个，根据余数可以判断使用的是每个fmp中3个anchor中的哪一个。
            anchor_idx = iou_ind - level * self.num_anchors  # anchor index

            # get the corresponding stride
            stride = fpn_strides[level]

            # compute the grid cell
            xc, yc = ctr_points
            xc_s = xc / stride
            yc_s = yc / stride
            grid_x = int(xc_s)    # 当前预测框中心点在当前level的fm中的坐标
            grid_y = int(yc_s)

            label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])
        else:            
            for iou_ind, iou_m in enumerate(iou_mask):
                if iou_m:
                    level = iou_ind // self.num_anchors              # pyramid level
                    anchor_idx = iou_ind - level * self.num_anchors  # anchor index

                    # get the corresponding stride
                    stride = fpn_strides[level]

                    # compute the gride cell
                    xc, yc = ctr_points
                    xc_s = xc / stride
                    yc_s = yc / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])

        return label_assignment_results


    def aspect_ratio_assignment(self, ctr_points, keeps, fpn_strides):
        label_assignment_results = []
        for keep_idx, keep in enumerate(keeps):
            if keep:
                level = keep_idx // self.num_anchors              # pyramid level
                anchor_idx = keep_idx - level * self.num_anchors  # anchor index

                # get the corresponding stride
                stride = fpn_strides[level]

                # compute the gride cell
                xc, yc = ctr_points
                xc_s = xc / stride
                yc_s = yc / stride
                grid_x = int(xc_s)
                grid_y = int(yc_s)

                label_assignment_results.append([grid_x, grid_y, xc_s, yc_s, level, anchor_idx])
        
        return label_assignment_results
    

    @torch.no_grad()
    def __call__(self, fmp_sizes, fpn_strides, targets):
        """
            fmp_size: (List) [fmp_h, fmp_w]
            fpn_strides: (List) -> [8, 16, 32, ...] stride of network output.
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """
        assert len(fmp_sizes) == len(fpn_strides)
        # prepare
        bs = len(targets)
        gt_objectness = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 1]) 
            for (fmp_h, fmp_w) in fmp_sizes
            ]
        gt_classes = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, self.num_classes]) 
            for (fmp_h, fmp_w) in fmp_sizes
            ]
        gt_bboxes = [
            torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 4]) 
            for (fmp_h, fmp_w) in fmp_sizes
            ]

        for batch_index in range(bs):
            targets_per_image = targets[batch_index]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            for gt_box, gt_label in zip(tgt_box, tgt_cls):
                # get a bbox coords
                x1, y1, x2, y2 = gt_box.tolist()
                # xyxy -> cxcywh
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1
                gt_box = np.array([[0., 0., bw, bh]])

                # check target
                if bw < 1. or bh < 1.:
                    # invalid target
                    continue

                # compute aspect ratio
                ratios = gt_box[..., 2:] / self.anchor_sizes
                keeps = np.maximum(ratios, 1 / ratios).max(-1) < self.anchor_theshold

                if keeps.sum() == 0:
                    label_assignment_results = self.iou_assignment([xc, yc], gt_box, fpn_strides)
                else:
                    label_assignment_results = self.aspect_ratio_assignment([xc, yc], keeps, fpn_strides)

                # label assignment
                for result in label_assignment_results:
                    # assignment
                    grid_x, grid_y, xc_s, yc_s, level, anchor_idx = result
                    stride = fpn_strides[level]
                    fmp_h, fmp_w = fmp_sizes[level]
                    # coord on the feature
                    x1s, y1s = x1 / stride, y1 / stride
                    x2s, y2s = x2 / stride, y2 / stride
                    # offset
                    off_x = xc_s - grid_x
                    off_y = yc_s - grid_y
 
                    if off_x <= 0.5 and off_y <= 0.5:  # top left
                        grids = [(grid_x-1, grid_y), (grid_x, grid_y-1), (grid_x, grid_y)]
                    elif off_x > 0.5 and off_y <= 0.5: # top right
                        grids = [(grid_x+1, grid_y), (grid_x, grid_y-1), (grid_x, grid_y)]
                    elif off_x <= 0.5 and off_y > 0.5: # bottom left
                        grids = [(grid_x-1, grid_y), (grid_x, grid_y+1), (grid_x, grid_y)]
                    elif off_x > 0.5 and off_y > 0.5:  # bottom right
                        grids = [(grid_x+1, grid_y), (grid_x, grid_y+1), (grid_x, grid_y)]

                    for (i, j) in grids:
                        is_in_box = (j >= y1s and j < y2s) and (i >= x1s and i < x2s)
                        is_valid = (j >= 0 and j < fmp_h) and (i >= 0 and i < fmp_w)

                        if is_in_box and is_valid:
                            # obj
                            gt_objectness[level][batch_index, j, i, anchor_idx] = 1.0
                            # cls
                            cls_ont_hot = torch.zeros(self.num_classes)
                            cls_ont_hot[int(gt_label)] = 1.0
                            gt_classes[level][batch_index, j, i, anchor_idx] = cls_ont_hot
                            # box
                            gt_bboxes[level][batch_index, j, i, anchor_idx] = torch.as_tensor([x1, y1, x2, y2])

        # [B, M, C]
        gt_objectness = torch.cat([gt.view(bs, -1, 1) for gt in gt_objectness], dim=1).float()
        gt_classes = torch.cat([gt.view(bs, -1, self.num_classes) for gt in gt_classes], dim=1).float()
        gt_bboxes = torch.cat([gt.view(bs, -1, 4) for gt in gt_bboxes], dim=1).float()

        return gt_objectness, gt_classes, gt_bboxes
