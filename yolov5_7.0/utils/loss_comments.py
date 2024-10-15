 # ---------------------------------------------------------
    # build_targets 函数用于获得在训练时计算 loss 所需要的目标框，也即正样本。与yolov3/v4的不同，yolov5支持跨网格预测。
    # 对于任何一个 GT bbox，三个预测特征层上都可能有先验框匹配，所以该函数输出的正样本框比传入的 targets （GT框）数目多
    # 具体处理过程:
    # (1)首先通过 bbox 与当前层 anchor 做一遍过滤。对于任何一层计算当前 bbox 与当前层 anchor 的匹配程度，不采用IoU，而采用shape比例。如果anchor与bbox的宽高比差距大于4，则认为不匹配，此时忽略相应的bbox，即当做背景;
    # (2)根据留下的bbox，在上下左右四个网格四个方向扩增采样（即对 bbox 计算落在的网格所有 anchors 都计算 loss(并不是直接和 GT 框比较计算 loss) )
    # 注意此时落在网格不再是一个，而是附近的多个，这样就增加了正样本数。
    # yolov5 没有 conf 分支忽略阈值(ignore_thresh)的操作，而yoloy3/v4有。
    # --------------------------------------------------------

    def build_targets(self, p, targets):
        
        """所有GT筛选相应的anchor正样本
        这里通过
        p       : list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
        targets : targets.shape[314, 6] 
        解析 build_targets(self, p, targets):函数
        Build targets for compute_loss()
        :params p: p[i]的作用只是得到每个feature map的shape
                   预测框 由模型构建中的三个检测头Detector返回的三个yolo层的输出
                   tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                   如: list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
                   [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                   可以看出来这里的预测值p是三个yolo层每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 6] [num_target,  image_index+class+xywh] xywh为归一化后的框
        :return tcls: 表示这个target所属的class index
                tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                indices: b: 表示这个target属于的image index
                         a: 表示这个target使用的anchor index
                        gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失)  gj表示这个网格的左上角y坐标
                        gi: 表示这个网格的左上角x坐标
                anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        # na = 3 ; nt = 314
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        
        tcls, tbox, indices, anch = [], [], [], []
        # gain.shape=[7]
        gain = flow.ones(7, device=self.device)  # normalized to gridspace gain
        # ai.shape = (na,nt) 生成anchor索引
        # anchor索引，后面有用，用于表示当前bbox和当前层的哪个anchor匹配
        # 需要在3个anchor上都进行训练 所以将标签赋值na=3个 
        #  ai代表3个anchor上在所有的target对应的anchor索引 就是用来标记下当前这个target属于哪个anchor
        # [1, 3] -> [3, 1] -> [3, 314]=[na, nt]   三行  第一行63个0  第二行63个1  第三行63个2
        # ai.shape  =[3, 314]
        ai = flow.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        
        # [314, 6] [3, 314] -> [3, 314, 6] [3, 314, 1] -> [3, 314, 7]  7: [image_index+class+xywh+anchor_index]
        # 对每一个feature map: 这一步是将target复制三份 对应一个feature map的三个anchor
        # 先假设所有的target都由这层的三个anchor进行检测(复制三份)  再进行筛选  并将ai加进去标记当前是哪个anchor的target
        # targets.shape = [3, 314, 7]
        targets = flow.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices
        # 这两个变量是用来扩展正样本的 因为预测框预测到target有可能不止当前的格子预测到了
        # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
        # 设置网格中心偏移量
        g = 0.5  # bias
        # 附近的4个框
        # 以自身 + 周围左上右下4个网格 = 5个网格  用来计算offsets
        off = (
            flow.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets
        # 对每个检测层进行处理 
        # 遍历三个feature 筛选gt的anchor正样本
        for i in range(self.nl): #  self.nl: number of detection layers   Detect的个数 = 3
            # anchors: 当前feature map对应的三个anchor尺寸(相对feature map)  [3, 2]
            anchors, shape = self.anchors[i], p[i].shape

            # gain: 保存每个输出feature map的宽高 -> gain[2:6] = flow.tensor(shape)[[3, 2, 3, 2]] 
            # [1, 1, 1, 1, 1, 1, 1] -> [1, 1, 112, 112, 112,112, 1]=image_index+class+xywh+anchor_index
            gain[2:6] = flow.tensor(p[i].shape, device=self.device)[[3, 2, 3, 2]].float()  # xyxy gain
            # Match targets to anchors
            # t.shape = [3, 314, 7]  将target中的xywh的归一化尺度放缩到相对当前feature map的坐标尺度
            #    [3, 314, image_index+class+xywh+anchor_index]
            t = targets * gain  # shape(3,n,7)
            if nt: # 如果有目标就开始匹配
                # Matches
                # 所有的gt与当前层的三个anchor的宽高比(w/w  h/h)
                # r.shape = [3, 314, 2]
                r = t[..., 4:6] / anchors[:, None]  # wh ratio              
                # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
                # flow.max(r, 1. / r)=[3, 314, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
                # j.shape = [3, 314]  False: 当前anchor是当前gt的负样本  True: 当前anchor是当前gt的正样本
                j = flow.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare 
                # yolov3 v4的筛选方法: wh_iou  GT与anchor的wh_iou超过一定的阈值就是正样本
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # 根据筛选条件j, 过滤负样本, 得到所有gt的anchor正样本(batch_size张图片)
                # 知道当前gt的坐标 属于哪张图片 正样本对应的idx 也就得到了当前gt的正样本anchor
                # t: [3, 314, 7] -> [555, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                t = t[j]  # filter
                # Offsets 筛选当前格子周围格子 找到 2 个离target中心最近的两个格子  
                # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 
                # 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个
                # 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                # grid xy 取target中心的坐标xy(相对feature map左上角的坐标)
                # gxy.shape = [555, 2]
                gxy = t[:, 2:4]  # grid xy
                # inverse  得到target中心点相对于右下角的坐标  gain[[2, 3]]为当前feature map的wh
                # gxi.shape = [555, 2]
                gxi = gain[[2, 3]] - gxy  # inverse
                # 筛选中心坐标距离当前grid_cell的左、上方偏移小于g=0.5 
                # 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [555] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [555] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                # 筛选中心坐标距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [555] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [555] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                # j.shape=[5, 555]
                j = flow.stack((flow.ones_like(j), j, k, l, m))
                # 得到筛选后所有格子的正样本 格子数<=3*555 都不在边上等号成立
                # t: [555, 7] -> 复制 5 份target[5, 555, 7]  分别对应当前格子和左上右下格子5个格子
                # 使用 j 筛选后 t 的形状: [1659, 7]  
                t = t.repeat((5, 1, 1))[j]
                # flow.zeros_like(gxy)[None]: [1, 555, 2]   off[:, None]: [5, 1, 2]  => [5, 555, 2]
                # 得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界
                # （左右上下边框）的偏移量，然后通过 j 筛选最终 offsets 的形状是 [1659, 2]
                offsets = (flow.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # bc.shape = [1659, 2]
            # gxy.shape = [1659, 2]
            # gwh.shape  = [1659, 2]
            # a.shape = [1659, 1]
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors

            # a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            # a.shape = [1659]
            # (b, c).shape = [1659, 2]
            a, (b, c) = (
                a.contiguous().long().view(-1),
                bc.contiguous().long().T,
            )  # anchors, image, class

            # gij = (gxy - offsets).long()
            # 预测真实框的网格所在的左上角坐标(有左上右下的网格)  
            # gij.shape = [1659, 2]
            gij = (gxy - offsets).contiguous().long() 
            # 这里的拆分我们可以用下面的示例代码来进行解释：
            # import oneflow as flow

            # x = flow.randn(3, 2)
            # y, z = x.T
            # print(y.shape)
            # print(z.shape)

            # => oneflow.Size([3])
            # => oneflow.Size([3])

            # 因此：
            # gi.shape = [1659]
            # gj.shape = [1659]
            gi, gj = gij.T  # grid indices

            # Append

            # indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            # gi.shape = [1659]
            # gj.shape = [1659]
            gi = gi.clamp(0, shape[3] - 1)
            gj = gj.clamp(0, shape[2] - 1)
            # b: image index  a: anchor index  gj: 网格的左上角y坐标  gi: 网格的左上角x坐标
            indices.append((b, a, gj, gi))  # image, anchor, grid
            # tbix: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(flow.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors   对应的所有anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def __call__(self, p, targets):  # predictions, targets
        """
        这里通过输入
        p       : list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
        targets : targets.shape[314, 6] 
        为例解析 __call__ 函数

        :params p:  预测框 由模型构建中的 Detect 层返回的三个yolo层的输出（注意是训练模式才返回三个yolo层的输出）
                    tensor格式 list列表 存放三个tensor 对应的是三个yolo层的输出
                    如: ([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
                    [bs, anchor_num, grid_h, grid_w, xywh+class+classes]
                    可以看出来这里的预测值 p 是三个yolo层每个 grid_cell
                    的预测值(每个 grid_cell 有三个预测值), 后面要进行正样本筛选
        :params targets: 数据增强后的真实框 [314, 6] [num_object,  batch_index+class+xywh]
        :params loss * bs: 整个batch的总损失（一个列表）  进行反向传播
        :params flow.cat((lbox, lobj, lcls, loss)).detach():
        回归损失、置信度损失、分类损失和总损失 这个参数只用来可视化参数或保存信息
        """
        # 初始化各个部分损失   始化lcls, lbox, lobj三种损失值  tensor([0.])
        # lcls.shape = [1]
        lcls = flow.zeros(1, device=self.device)  # class loss 
        # lbox.shape = [1]
        lbox = flow.zeros(1, device=self.device)  # box loss
        # lobj.shape = [1]
        lobj = flow.zeros(1, device=self.device)  # object loss
        # 获得标签分类, 边框, 索引， anchors
        # 每一个都是列表， 有 feature map 个 
        # 都是当前这个feature map中3个anchor筛选出的所有的target(3个grid_cell进行预测)
        # tcls: 表示这个target所属的class index
        # tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
        # indices: b: 表示这个target属于的image index
        #          a: 表示这个target使用的anchor index
        #          gj: 经过筛选后确定某个target在某个网格中进行预测(计算损失) 
        #          gj表示这个网格的左上角y坐标
        #          gi: 表示这个网格的左上角x坐标
        # anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  
        # 可能一个target会使用大小不同anchor进行计算
        """shape
        p       : list([16, 3, 80, 80, 85], [16, 3, 40, 40, 85],[16, 3, 20, 20, 85])
        targets : [314, 6]
        tcls    : list([1659], [1625], [921])
        tbox    : list([1659, 4], [1625, 4], [921, 4])
        indices : list( list([1659],[1659],[1659],[1659]), list([1625],[1625],[1625],[1625]) , list([921],[921],[921],[921])  )
        anchors : list([1659, 2], [1625, 2], [921, 2])
        """
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses 依次遍历三个feature map的预测输出pi
        for i, pi in enumerate(p):  # layer index, layer predictions
            # 这里通过 pi 形状为[16, 3, 80, 80, 85] 进行解析
            """shape
            b   : [1659]
            a   : [1659]
            gj  : [1659]
            gi  : [1659]
            """
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            # tobj = flow.zeros( pi.shape[:4] , dtype=pi.dtype, device=self.device)  # target obj
            # 初始化target置信度(先全是负样本 后面再筛选正样本赋值)
            # tobj.shape = [16, 3, 80, 80]
            tobj = flow.zeros((pi.shape[:4]), dtype=pi.dtype, device=self.device)  # target obj
            # n = 1659
            n = b.shape[0]  # number of targets
            if n:
                # 精确得到第 b 张图片的第 a 个 feature map 的 grid_cell(gi, gj) 对应的预测值
                # 用这个预测值与我们筛选的这个 grid_cell 的真实框进行预测(计算损失)
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)
                """shape
                pxy     : [1659, 2]
                pwh     : [1659, 2]
                _       : [1659, 1]
                pcls    : [1659, 80]
                """
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression loss  只计算所有正样本的回归损失
                # 新的公式:  pxy = [-0.5 + cx, 1.5 + cx]    pwh = [0, 4pw]   这个区域内都是正样本
                # Get more positive samples, accelerate convergence and be more stable
                # pxy.shape = [1659, 2]
                pxy = pxy.sigmoid() * 2 - 0.5
                # https://github.com/ultralytics/yolov3/issues/168
                # pwh.shape = [1659, 2]
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i] # 和论文里不同 这里是作者自己提出的公式
                # pbox.shape = [1659, 4]
                pbox = flow.cat((pxy, pwh), 1)  # predicted box
                # 这里的tbox[i]中的xy是这个target对当前grid_cell左上角的偏移量[0,1]  而pbox.T是一个归一化的值
                # 就是要用这种方式训练 传回loss 修改梯度 让pbox越来越接近tbox(偏移量)
                # iou.shape = [1659]
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                # lbox.shape = [1]
                lbox = lbox + (1.0 - iou).mean()  # iou loss

                # Objectness
                # iou.detach()  不会更新iou梯度  iou并不是反向传播的参数 所以不需要反向传播梯度信息
                # iou.shape = [1659]
                iou = iou.detach().clamp(0).type(tobj.dtype)
                # 这里对 iou 进行排序再做一个优化：当一个正样本出现多个 GT 的情况也就是同一个 grid 中有两个 gt (密集型且形状差不多物体)
                # There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
                if self.sort_obj_iou:
                    # https://github.com/ultralytics/yolov5/issues/3605
                    # There maybe several GTs match the same anchor when calculate ComputeLoss in the scene with dense targets
                    j = iou.argsort()
                    # 如果同一个 grid 出现两个 GT 那么经过排序之后每个 grid 中的 score_iou 都能保证是最大的
                    # (小的会被覆盖 因为同一个grid坐标肯定相同)那么从时间顺序的话, 最后一个总是和最大的 iou 去计算 loss
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                # 预测信息有置信度 但是真实框信息是没有置信度的 所以需要我们人为的给一个标准置信度
                # self.gr是iou ratio [0, 1]  self.gr越大置信度越接近iou  self.gr越小置信度越接近1(人为加大训练难度)
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification 只计算所有正样本的分类损失 
                # self.nc = 80
                if self.nc > 1:  # cls loss (only if multiple classes)
                    # targets 原本负样本是0  这里使用smooth label 就是cn
                    # t.shape = [1659,80]
                    t = flow.full_like(pcls, self.cn, device=self.device)  # targets

                    # t[range(n), tcls[i]] = self.cp  筛选到的正样本对应位置值是cp 
                
                    t[flow.arange(n, device=self.device), tcls[i]] = self.cp
                    # lcls.shape = [1]
                    lcls = lcls + self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in flow.cat((txy[i], twh[i]), 1)]
            #  置信度损失是用所有样本(正样本 + 负样本)一起计算损失的
            obji = self.BCEobj(pi[..., 4], tobj)
            # 每个 feature map 的置信度损失权重不同  要乘以相应的权重系数 self.balance[i]
            # 一般来说，检测小物体的难度大一点，所以会增加大特征图的损失系数，让模型更加侧重小物体的检测
            lobj = lobj + (obji * self.balance[i])  # obj loss

            if self.autobalance:
                # 自动更新各个 feature map 的置信度损失系数
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        # 根据超参中的损失权重参数 对各个损失进行平衡  防止总损失被某个损失主导
        """shape
        lbox    : [1]
        lobj    : [1]
        lcls    : [1]
        """
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        # loss = lbox + lobj + lcls  平均每张图片的总损失
        # loss * bs: 整个batch的总损失
        # .detach()  利用损失值进行反向传播
        return (lbox + lobj + lcls) * bs, flow.cat((lbox, lobj, lcls)).detach()