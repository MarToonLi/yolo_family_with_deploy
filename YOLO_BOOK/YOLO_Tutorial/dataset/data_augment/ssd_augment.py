import cv2
import numpy as np
import torch
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


#! 仅修改了image的数据类型的astype操作
class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

#! 仅涉及BGR和HSV之间的cvtColor操作
# 未使用
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels

#! ReSiaze: 将图像和标签坐标进行同步resize
class Resize(object):
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, image, boxes=None, labels=None):
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))
        # rescale bbox
        if boxes is not None:
            img_h, img_w = image.shape[:2]
            boxes[..., [0, 2]] = boxes[..., [0, 2]] / orig_w * img_w
            boxes[..., [1, 3]] = boxes[..., [1, 3]] / orig_h * img_h

        return image, boxes, labels

#! 随机饱和度调整；1. image必须是HSV类型，进队S通道进行调整; 2. 随机体现在，一张图像是否进行调整，饱和度调整的幅度随机；
# 未使用
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)   #? 如果超过255咋整? 
            #! 为什么限于G通道的饱和度: 1. 简化实现；2. 满足特定任务需求（特定任务对绿色比较敏感）

        return image, boxes, labels

#! 随机色调调整: image必须是HSV类型，进队H通道进行调整
# 未使用
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0  # 保持H通道在有效范围
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0    # 保持H通道在有效范围
        return image, boxes, labels

#! 随机亮度噪音: 竟然是从通道随机交换实现的
# 未使用
class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels

#! 随机对比度调整: 竟然是全通道像素值在0.5和1.5的比例之间调整
# 未使用
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels

#! 随机亮度调整: 竟然也是在全通道像素值上进行加运算调整；
# 未使用
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels

#! 随机裁剪: 1. 随机体现在图像宽高；同步变化的是GT框的数目和bbox坐标；2. 宽高的随机被两个条件限制: 1) 0.5w < h < 2w 2) 裁剪区域与M个GT框的IOU值中的最小值。
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        # check
        if len(boxes) == 0:
            return image, boxes, labels

        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options))  
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                #! [condition1] 0.5w < h < 2w
                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                #! [condition2] boxes(num_boxes, 4); 一般仅对min_iou有限制，对max_iou没有限制；
                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue
                
                #! 如何具体的裁剪
                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

#! 随机膨胀调整: 图像宽高按比例调整，原图像处于图像右下角。
class Expand(object):
    def __call__(self, image, boxes, labels = None):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

#! 随机水平翻转调整: 图像和标签同步修改，利用切片快速实现
class RandomHorizontalFlip(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]  #! 0(x1) 2(x2)
        return image, boxes, classes

# 未使用
class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image

#! 图像光学畸变: 一些图像增强方法的组合
# 未使用
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels


# ----------------------- Main Functions -----------------------
##! Augmentation --> tensor
## SSD-style Augmentation
class SSDAugmentation(object):
    def __init__(self, img_size=640):
        self.img_size = img_size
        self.augment = Compose([
            ConvertFromInts(),                         #! 将int类型转换为float32类型 (必须最前)
            PhotometricDistort(),                      # 图像颜色增强
            Expand(),                                  # 扩充增强
            RandomSampleCrop(),                        # 随机剪裁
            RandomHorizontalFlip(),                    # 随机水平翻转
            Resize(self.img_size)                      #! resize操作 (必须最后)
        ])

    def __call__(self, image, target, mosaic=False):
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()
        deltas = None
        
        # augment
        image, boxes, labels = self.augment(image, boxes, labels)

        # to tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        target['boxes'] = torch.from_numpy(boxes).float()
        target['labels'] = torch.from_numpy(labels).float()

        return img_tensor, target, deltas
    
    
##! resize --> tensor
## SSD-style valTransform
class SSDBaseTransform(object):                            
    def __init__(self, img_size):
        self.img_size = img_size

    def __call__(self, image, target=None, mosaic=False):  #! SSDBaseTransform的应用场景主要是, 模型的validation和inference;
        """
        image: (numpy.array) [H, W, B]
        """
        deltas = None
        
        # resize
        orig_h, orig_w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size)).astype(np.float32)       #! 从整型转换为浮点型
        
        # scale targets (跟随img_size发生变化)
        if target is not None:
            boxes = target['boxes'].copy()
            labels = target['labels'].copy()
            img_h, img_w = image.shape[:2]
            boxes[..., [0, 2]] = boxes[..., [0, 2]] / orig_w * img_w   # x, w
            boxes[..., [1, 3]] = boxes[..., [1, 3]] / orig_h * img_h   # y, h
            target['boxes'] = boxes
        
        # to tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()  # [H, W, B] --> [B, H, W]
        if target is not None:
            target['boxes'] = torch.from_numpy(boxes).float()       # 可能都是整形，所以需要从整型变成浮点型，便于模型训练和训练优化
            target['labels'] = torch.from_numpy(labels).float()
            
        return img_tensor, target, deltas


        """
        1.  object类支持更高级的特性
        2. 实现了 __call__ 的类的实例(对象)可以使用 () 运算符进行调用。
            # 实例化对象
            callable_instance = CallableClass("Alice")
            # 调用对象，就像调用函数一样
            result = callable_instance("Hello")
        3. 所有的数据增强方法都不会对超过255的值和小于0的值进行处理；
        4. HSV各通道的取值范围: 0~360; 0~1.0; 0~1.0
        
        """