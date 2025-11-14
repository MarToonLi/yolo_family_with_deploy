# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Auto-batch utils
"""

from copy import deepcopy

import numpy as np
import torch

from utils.general import LOGGER, colorstr
from utils.torch_utils import profile


def check_train_batch_size(model, imgsz=640, amp=True):
    # Check YOLOv5 training batch size
    with torch.cuda.amp.autocast(amp):
        return autobatch(deepcopy(model).train(), imgsz)  # compute optimal batch size


def autobatch(model, imgsz=640, fraction=0.8, batch_size=16):
    # Automatically estimate best YOLOv5 batch size to use `fraction` of available CUDA memory
    # Usage:
    #     import torch
    #     from utils.autobatch import autobatch
    #     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False)
    #     print(autobatch(model))

    # Check device
    prefix = colorstr('AutoBatch: ')
    LOGGER.info(f'{prefix}Computing optimal batch size for --imgsz {imgsz}')
    #? 有意思的是，检测模型的设备是根据模型的第一个tensor参数的设备信息来判断的，而且device是一个对象，包含了设备的类型和索引！
    #? model.parameters() 是一个生成器，返回模型的所有参数，包括权重和偏置项！
    device = next(model.parameters()).device  # get model device
    if device.type == 'cpu':
        LOGGER.info(f'{prefix}CUDA not detected, using default CPU batch-size {batch_size}')
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f'{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}')
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30             # bytes to GiB (1024 ** 3)  #? 位运算表达式，数字1左移30位，相当于乘以2的30次方，即1024的3次方，即1GB
    d = str(device).upper()  # 'CUDA:0' 获取的是device对象的type属性值和index属性值的组合，例如'CUDA:0'
    #? 虽然device不包含显卡的信息，但是可以根据device索引调用CUDAAPI来获取显卡的信息
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb              # GiB total
    r = torch.cuda.memory_reserved(device) / gb   # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f'{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free')

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)
    except Exception as e:
        LOGGER.warning(f'{prefix}{e}')

    # Fit a solution
    y = [x[2] for x in results if x]  # memory [2]
    p = np.polyfit(batch_sizes[:len(y)], y, deg=1)  # first degree polynomial fit
    b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
    if None in results:  # some sizes failed
        i = results.index(None)  # first fail index
        if b >= batch_sizes[i]:  # y intercept above failure point
            b = batch_sizes[max(i - 1, 0)]  # select prior safe point
    if b < 1 or b > 1024:  # b outside of safe range
        b = batch_size
        LOGGER.warning(f'{prefix}WARNING ⚠️ CUDA anomaly detected, recommend restart environment and retry command.')

    fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
    LOGGER.info(f'{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅')
    return b
