# YOLOv5 🚀 by tong, 20251104
"""
Train a YOLOv5 model on a custom dataset by modifying train.py to support Ray Tune HPO.

Usage - Single-GPU training:
    $ python train_raytune_copy.py --device 0 --batch 24 --imgsz 1120 --epochs 500 --cache --cos-lr  --image-weights 
    --hyp '.../apple_3_7_hyp_evolve_20251024_1126.yaml' --data '.../apple_3_7_jpg_train.yaml' --cfg '.../yolov5s.yaml' --weights '.../yolov5s.pt'
    #! 命令行只能调整部分parse_opt()中的参数,  还有其他的一些参数需要人为手动修改:
    #! 1. 设置 opt.raytune = True；2. update default_space, max_generations, trials等参数

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

"""
 
import argparse
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()                 # FILE: "F:\Projects\yolo_family\yolov5_7.0\train.py"
ROOT = FILE.parents[0]                          # YOLOv5 root directory (e.g. F:\Projects\yolo_family\yolov5_7.0)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))                  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative (e.g. ".")
# print("ROOT: {}".format(ROOT))

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve, plot_lr_scheduler
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)


# 导包
import ray
from ray import tune
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch


# https://pytorch.org/docs/stable/elastic/run.html
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))     # LOCAL_RANK 通常用于指示当前进程在本地机器上的排名，特别是在多GPU训练时用于标识每个 GPU。
RANK       = int(os.getenv('RANK', -1))           # RANK 表示当前进程在整个分布式训练中的全球排名，通常用于跨多个机器或节点的分布式训练。
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))      # WORLD_SIZE 代表分布式训练中总的进程数量或节点数量，即总的工作量大小。
GIT_INFO = check_git_info()


def ray_train(config, default_config, opt, device, callbacks):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze                 # ! 类似地，参数在函数间传递需要使用结构体，而在函数内需要首先unpack
    callbacks.run('on_pretrain_routine_start')                                     # todo: callbacks回调函数时如何使用的

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'


    default_config.update(config)
    hyp = default_config

    #! ray_train 替换 train: 超参数内容提前背读取
    # Hyperparameters
    # if isinstance(hyp, str):
    #     LOGGER.info(colorstr('hyperparameter file path: ') + hyp)
    #     with open(hyp, errors='ignore') as f:
    #         hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:                                                             # todo:  RANK 如果等于0或者-1代表什么
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')                                                #! load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()                                    # ! checkpoint state_dict as FP32 (float32)
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)             # ? intersect 交叉 去除head部分的内容
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)      # create
    amp = check_amp(model)                                                          # ! check AMP 通过检查两个张量是否在某个容忍度范围内近似相等

    # Freeze
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results) （因训练结果不稳定而评论）
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)                # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # ! verify imgsz is gs-multiple, 还没有进行resize

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64                                              # nominal batch size 名义上的批量
    accumulate = max(round(nbs / batch_size), 1)          # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # ! scale weight_decay 这么神奇的trick操作吗？
    LOGGER.info(f"optimizer: lr0 = {hyp['lr0']:.6f}")
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)                              # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)             
    plot_lr_scheduler(optimizer, scheduler, epochs)                        # todo: 通常用于train.py中学习率设置后可视化一下
    
    
    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None                     # ! 指数移动平均；一种给与近期数据更高权重的平均方法 相当于做了一个learning rate decay
                                                                           # ! 不能在多GPU训练时使用

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')  # todo: 啥？搞了半天，yolov5不支持多GPU？
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:                                           # ! batch_size较小时，需要它来使得统计量更加准确
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,           # todo: opt中的cache似乎只可能是ram或者disk？
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'  #! 很有意思的一步
    #! 输出dataset.albumentations.transform
    if hasattr(dataset.albumentations, 'transform') and dataset.albumentations.transform is not None:
        LOGGER.info(f"{colorstr('train: ')}albumentations.transform: {dataset.albumentations.transform}")

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,                                   # ! 它与train的配置不一样
                                       rank=-1,                                     # !
                                       workers=workers * 2,                         # !
                                       pad=0.5,                                     # !
                                       prefix=colorstr('val: '))[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    nl = de_parallel(model).model[-1].nl       # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl                       # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl             # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc     # attach number of classes to model
    model.hyp = hyp   # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights  # ! 类别均衡
    model.names = names

    # Start training
    t0 = time.time()
    nb = len(train_loader)                           # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations), 有讲究，本质看迭代数
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)                                             # todo: mAP per class 记录当前模型对每种类别的检测效果mAP值，用于image-eights
    results = (0, 0, 0, 0, 0, 0, 0)                                 # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1                          # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)                               # init loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)
        if opt.image_weights:
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc                # class weights
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)        # image weights
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255                # ! [输入处理] uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)    # ! 通过差值来resize

            # Forward
            with torch.cuda.amp.autocast(amp):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            if not noval or final_epoch:                           # ! Calculate mAP ==> noval: 默认false，即一直val
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)
                


            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)         # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)



            #! Ray.tune: 避免非ray.tune时报错
            if opt.raytune:
                # Report metrics
                metrics = {
                    "precision": results[0],
                    "recall": results[1],
                    "mAP@.5": results[2],
                    "mAP@.5-.95": results[3],
                    "fitness": fi[0]
                }
                tune.report(metrics)


            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(                                                # todo: 它与上面的validate.run有啥不同
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),          # !
                        iou_thres=0.65 if is_coco else 0.60,           # best pycocotools at iou 0.65
                        single_cls=single_cls,                         # 
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,                             # !
                        verbose=True,                                  # !
                        plots=plots,                                   # !
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run('on_train_end', last, best, epoch, results)

    torch.cuda.empty_cache()
    return results



def check_os():
    import platform
    
    system = platform.system()
    if system == "Windows":
        return "Windows"
    # elif system == "Linux":
    #     # 进一步检查是否是 Ubuntu
    #     distro = platform.linux_distribution()
    #     if "Ubuntu" in distro:
    #         return "Ubuntu"
    #     else:
    #         return "Linux (非 Ubuntu)"
    else:
        return "其他操作系统"
    
def check_lanyun_env():
    # 检查 /root/lanyun-tmp/ 路径是否存在，以判断是否在蓝云环境中运行
    lanyun_path = "/root/lanyun-tmp/"
    return os.path.exists(lanyun_path)



def parse_opt(known=False):

    #! 设置yolo.train 参数 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if check_os() == "Windows":  # 本机调试环境
        root = r"D:\ProjectsRelated\CoreProjects\yolo_family_with_deploy"
        weights = os.path.join(root, "resources\models\yolov5\yolov5s.pt")
        cfg     = os.path.join(root, "yolov5_7.0\models/apple_3_7/yolov5s.yaml")
        data    = os.path.join(root, "yolov5_7.0\data/cable/apple_3_7_jpg_train.yaml")
        hyp     = os.path.join(root, "yolov5_7.0\data/hyps/apple_3_7_hyp_evolve.yaml")
        project = os.path.join(root, "yolov5_7.0/runs/train")
    elif check_lanyun_env():      # 蓝耘环境
        root = r"/root/lanyun-tmp/projects/yolo_family_with_deploy"
        # weights = os.path.join(root, r"yolov5_7.0/runs/train/exp14/weights/best.pt")
        # weights = os.path.join(root, r"yolov5_7.0/runs/train/exp42/weights/best.pt")
        weights = os.path.join(root, r"resources/models/yolov5/yolov5s.pt")
        cfg     = os.path.join(root, "yolov5_7.0/models/apple_3_7/yolov5s_P2.yaml")
        data    = os.path.join(root, "yolov5_7.0/data/cable/apple_3_7_jpg_train_remote.yaml")
        hyp     = os.path.join(root, "yolov5_7.0/data/hyps/apple_3_7_hyp_evolve_20251108_4090.yaml")
        project = os.path.join(root, "yolov5_7.0/runs/train")
    else:                      # 其他Linux环境，直接退出
        sys.exit("当前环境非Windows且非蓝云环境，程序退出！请根据实际情况修改train_raytune.py中的默认路径参数。")
        sys.exit(0)
    

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()
    #? 最为常用的参数: 默认参数，默认全部为本地运行服务(很重要的原则)
    # ROOT / 'runs/train/exp8/weights/best.pt'
    parser.add_argument('--weights',         type=str, default=weights,          help='initial weights path')  
    parser.add_argument('--cfg',             type=str, default=cfg ,             help='模型配置文件路径')
    parser.add_argument('--data',            type=str, default=data,             help='dataset.yaml path')
    parser.add_argument('--hyp',             type=str, default=hyp ,             help='训练超参数配置文件路径')
    parser.add_argument('--epochs',          type=int, default=5,                                     help='total training epochs')  
    parser.add_argument('--batch-size',      type=int, default=12,                                       help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640,                       help='train, val image size (pixels)')
    parser.add_argument('--optimizer',       type=str, choices=['SGD', 'Adam', 'AdamW'], default='Adam', help='optimizer')
    parser.add_argument('--cos-lr',          action='store_true',                                       help='cosine LR scheduler')
    parser.add_argument('--patience',        type=int, default=50,                                     help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze',          nargs='+', type=int, default=[0],                          help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period',     type=int, default=-1,                                      help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed',            type=int, default=0,                                       help='Global training seed')
    
    
    # 训练结果名称控制    
    parser.add_argument('--project',         default=project,                          help='save to project/name')  # 设置每次训练结果保存的主路径名称
    parser.add_argument('--name',            default='exp',                            help='save to project/name')  # 子路径名称
    parser.add_argument('--exist-ok',        action='store_true',                      help='existing project/name ok, do not increment') # 是否覆盖同名的训练结果保存路径，默认关闭，表示不覆盖

    
    # trick参数
    parser.add_argument('--rect',            action='store_true',                      help='[trick] rectangular training')   # 矩形训练，默认关闭 
    parser.add_argument('--noautoanchor',    action='store_true',                      help='[trick] disable AutoAnchor')    #? 常用: 关闭自动计算锚框功能，默认关闭，即会自动计算
    parser.add_argument('--evolve',          type=int, nargs='?', const=300,           help='[trick] (depreciated) evolve hyperparameters for x generations')  # ? 使用超参数优化算法进行自动调参（在当前文件中不起作用）
    parser.add_argument('--raytune',         action='store_true',                      help='[trick] evolve hyperparameters for x generations by raytune')  # ? 常用: 使用超参数优化算法进行自动调参
    parser.add_argument('--cache',           type=str, nargs='?', const='ram',         help='[trick] image --cache ram/disk')        #? 常用: 缓存数据集，默认关闭
    parser.add_argument('--image-weights',   action='store_true',                      help='[trick] use weighted image selection for training')  #? 常用: 对数据集图片进行加权训练
    parser.add_argument('--multi-scale',     action='store_true',                      help='[trick] vary img-size +/- 50%%')    # 多尺度训练，训练过程中每次输入图片会放大或缩小50%。
    parser.add_argument('--label-smoothing', type=float, default=0.1,                  help='[trick] Label smoothing epsilon')   # 表示在每个标签的真实概率上添加一个 epsilon=0.1的噪声，从而使模型对标签的波动更加鲁棒；
    
    
    # DDP和多GPU等相关
    parser.add_argument('--device',          default="0",                              help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  #? 常用: 单GPU训练时指定GPU编号，多GPU训练时指定多个GPU编号，CPU训练时指定cpu
    parser.add_argument('--resume',          nargs='?', const=True, default=False,     help='resume most recent training')    # 断点续训
    parser.add_argument('--nosave',          action='store_true',                      help='only save final checkpoint')
    parser.add_argument('--noval',           action='store_true',                      help='only validate final epoch')
    parser.add_argument('--noplots',         action='store_true',                      help='save no plot files')
    parser.add_argument('--bucket',          type=str, default='',                     help='gsutil bucket')
    parser.add_argument('--single-cls',      action='store_true',                      help='train multi-class data as single-class')
    parser.add_argument('--sync-bn',         action='store_true',                      help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers',         type=int, default=8,                      help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--local_rank',      type=int, default=-1,                     help='Automatic DDP Multi-GPU argument, do not modify')
    parser.add_argument('--quad',            action='store_true',                      help='quad dataloader')


    # Logger arguments
    parser.add_argument('--entity',         default=None,                         help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval',  type=int, default=-1,                 help='Set bounding-box image logging interval') # 每隔多少个epoch记录一次带有边界框的图片
    parser.add_argument('--artifact_alias', type=str, default='latest',           help='Version of dataset artifact to use')
    #! 设置yolo.train 参数 ----------------------------------------------------------------------------------------------------------

    # 解析命令行传入的参数：parser.parse_args()
    return parser.parse_known_args()[0] if known else parser.parse_args()


def ray_main(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements()   # 检查库是否存在，否则会自动安装

    # 完善opt参数
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'

    # ray: 更换save_dir路径，raytune数据和最终训练的结果都放在这里
    if opt.raytune:
        if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
            opt.project = str(ROOT / 'runs/evolve_ray')
        opt.project = opt.project.replace("train", "evolve_ray")
        opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
    # if opt.name == 'cfg':
        # opt.name = Path(opt.cfg).stem  # use model.yaml as name   # stem返回路径的文件名部分，不包括扩展名； 默认地opt中为exp
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # 初始化save_dir到opt

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)


    with open(opt.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    default_config = hyp.copy()
    config = {}


    # ===============================================================================================
    # Train
    if not opt.raytune:
        ray_train(config, default_config, opt, device, callbacks)

    # 使用 Ray Tune进行超参数优化
    else:
        
        #! Ray Tune 相关参数 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # 定义超参数搜索空间
        #! 建议 1) 部分参数使用tune.choice离散化取值范围和tune.uniform连续取值范围相结合的方式进行定义；
        default_space = {
            # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
            "lr0": tune.uniform(1e-4, 1e-2),
            "lrf": tune.uniform(0.01, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
            # "weight_decay": tune.uniform(0.0, 0.001),  # optimizer weight decay
            # "warmup_epochs": tune.uniform(0.0, 5.0),  # warmup epochs (fractions ok)
            # "warmup_momentum": tune.uniform(0.0, 0.95),  # warmup initial momentum
            # "box": tune.uniform(0.02, 0.2),  # box loss gain
            # "cls": tune.uniform(0.2, 4.0),  # cls loss gain (scale with pixels)
            # "hsv_h": tune.uniform(0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            # "hsv_s": tune.uniform(0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            # "hsv_v": tune.uniform(0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": tune.randint(2, 10),  # image rotation (+/- deg)
            "translate": tune.choice([round(x * 0.1, 1) for x in range(0, 11)]),  # image translation (+/- fraction)
            "scale": tune.choice([round(x * 0.1, 1) for x in range(0, 11)]),  # image scale (+/- gain)
            # "shear": tune.uniform(0.0, 10.0),  # image shear (+/- deg)
            # "perspective": tune.uniform(0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": tune.choice([round(x * 0.1, 1) for x in range(0, 11)]),  # image flip up-down (probability)
            "fliplr": tune.choice([round(x * 0.1, 1) for x in range(0, 11)]),  # image flip left-right (probability)
            # "bgr": tune.uniform(0.0, 1.0),  # image channel BGR (probability)
            "mosaic": tune.choice([round(x * 0.1, 1) for x in range(0, 11)]),  # image mosaic (probability)
            # "mixup": tune.uniform(0.0, 1.0),  # image mixup (probability)
            # "cutmix": tune.uniform(0.0, 1.0),  # image cutmix (probability)
            # "copy_paste": tune.uniform(0.0, 1.0),  # segment copy-paste (probability)
        }  

        # 定义ray.tune的控制参数
        #! 目前仅支持单GPU的超参数优化
        if not torch.cuda.is_available():
            import sys
            print("No GPU found, exiting...")
            sys.exit(1)

        if check_os() == "Windows":
            max_generations = 1
            gpus_per_trial = 1  #! 仅考虑单GPU的情况
            cpus_per_trial = 8
            opt.workers = cpus_per_trial
            opt.device = "0"
            device = torch.device('cuda:0')
            trials = 1
            grace_period = 1
        else:
            max_generations = 20
            gpus_per_trial = 1  #! 仅考虑单GPU的情况
            cpus_per_trial = 8
            opt.workers = cpus_per_trial
            opt.device = "0"
            device = torch.device('cuda:0')
            trials = 30
            grace_period = 2

        #! Ray Tune 相关参数 ----------------------------------------------------------------------------------------------------------


        # 检查路径是否存在，不存在则创建
        save_dir = Path(opt.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        tune_default_config = deepcopy(default_config)
        tune_opt = deepcopy(opt)

        # 定义搜索算法和调度器
        algo = OptunaSearch()   # todo: 还有其他搜索算法可供选择

        # 创建Tuner并运行超参数优化
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(ray_train, default_config=tune_default_config, opt=tune_opt, device=device, callbacks=callbacks),  # 传递额外参数到train函数中
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}  # 每个试验分配的资源
            ),
            tune_config=tune.TuneConfig( 
                metric="fitness",
                mode="max",
                search_alg=algo,
                scheduler=ASHAScheduler(
                    # metric="recall",  #? 与train函数中tune.report的metric名称对应；与tuneconfig中的metric只能存在一个
                    # mode="max",
                    max_t=max_generations,
                    grace_period=grace_period,
                    time_attr="training_iteration",
                    reduction_factor=2),
                num_samples=trials,
                trial_name_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
                trial_dirname_creator=lambda trial: f"{trial.trainable_name}_{trial.trial_id}",
            ),
            param_space=default_space,
            run_config=tune.RunConfig(
                name=Path(opt.save_dir).name,
                storage_path=Path(opt.save_dir).parent,
                stop={"training_iteration": max_generations},
                verbose=2,
                log_to_file=True,
            ),
        )

        LOGGER.info(f"opt.project: {opt.project}")
        LOGGER.info(f"opt.save_dir: {opt.save_dir}")

        # 运行超参数优化
        LOGGER.info("Starting Ray Tune hyperparameter optimization...")
        results = tuner.fit()

        # 输出最佳结果
        best_config = results.get_best_result(metric="fitness",mode="max").config
        test_default_config = deepcopy(default_config)
        test_opt = deepcopy(opt)

        try:
            result_df = results.get_dataframe(filter_metric="fitness", filter_mode="max")
            LOGGER.info(f"Best config is: {best_config}")

            # 将config持久化到yaml文件中
            best_config_yaml_path = Path(opt.save_dir) / 'best_hyp.yaml'
            with open(best_config_yaml_path, 'w') as f:
                yaml.dump(best_config, f)
                LOGGER.info(f"Best config yaml saved to: {best_config_yaml_path}")
            # 将 result_df 保存到 CSV 文件中
            result_csv_path = Path(opt.save_dir) / 'ray_tune_results.csv'
            result_df.to_csv(result_csv_path, index=False)
            LOGGER.info(f"All trial results saved to: {result_csv_path}")
            ray.shutdown()   #


            # 进行最终训练
            test_opt.save_dir = str(Path(opt.save_dir) / "final_train")

        except Exception as e:
            LOGGER.error(f"Error processing Ray Tune results: {e}")


        total_mem = torch.cuda.get_device_properties(0).total_memory        # 总显存（字节）
        allocated_mem = torch.cuda.memory_allocated()                       # 已分配显存（字节）
        used_ratio = allocated_mem / total_mem                              # 已用比例

        print(f"[DEBUG] 显存使用: {allocated_mem / 1024**2:.2f} MB / {total_mem / 1024**2:.2f} MB "
            f"(占比: {used_ratio * 100:.1f}%)")
        
        if used_ratio > 0.5:
            torch.cuda.empty_cache()
            import time
            time.sleep(60)
            torch.cuda.empty_cache()
            ray.shutdown()   # 防止仍然占用较大的显存，影响后续训练！ #todo: 待验证
        else:
            ray.shutdown()   # 防止仍然占用较大的显存，影响后续训练！ #todo: 待验证



        LOGGER.info("Starting final training with best hyperparameters...")
        ray_train(best_config, test_default_config,  test_opt, device, callbacks)



if __name__ == "__main__":
    opt = parse_opt()  


    #! usage: 如果使用 ray.tune 进行超参数优化
    #! 1. 设置 opt.raytune = True；2. update parse_opt() 中的默认参数; 3. update default_space, max_generations, trials等参数
    opt.raytune = True
    LOGGER.info(f"opt.hyp: {opt.hyp}" )

    if opt.raytune:
        opt.nosave = True
        ray_main(opt)
    else:
        ray_main(opt)

    print()
