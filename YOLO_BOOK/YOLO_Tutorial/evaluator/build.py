import os

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.ourdataset_evaluator import OurDatasetEvaluator
from evaluator.apple37_evaluator import APPLE37Evaluator



def build_evluator(args, data_cfg, transform, device):
    # Basic parameters
    data_dir = os.path.join(args.root, data_cfg['data_name'])

    # Evaluator
    ## VOC Evaluator
    if args.dataset == 'voc':
        evaluator = VOCAPIEvaluator(
            data_dir=data_dir,
            device=device,
            transform=transform)
    ## COCO Evaluator
    elif args.dataset == 'coco':
        evaluator = COCOAPIEvaluator(
            data_dir=data_dir,
            device=device,
            transform=transform
            )
    ## Custom dataset Evaluator
    elif args.dataset == 'ourdataset':
        evaluator = OurDatasetEvaluator(
            data_dir=data_dir,
            device=device,
            image_set='val',
            transform=transform
        )
    elif args.dataset == 'apple37':
        evaluator = APPLE37Evaluator(
            data_dir=data_dir,
            device=device,
            image_set='apple37',
            transform=transform
        )
    return evaluator
