import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os
 
if __name__ == '__main__':

    root_path = r'/ns_data/projets/yolo_family_with_deploy'
    model_yaml_path = os.path.join(root_path, r'/yolov11/ultralyticsx/cfg/models/11/yolo11s.yaml')
    pretrained_weights_path = os.path.join(root_path, r'/resources/models/yolov11/yolo11s.pt')
    data_yaml_path = os.path.join(root_path, r'/yolov5_7.0/data/cable/apple_3_7_jpg_train_remote.yaml')

    model = YOLO(model_yaml_path)
    # /ns_data/projets/yolo_family_with_deploy/yolov11/ultralyticsx/cfg/models/11/yolo11s.yaml
    # 如何切换模型版本, 上面的ymal文件可以改为 yolov11s.yaml就是使用的v11s,
    # 类似某个改进的yaml文件名称为yolov11-XXX.yaml那么如果想使用其它版本就把上面的名称改为yolov11l-XXX.yaml即可（改的是上面YOLO中间的名字不是配置文件的）！
    model.load(pretrained_weights_path) 
    # 是否加载预训练权重,科研不建议大家加载否则很难提升精度
    model.train(data=data_yaml_path,
                # cfg = r"/ns_data/projets/yolo_family_with_deploy/yolov11/case_tasks/runs/detect/tune12/best_hyperparameters.yaml",
                # /ns_data/projets/yolo_family_with_deploy/yolov5_7.0/data/cable/apple_3_7_train_remote.yaml
                # /ns_data/projets/yolo_family_with_deploy/yolov11/case_tasks/runs/detect/tune12/best_hyperparameters.yaml
                # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                cache="disk",
                imgsz=1120,
                epochs=500,
                #single_cls=False,  # 是否是单类别检测
                batch=-1,
                #close_mosaic=0,
                device='0',
                #optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                #amp=True,  # 如果出现训练损失为Nan可以关闭amp
                #project='runs/train',
                #name='exp',
                )
 
 

