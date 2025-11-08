"""
https://docs.ultralytics.com/zh/modes/train/

train 的参数：

#!? 最常用的参数
- pretrained：是否使用预训练权重或指定权重路径
- model：模型文件路径（.pt 或 .yaml），定义结构或初始化权重
- data：数据集配置文件路径（如 coco8.yaml），包含类别、路径等信息
- hyp: #! yolov5中hyp参数的超参数被直接呈现出来
- epochs：训练轮数，控制训练持续时间
- batch：具有三种模式：设置为整数（例如， batch=16），自动模式，GPU 内存利用率为 60%（batch=-1），或具有指定利用率分数的自动模式（batch=0.70）。
- imgsz：输入图像尺寸
- optimizer：str	'auto'	训练优化器的选择。选项包括 SGD, Adam, AdamW, NAdam, RAdam, RMSProp 等等，或者 auto 用于基于模型配置自动选择。影响收敛速度和稳定性。
- cos_lr：使用余弦学习率调度器，在 epochs 上按照余弦曲线调整学习率
- patience：早停策略等待轮数
- freeze：冻结模型的前 N 层或按索引指定的层，从而减少可训练参数的数量。适用于微调或迁移学习。
- save_period：权重保存间隔（单位：epoch）
- seed：随机种子
fraction	float	1.0	指定用于训练的数据集比例。允许在完整数据集的子集上进行训练，这在实验或资源有限时非常有用。
nbs	int	64	用于损失归一化的标称批量大小。

- project/name/exist_ok：输出目录及相关设置
- time	float	None	最长训练时间（以小时为单位）。如果设置此参数，它将覆盖 epochs 参数，允许训练在指定时长后自动停止。适用于时间受限的训练场景。


- rect：启用最小填充策略——批量中的图像被最小程度地填充以达到一个共同的大小，最长边等于 imgsz。可以提高效率和速度，但可能会影响模型精度。
- noautoanchor：#! no
- evolve：#! no
- cache：数据集缓存方式（True/ram/disk/False）
- image-weights： #! no
- multi_scale：是否使用多尺度训练  #?
- label-smoothing
classes	list[int]	None	指定要训练的类 ID 列表。可用于在训练期间过滤掉并仅关注某些类。
close_mosaic	int	10	在最后 N 个 epochs 中禁用 mosaic 数据增强，以在完成前稳定训练。设置为 0 可禁用此功能。


- device：int 或 str 或 list	None	指定用于训练的计算设备：
单个 GPU（device=0），多个 GPU（device=[0,1]），CPU（device=cpu），适用于 Apple 芯片的 MPS（device=mps），
或自动选择最空闲的 GPU（device=-1）或多个空闲 GPU （device=[-1,-1])
- resume：从上次保存的检查点恢复训练。自动加载模型权重、优化器状态和 epoch 计数，无缝继续训练。
- save：启用保存训练检查点和最终模型权重。可用于恢复训练或模型部署。
save_period	int	-1	保存模型检查点的频率，以 epoch 为单位指定。值为 -1 时禁用此功能。适用于在长时间训练期间保存临时模型。
- val：训练时是否进行验证
- plots：是否生成训练/验证指标图
- bucket： #! no
- single_cls：在多类别数据集中，将所有类别视为单个类别进行训练。适用于二元分类任务或侧重于对象是否存在而非分类时。
- sync-bn：#! no
- workers：用于数据加载的工作线程数（每个 RANK ，如果是多 GPU 训练）。影响数据预处理和输入模型的速度，在多 GPU 设置中尤其有用。
- local-rank: #! no
- quad:       #! no
deterministic	bool	    True	强制使用确定性算法，确保可重复性，但由于限制了非确定性算法，可能会影响性能和速度。
amp	            bool	    True	启用自动混合精度（AMP）训练，减少内存使用，并可能在对准确性影响最小的情况下加快训练速度。
profile	        bool	    False	在训练期间启用 ONNX 和 TensorRT 速度的分析，有助于优化模型部署。
compile	        bool 或 str	False	启用 PyTorch 2.x torch.compile 使用以下方式进行图形编译 backend='inductor'。接受 True → "default", False → 禁用，或字符串模式，例如 "default", "reduce-overhead", "max-autotune-no-cudagraphs"。如果不支持，则会发出警告并回退到 Eager 模式。
关于批量大小设置的说明

- entity:          #! no
- upload_dataset:  #! no
- bbox_interval:   #! no
- artifact_alias:  #! no



# 超参数
lr0	            float	0.01	初始学习率（即 SGD=1E-2, Adam=1E-3)。调整此值对于优化过程至关重要，它会影响模型权重更新的速度。
lrf	            float	0.01	最终学习率作为初始速率的一部分 = (lr0 * lrf），与调度器结合使用以随时间调整学习率。
momentum	    float	0.937	SGD 的动量因子或 Adam 优化器的 beta1，影响当前更新中过去梯度的合并。
weight_decay	float	0.0005	L2 正则化项，惩罚大权重以防止过拟合。
warmup_epochs	float	3.0	    学习率预热的 epochs 数，将学习率从低值逐渐增加到初始学习率，以在早期稳定训练。
warmup_momentum	float	0.8	    预热阶段的初始动量，在预热期间逐渐调整到设定的动量。
warmup_bias_lr	float	0.1	    预热阶段偏差参数的学习率，有助于稳定初始 epochs 中的模型训练。
box	            float	7.5	    损失函数中框损失分量的权重，影响对准确预测边界框坐标的重视程度。
cls	            float	0.5	    分类损失在总损失函数中的权重，影响正确类别预测相对于其他成分的重要性。
dfl	            float	1.5	    分布焦点损失的权重，在某些 YOLO 版本中用于细粒度分类。

# pose || mask || classify
pose	      float	12.0	在为姿势估计训练的模型中，姿势损失的权重会影响对准确预测姿势关键点的强调。  #? add
kobj	      float	2.0	    姿势估计模型中关键点对象性损失的权重，用于平衡检测置信度和姿势准确性。  #? add
overlap_mask  bool	True	确定是否应将对象掩码合并为单个掩码以进行训练，还是为每个对象保持分离。如果发生重叠，则在合并期间，较小的掩码会覆盖在较大的掩码之上。  #? add
mask_ratio	  int	4	    分割掩码的下采样率，影响训练期间使用的掩码分辨率。  #? add
dropout	      float	0.0	    分类任务中用于正则化的 Dropout 率，通过在训练期间随机省略单元来防止过拟合。  #? add


数据增强相关参数（常用于提升模型泛化能力）
hsv_h	float	0.015	detect, segment, pose, obb, classify	0.0 - 1.0	通过色轮的一小部分调整图像的色调，从而引入颜色变化。帮助模型在不同的光照条件下进行泛化。
hsv_s	float	0.7	    detect, segment, pose, obb, classify	0.0 - 1.0	通过一小部分改变图像的饱和度，从而影响颜色的强度。可用于模拟不同的环境条件。
hsv_v	float	0.4	    detect, segment, pose, obb, classify	0.0 - 1.0	通过一小部分修改图像的明度（亮度），帮助模型在各种光照条件下表现良好。
degrees	float	0.0	    detect, segment, pose, obb	            0.0 - 180	在指定的角度范围内随机旋转图像，提高模型识别各种方向物体的能力。
translate	float	0.1	detect, segment, pose, obb	            0.0 - 1.0	通过图像尺寸的一小部分在水平和垂直方向上平移图像，帮助学习检测部分可见的物体。
scale	float	0.5	    detect, segment, pose, obb, classify	>=0.0	    通过增益因子缩放图像，模拟物体与相机的不同距离。
shear	float	0.0	    detect, segment, pose, obb	            -180 - +180	按指定的角度错切图像，模仿从不同角度观察物体的效果。
perspective	float	0.0	detect, segment, pose, obb	            0.0 - 0.001	对图像应用随机透视变换，增强模型理解 3D 空间中物体的能力。
flipud	float	0.0	    detect, segment, pose, obb, classify	0.0 - 1.0	以指定的概率将图像上下翻转，增加数据变化，而不影响物体的特征。
fliplr	float	0.5	    detect, segment, pose, obb, classify	0.0 - 1.0	以指定的概率将图像左右翻转，有助于学习对称物体并增加数据集的多样性。
mosaic	float	1.0	    detect, segment, pose, obb	            0.0 - 1.0	将四个训练图像组合成一个，模拟不同的场景组成和物体交互。对于复杂的场景理解非常有效。
mixup	float	0.0	    detect, segment, pose, obb	            0.0 - 1.0	混合两个图像及其标签，创建一个合成图像。通过引入标签噪声和视觉变化，增强模型的泛化能力。
cutmix	float	0.0	    detect, segment, pose, obb	            0.0 - 1.0	组合两张图像的部分区域，创建局部混合，同时保持清晰的区域。通过创建遮挡场景来增强模型的鲁棒性。
copy_paste	float	0.0	        segment	                        0.0 - 1.0	跨图像复制和粘贴对象以增加对象实例。
copy_paste_mode	str	flip	    segment	                            -	    指定 copy-paste 要使用的策略。选项包括 'flip' 和 'mixup'.    #? add
auto_augment	str	randaugment	                    classify	    -	    应用预定义的增强策略（'randaugment', 'autoaugment'或 'augmix'）通过视觉多样性来增强模型性能。  #? add
erasing	float	0.4	                                classify	0.0 - 0.9	在训练期间随机擦除图像区域，以鼓励模型关注不太明显的特征。  #? add
bgr	float	0.0	        detect, segment, pose, obb	            0.0 - 1.0	以指定的概率将图像通道从 RGB 翻转到 BGR，有助于提高对不正确通道排序的鲁棒性。  #? add



"""



import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import settings
import os


settings["tensorboard"]=True


if __name__ == '__main__':

    root_path = r'D:\ProjectsRelated\CoreProjects\yolo_family_with_deploy'
    model_yaml_path = os.path.join(root_path, r'yolov11/ultralyticsx/cfg/models/11/yolo11s.yaml')
    pretrained_weights_path = os.path.join(root_path, r'resources/models/yolov11/yolo11s.pt')
    data_yaml_path = os.path.join(root_path, r'yolov5_7.0/data/cable/apple_3_7_jpg_train.yaml')

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
                # cache="ram",
                imgsz=1120,
                epochs=500,
                #single_cls=False,  # 是否是单类别检测
                batch=8,
                #close_mosaic=0,
                device='0',
                #optimizer='SGD', # using SGD 优化器 默认为auto建议大家使用固定的.
                # resume=, # 续训的话这里填写True, yaml文件的地方改为lats.pt的地址,需要注意的是如果你设置训练200轮次模型训练了200轮次是没有办法进行续训的.
                #amp=True,  # 如果出现训练损失为Nan可以关闭amp
                #project='runs/train',
                #name='exp',
                )
 
 

