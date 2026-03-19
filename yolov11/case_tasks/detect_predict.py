
from ultralytics import YOLO
import os

if __name__ == '__main__':
    root_path = r'D:\ProjectsRelated\CoreProjects\yolo_family_with_deploy'
    
    model_path = r"D:\ProjectsRelated\CoreProjects\yolo_family_with_deploy\runs\detect\train6\weights\best.pt"
    data_yaml_path = os.path.join(root_path, r'yolov5_7.0/data/trans/train_det.yaml')
    
    # Load a model
    model = YOLO(model_path, verbose=True)  # load a custom model

    # Validate the model
    metrics = model.predict(source=r"E:\Resources\datasets\JY\jy_steelvol_v1\t4_part2_b6\side_20240906170655.jpg",
                            save=True,
                            show=True,
                            device="0"
                            )