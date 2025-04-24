
from ultralytics import YOLO

if __name__ == '__main__':
    model_path = r"F:\Projects\yolo_family\runs\train\exp\weights\best.pt"
    data_path = r"F:\Projects\yolo_family\yolov5_7.0\data\cable\apple_3_7_val.yaml"
    
    
    # Load a model
    model = YOLO(model_path, verbose=True)  # load a custom model

    # Validate the model
    metrics = model.val(data=data_path, 
                        # save_txt = True, # 保存每个样本预测结果的txt文件
                        save_json = True,  # 保存每个样本预测结果到一个json文件，元素是以预测框为单位，而不是样本。
                        imgsz=640, 
                        batch=8, 
                        conf=0.25, 
                        iou=0.6, 
                        device="0")
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category