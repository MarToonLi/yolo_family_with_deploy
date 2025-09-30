import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import matplotlib.pyplot as plt
#import wandb
#from ray import tune


def process(storage_path, exp_name, train_mnist):
    experiment_path = f"{storage_path}/{exp_name}"
    print(f"Loading results from {experiment_path}...")

    restored_tuner = tune.Tuner.restore(experiment_path, trainable=train_mnist)
    result_grid = restored_tuner.get_results()
    
    
    if result_grid.errors:
        print("One or more trials failed!")
    else:
        print("No errors!")
        
    for i, result in enumerate(result_grid):
        print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")
        
        
        

    for i, result in enumerate(result_grid):
        plt.plot(
            result.metrics_dataframe["training_iteration"],
            result.metrics_dataframe["mean_accuracy"],
            label=f"Trial {i}",
        )

    plt.xlabel("Training Iterations")
    plt.ylabel("Mean Accuracy")
    plt.legend()
    plt.show()
 
if __name__ == '__main__':
    #wandb.login(key="efa8419f331992de149aa4358575237ed3e8704d")
    #wandb.init(project="YOLO-Tuning", entity="1437623218-dmu")
    
    model = YOLO(model = '/ns_data/projets/yolo_family_with_deploy/resources/models/yolov11/yolo11s.pt')
    result_grid  = model.tune(data=r"/ns_data/projets/yolo_family_with_deploy/yolov5_7.0/data/cable/apple_3_7_train_remote.yaml",
                # use_ray=True,
                iterations=300, 

                batch = 64,
                imgsz=640,
                epochs=50,
                device=[0, 1],
                
                #workers = 0,
                #gpu_per_trial = 2,
                )
    
    #results = model.tune(use_ray=True, iterations=20, data=r"/ns_data/projets/yolo_family_with_deploy/yolov5_7.0/data/cable/apple_3_7_train_remote.yaml")
 