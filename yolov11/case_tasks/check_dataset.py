import os
import cv2
from pathlib import Path
from tqdm import tqdm

def find_images_in_path(root_path):
    """
    在指定路径下递归查找所有图像文件，并返回它们的路径列表。
    
    Args:
        root_path (str): 要搜索的根目录路径。
        
    Returns:
        list: 所有图像文件的路径列表。
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']  # 支持的图像扩展名
    image_paths = []
    
    for root, _, files in os.walk(root_path):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def check_image_shapes(image_paths):
    """
    检查图像文件的 shape 是否为三维，如果不是则打印路径。
    
    Args:
        image_paths (list): 图像文件的路径列表。
    """
    for image_path in tqdm(image_paths, desc="检查图像 shape"):
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue
            
            if len(image.shape) != 3:
                print(f"非三维图像: {image_path}, shape: {image.shape}")
        except Exception as e:
            print(f"处理图像时出错: {image_path}, 错误: {e}")

if __name__ == "__main__":
    # 设置要检查的路径
    root_path = r"X:\ex_space\datasets\3-7-v1\images"  # 替换为你的路径
    
    # 查找所有图像文件
    image_paths = find_images_in_path(root_path)
    print(f"找到 {len(image_paths)} 张图像")
    
    # 检查图像 shape
    check_image_shapes(image_paths)