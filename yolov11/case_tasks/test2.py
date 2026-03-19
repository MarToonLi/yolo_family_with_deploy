from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def convert_labels(labels_dir: str, out_dir: str = None):
    """
    将 labels_dir 下的所有 PNG 图像中像素 (255,0,0) -> (1,0,0)，(0,0,0) 保持不变。
    转换后的图像保存在 labels_convert 文件夹（或 out_dir 指定的文件夹）下。
    """
    labels_path = Path(labels_dir)
    if out_dir is None:
        out_path = labels_path / "labels_index"
    else:
        out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for p in tqdm(labels_path.glob("*.png"), desc="处理标签图像", total=len(list(labels_path.glob("*.png")))):
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.uint8)

        # 判断精确颜色
        red_mask = np.all(arr == [255, 0, 0], axis=-1)

        # 构造输出数组，默认全 0（黑）
        out_arr = np.zeros((arr.shape[0], arr.shape[1]), dtype=np.uint8)
        out_arr[red_mask] = 1 # 红色位置设为 (1,0,0)

        out_img = Image.fromarray(out_arr, mode="L")
        out_img.save(out_path / p.name)

if __name__ == "__main__":
    # 简单示例：修改为你的 labels 文件夹路径后运行
    convert_labels(r"E:\Resources\datasets\JY\jingye_flysteel\train_jingye\labels")