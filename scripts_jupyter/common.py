from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import os
import matplotlib.patches as patches
from PIL import Image
import matplotlib.pyplot as plt



def count_targets_in_txt_files(directory: str, debug: bool) -> dict:
    """
    在指定目录（及子目录）下查找所有 .txt 文件，
    每行格式为：类别索引, x, y, w, h
    统计：
    - 所有目标的总数
    - 每个类别目标的数量

    Args:
        directory (str): 要搜索的根目录路径

    Returns:
        dict: {
            "total_targets": int,           # 所有目标的总数
            "class_counts": dict,           # 每个类别的目标数，如 {"0": 10, "1": 5}
            "processed_files": int          # 处理了多少个txt文件（可选，调试用）
        }
    """
    root_path = Path(directory)

    print(f"[INFO] 正在扫描目录：{root_path}，寻找 .txt 文件...")
    if not root_path.exists():
        print(f"[ERROR] 目录不存在：{root_path}")
        return {"total_targets": 0, "class_counts": {}, "processed_files": 0}
    if not root_path.is_dir():
        print(f"[ERROR] 提供的路径不是目录：{root_path}")
        return {"total_targets": 0, "class_counts": {}, "processed_files": 0}

    # 初始化统计变量
    total_targets = 0
    class_counts = defaultdict(int)  # 如 {"0": 5, "1": 3}
    class_paths = defaultdict(list)
    processed_files = 0

    # 查找所有 .txt 文件（递归）
    txt_files = list(root_path.rglob("*.txt"))

    print(f"[INFO] 共找到 {len(txt_files)} 个 .txt 文件，开始解析...")

    for txt_file in tqdm(txt_files, desc="正在处理 TXT 文件"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()  # 去掉首尾空格/换行
                    if not line:
                        continue  # 跳过空行

                    parts = line.split(' ')  # 按逗号分割
                    if len(parts) < 1:
                        continue  # 无效行，至少得有个类别索引

                    class_index = parts[0].strip()  # 第1个字段是类别索引，去掉空格

                    # 尝试转为 int（如果类别是数字，比如 0, 1, 2... 更规范）
                    # 如果类别可能是字符串如 "cat", "dog"，则无需转换，直接使用 class_index
                    # 这里我们假设类别是数字索引，比如 "0", "1"，但保留为字符串形式统计
                    # 如果你想转为整数统计，可以取消下行注释：
                    # class_index = int(class_index)
                    
                    try:
                        int(class_index)
                    except:
                        if debug: print(txt_file, class_index)
                        continue

                    class_counts[class_index] += 1
                    class_paths[class_index].append(txt_file)
                    total_targets += 1
        except Exception as e:
            print(f"[警告] 无法读取文件 {txt_file}，原因：{e}")
            continue

        processed_files += 1

    print(f"[INFO] 解析完成，共处理 {processed_files} 个 .txt 文件。")
    return {
        "total_targets": total_targets,
        "class_counts": dict(class_counts),  # 转为普通字典
        "class_paths": dict(class_paths),  # 转为普通字典
        "processed_files": processed_files
    }


def test_count_targets_in_txt_files(search_directory=r"/ns_data/datasets/3-7_jpg", debug=True):
    print("=" * 60)
    print("[TEST] 开始统计 .txt 文件中的目标类别分布")
    result = count_targets_in_txt_files(search_directory, debug=debug)

    total = result.get("total_targets", 0)
    class_counts = result.get("class_counts", {})
    class_paths = result.get("class_paths", {})
    processed = result.get("processed_files", 0)

    print(f"\n[统计结果]")
    print(f"总目标数: {total}")
    print(f"处理的 TXT 文件数: {processed}")
    print(f"每个类别的目标数量:")
    for class_id, count in sorted(class_counts.items(), key=lambda x: int(x[0])):  # 按类别ID排序
        print(f"  类别 {class_id}: {count} 个目标")


    print(f"每个类别的代表性样本路径:")
    for class_id, paths in sorted(class_paths.items(), key=lambda x: int(x[0])):  # 按类别ID排序
        print(f"  类别 {class_id}:")
        for path in paths[:5]:
            print(f"      {path}")
        
    # 可选：你也可以按数量倒序输出
    # sorted_counts = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    # for class_id, count in sorted_counts:
    #     print(f"  类别 {class_id}: {count} 个目标")
    
    return result


def print_directory_tree(startpath: str = r"/ns_data/datasets/3-7_jpg", prefix: str = ""):
    """
    打印目录树形结构（类似 tree 命令的文本版）
    
    Args:
        startpath (str): 要显示树形结构的目录路径
        prefix (str): 用于缩进的前缀（递归时使用，用户一般不用填）
    """
    if not os.path.isdir(startpath):
        print(f"[错误] 路径不是目录或不存在：{startpath}")
        return

    # 当前目录名
    title = os.path.basename(startpath) if os.path.basename(startpath) != "" else startpath
    # print(f"{prefix}📁 {title}/")

    try:
        entries = sorted(os.listdir(startpath))
    except PermissionError:
        print(f"{prefix}    [权限不足，无法访问]")
        return

    entries = [e for e in entries if not e.startswith('.')]  # 可选：忽略隐藏文件/目录

    for i, entry in enumerate(entries):
        path = os.path.join(startpath, entry)
        is_last = i == len(entries) - 1

        if os.path.isdir(path):
            # 是目录
            extension = "    " if is_last else "│   "
            files_num = len(os.listdir(path))
            print(f"{prefix}{'' if is_last else '│'}   └── 📁 {entry}/ ==> {files_num}")
            next_prefix = prefix + ("" if is_last else "│   ")
            print_directory_tree(path, next_prefix)
        else:
            pass
            # 是文件（可选：不显示文件，只显示目录）
            # extension = "    " if is_last else "│   "
            # print(f"{prefix}{'' if is_last else '│'}   └── 📄 {entry}")




def count_files_and_folders_per_directory(root_dir: str) -> dict:
    """
    统计根目录下：
    - 每个子目录的：
        - 文件夹数量（子目录数）
        - 文件数量
        - 各类文件后缀统计（仅限该目录）
    - 整个根目录下：
        - 所有文件的总数
        - 各类文件后缀的总数量（跨所有目录统计）

    Args:
        root_dir (str): 数据集或目录的根路径

    Returns:
        dict: {
            "global_file_count": int,          # 所有文件的总数
            "global_suffix_counts": dict,      # 所有文件中，每种后缀的总数，如 {".jpg": 120, ".txt": 45}
            "per_directory": {                 # 按目录统计
                "子目录相对路径": {
                    "folder_count": int,       # 该目录下的子目录数
                    "file_count": int,         # 该目录下的文件数
                    "suffix_counts": dict      # 该目录下各后缀的文件数，如 {".jpg": 5}
                },
                ...
            }
        }
    """
    root_path = Path(root_dir)

    print(f"[INFO] 正在扫描目录：{root_path}")
    if not root_path.exists():
        print(f"[ERROR] 目录不存在：{root_path}")
        return {}
    if not root_path.is_dir():
        print(f"[ERROR] 提供的路径不是目录：{root_path}")
        return {}

    print("[INFO] 目录校验通过，开始统计文件与文件夹信息...")
    result_per_directory = {}  # 按目录统计
    global_file_count = 0      # 全局文件总数
    global_suffix_counts = defaultdict(int)  # 全局后缀统计

    all_dirs = list(root_path.rglob('*'))  # 遍历根目录下所有文件和目录

    for dir_path in tqdm(all_dirs, desc="正在分析子目录"):
        if not dir_path.is_dir():
            continue  # 只处理目录

        # 当前目录的相对路径
        try:
            rel_dir_path = str(dir_path.relative_to(root_path))
        except ValueError:
            rel_dir_path = str(dir_path)

        # 统计该目录下的文件夹数、文件数、后缀统计
        folder_count = 0
        file_count = 0
        suffix_counts = defaultdict(int)

        for item in dir_path.iterdir():  # 遍历该目录下的直接子项
            if item.is_dir():
                folder_count += 1
            elif item.is_file():
                file_count += 1
                global_file_count += 1  # 累计全局文件数

                suffix = item.suffix.lower()  # 如 '.jpg'
                if suffix:  # 有后缀才统计
                    suffix_counts[suffix] += 1
                    global_suffix_counts[suffix] += 1  # 累计全局后缀数

        # 保存该目录的统计信息
        result_per_directory[rel_dir_path] = {
            "folder_count": folder_count,
            "file_count": file_count,
            "suffix_counts": dict(suffix_counts)
        }

    # 返回：全局统计 + 按目录统计
    return {
        "global_file_count": global_file_count,
        "global_suffix_counts": dict(global_suffix_counts),  # 转为普通字典
        "per_directory": result_per_directory
    }


def test_count_files_and_folders_per_directory(dataset_root=r"/ns_data/datasets/3-7_jpg"):
    print("=" * 60)
    print("[TEST] 开始统计目录下的文件夹、文件及后缀分布信息")
    stats = count_files_and_folders_per_directory(dataset_root)

    # 打印全局统计
    global_count = stats.get("global_file_count", 0)
    global_suffix = stats.get("global_suffix_counts", {})
    print(f"\n[全局统计] 根目录及所有子目录下的 文件总数: {global_count}")
    print("  各类文件后缀（全局）统计:")
    for suffix, cnt in global_suffix.items():
        print(f"    {suffix}: {cnt} 个")

    # 打印每个目录的统计
    per_dir_stats = stats.get("per_directory", {})
    for dir_name, counts in per_dir_stats.items():
        temp_str = "{"
        for suffix, cnt in counts['suffix_counts'].items():
            temp_str += f"{suffix}: {cnt};"
        temp_str += "}"
        print(f"[目录: {dir_name}];  子目录数: {counts['folder_count']};  文件数: {counts['file_count']};  各类文件后缀统计: {temp_str}.")





def convert_label_path_to_image_path(label_path: str) -> str | None:
    """
    将标签文件路径（如 .../labels/.../xxx.txt）转换为可能的图像路径（.../images/.../xxx.png 或 .jpg）

    规则：
    1. 把路径中的 '/labels/' 替换为 '/images/'
    2. 去掉原后缀 '.txt'
    3. 在新的路径下，查找是否存在同名的 .png 或 .jpg 文件
    4. 优先返回 .png，其次 .jpg，如果都不存在，返回 None
    """
    path = Path(label_path)

    # 1. 替换 labels -> images
    try:
        image_path = path.parent.parent.joinpath('images', path.parent.name, path.stem)
        # 说明：
        # - path.parent: 比如 .../labels/0923
        # - path.parent.parent: .../labels
        # - 我们要替换的是路径中包含的 'labels' 部分为 'images'
        # 更通用的方法是直接替换路径字符串中的 '/labels/' -> '/images/'
    except Exception:
        return None

    # 更推荐直接用字符串替换（更简单、更通用，不管路径层级怎么变）
    path_str = str(path)
    image_path_str = path_str.replace('/labels/', '/images/')
    image_path = Path(image_path_str).with_suffix('')  # 去掉 .txt，只保留主干

    # 2. 尝试匹配 .png 或 .jpg
    possible_extensions = ['.png', '.jpg']
    for ext in possible_extensions:
        candidate = image_path.with_suffix(ext)
        if candidate.exists():
            return str(candidate)

    # 找不到对应图像
    return None


def parse_label_line(line: str):
    """
    解析单行 label 数据，格式：类别索引, x, y, w, h
    返回：class_id, x_center, y_center, width, height（均为 float，且是归一化值）
    """
    parts = line.strip().split(' ')
    if len(parts) != 5:
        return None
    try:
        class_id = int(parts[0].strip())
        x_center = float(parts[1].strip())
        y_center = float(parts[2].strip())
        width = float(parts[3].strip())
        height = float(parts[4].strip())
        return class_id, x_center, y_center, width, height
    except ValueError:
        return None


def display_images_from_label_paths(label_paths: list[str]):
    """
    输入：一个包含 N 条 label txt 文件路径的列表
    功能：将每条路径尝试转换为对应的图像路径，并展示所有能找到的图像
    """
    label_paths = label_paths[:5]
    
    images_found = []
    for i, label_path in enumerate(label_paths):
        image_path = convert_label_path_to_image_path(label_path)
        if image_path:
            print(f"[{i+1}/{len(label_paths)}] 找到图像: {image_path}")
            images_found.append((image_path, label_path))
        else:
            print(f"[{i+1}/{len(label_paths)}] 未找到对应图像: {label_path}")

    # 3. 展示所有找到的图像
    if not images_found:
        print("❌ 没有找到任何有效的图像文件。")
        return

    print(f"\n🖼️ 共找到 {len(images_found)} 张图像，开始展示...")
    num_cols = 2  # 每行显示几张图
    num_rows = (len(images_found) + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 5 * num_rows))
    for idx, (img_path, label_path) in enumerate(images_found):
        plt.subplot(num_rows, num_cols, idx + 1)
        img = Image.open(img_path)
        img_width, img_height = img.size
        plt.imshow(img)
        plt.title(f"Image {idx+1}\n{Path(img_path).parent}", fontsize=8)
        plt.axis('off')
        
                # 读取对应的 label 文件
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"成功读取到标签文件 {label_path}: {len(lines)}")
        except Exception as e:
            print(f"⚠️ 无法读取标签文件 {label_path}: {e}")
            continue

        # 绘制每个目标框
        for line in lines:
            parsed = parse_label_line(line)
            if not parsed:
                print("parsed failed.")
                continue

            class_id, x_center_norm, y_center_norm, width_norm, height_norm = parsed

            # 反归一化
            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            width = width_norm * img_width
            height = height_norm * img_height

            x_min = x_center - width / 2
            y_min = y_center - height / 2
            x_max = x_center + width / 2
            y_max = y_center + height / 2
            
            print((x_min, y_min), width, height)

            # 绘制矩形框
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=1, edgecolor='red', facecolor='none'
            )
            plt.gca().add_patch(rect)

            # 可选：在框左上角显示类别 ID
            plt.text(
                x_min - 10, y_min - 10, f'{class_id}',
                color='red', 
                fontsize=10, 
                # weight='bold',
                # verticalalignment='top', 
                # backgroundcolor='white'
            )
        
        
    plt.tight_layout()
    # plt.show()
    
    return plt




if __name__ == "__main__":
    test_count_files_and_folders_per_directory()
    
    test_count_targets_in_txt_files()
    
    print_directory_tree()
    