# 清理一个指定目录下所有以npy结尾的文件
import os
import glob
import shutil


""" 查看root目录下所有以npy结尾的文件并删除 """
def delete_npy_files_glob(root_dir):
    npy_files = glob.glob(os.path.join(root_dir, "**/*.npy"), recursive=True)
    for file_path in npy_files:
        try:
            os.remove(file_path)
            print(f"已删除: {file_path}")
        except PermissionError:
            print(f"权限不足，无法删除: {file_path}")
        except Exception as e:
            print(f"删除失败 ({file_path}): {e}")


if __name__ == "__main__":
    root_dir = os.path.abspath(r"D:\Datasets\3-7_jpg_v2")
    delete_npy_files_glob(root_dir)