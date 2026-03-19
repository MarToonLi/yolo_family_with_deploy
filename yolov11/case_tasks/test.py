from pathlib import Path
import shutil
from typing import Union, List, Tuple

"""
GitHub Copilot

在新文件中保存此代码，例如: /d:/ProjectsRelated/CoreProjects/yolo_family_with_deploy/yolov11/case_tasks/test.py

功能：
- split_train_val(train_txt: str | Path, images_dir: str | Path)
    将 train_txt 中出现的图像（按文件名或不带扩展的文件名匹配）移动到 images_dir/train，
    images_dir 下未在 train_txt 中出现的文件移动到 images_dir/val。
"""


def split_train_val(train_txt: Union[str, Path], images_dir: Union[str, Path]) -> Tuple[int, int, List[str]]:
        """
        参数:
            train_txt: train.txt 文件路径，包含要作为训练集的图像名（每行一个，可带或不带扩展名）
            images_dir: images 文件夹路径，里面包含图像文件

        返回:
            (moved_to_train, moved_to_val, missing_list)
        """
        train_txt = Path(train_txt)
        images_dir = Path(images_dir)

        if not train_txt.is_file():
                raise FileNotFoundError(f"train_txt not found: {train_txt}")
        if not images_dir.is_dir():
                raise NotADirectoryError(f"images_dir not found: {images_dir}")

        train_dir = images_dir / "train"
        val_dir = images_dir / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        # 读取 train.txt，去掉空行和注释
        with train_txt.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

        # 保留原始名和不带扩展名两种匹配方式
        wanted = set()
        wanted_stems = set()
        for ln in lines:
                name = Path(ln).name
                wanted.add(name)
                wanted_stems.add(Path(name).stem)

        # 列出 images_dir 下的文件（不递归），排除 train/val 目录本身
        files = [p for p in images_dir.iterdir() if p.is_file()]

        moved_train = 0
        moved_val = 0
        missing = []

        # 建立 stem->paths 映射（以便支持一个 stem 对应多种扩展的情况）
        stem_map = {}
        for p in files:
                stem_map.setdefault(p.stem, []).append(p)

        # 先处理 train 列表：按完整名匹配优先，再按 stem 匹配
        processed = set()
        for name in wanted:
                matched = []
                # 先尝试完整名匹配
                for p in files:
                        if p.name == name:
                                matched.append(p)
                # 若无完整名，再按 stem 匹配
                if not matched:
                        stem = Path(name).stem
                        matched = stem_map.get(stem, [])

                if matched:
                        for p in matched:
                                # 跳过已经移动的文件
                                if p.exists() and p.parent == images_dir:
                                        shutil.move(str(p), str(train_dir / p.name))
                                        moved_train += 1
                        processed.add(name)
                else:
                        missing.append(name)

        # 处理剩下未移动的文件，移动到 val（排除 train.txt 本身）
        for p in images_dir.iterdir():
                if not p.is_file():
                        continue
                if p.parent != images_dir:
                        continue
                # 不把 train.txt (如果放在 images_dir 下) 移动
                if p.resolve() == train_txt.resolve():
                        continue
                # 若已在 train_dir/val_dir 跳过
                if p.name in {".", ".."}:
                        continue
                # 如果文件不在 train_dir 且不在 val_dir，则移动到 val
                target = val_dir / p.name
                shutil.move(str(p), str(target))
                moved_val += 1

        return moved_train, moved_val, missing

# 简单示例（注释掉或按需修改路径）
if __name__ == "__main__":
        # 示例用法：
        moved_train, moved_val, missing = split_train_val(r"E:\Resources\datasets\JY\jingye_flysteel\train_jingye/train.txt", r"E:\Resources\datasets\JY\jingye_flysteel\train_jingye\labels")
        print(f"train: {moved_train}, val: {moved_val}, missing: {missing}")
        pass