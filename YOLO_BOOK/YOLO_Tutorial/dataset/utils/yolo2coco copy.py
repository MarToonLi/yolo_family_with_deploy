import argparse
import json
import os
import sys
import cv2
from datetime import datetime

# 初始化COCO格式的字典
coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []
json_path = ""

# 初始化图片和标注的ID
image_id = 000000
annotation_id = 0

def addCatItem(category_dict):
    for k, v in category_dict.items():
        category_item = {'supercategory': 'none', 'id': int(k), 'name': v}
        coco['categories'].append(category_item)

def addImgItem(subset_name, file_name, size):
    global image_id
    image_id += 1
    image_item = {
        'id': image_id,
        'subset_name': subset_name,
        'file_name': file_name,
        'width': size[1],
        'height': size[0],
        'license': None,
        'flickr_url': None,
        'coco_url': None,
        'date_captured': str(datetime.today())
    }
    coco['images'].append(image_item)
    return image_id

def addAnnoItem(image_id, category_id, bbox):
    global annotation_id
    annotation_id += 1
    seg = [bbox[0], bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[0] + bbox[2], bbox[1]]
    annotation_item = {
        'segmentation': [seg],
        'area': bbox[2] * bbox[3],
        'iscrowd': 0,
        'ignore': 0,
        'image_id': image_id,
        'bbox': bbox,
        'category_id': category_id,
        'id': annotation_id
    }
    coco['annotations'].append(annotation_item)

def xywhn2xywh(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    w = bbox[2] * size[1]
    h = bbox[3] * size[0]
    return [int(xmin), int(ymin), int(w), int(h)]

def parseXmlFiles(image_path, anno_path):
    assert os.path.exists(image_path), f"ERROR {image_path} does not exist."
    assert os.path.exists(anno_path), f"ERROR {anno_path} does not exist."

    subset_name = os.path.basename(image_path)

    images = [os.path.join(image_path, i) for i in os.listdir(image_path) if i.lower().endswith(('.jpg', '.png'))]
    files = [os.path.join(anno_path, i) for i in os.listdir(anno_path) if i.endswith('.txt') and not i.startswith('classes')]
    
    from tqdm import tqdm
    for file in tqdm(files):
        filename = os.path.basename(file)[:-4]
        img_file = next((img for img in images if img.endswith(filename + '.jpg') or img.endswith(filename + '.png')), None)
        if img_file:
            img = cv2.imread(img_file)
            shape = img.shape
            current_image_id = addImgItem(subset_name, os.path.basename(img_file), shape)
            with open(file, 'r') as fid:
                for line in fid.readlines():
                    parts = line.strip().split()
                    category = int(parts[0])
                    bbox = xywhn2xywh(parts[1:5], shape)
                    addAnnoItem(current_image_id, category, bbox)



def parseXmlFiles_root(root_image_path, root_anno_path, save_path, json_name='train.json'):
    subsets = os.listdir(root_image_path)
    subset_image_path = [os.path.join(root_image_path, subset ) for subset in subsets]
    subset_anno_path = [os.path.join(root_anno_path, subset ) for subset in subsets]
    json_path = os.path.join(save_path, "all.json")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(json_path)
        
    # 读取类别信息
    with open(os.path.join(root_anno_path, 'classes.txt'), 'r') as f:
        categories = [line.strip() for line in f.readlines()]
    category_id = dict((k, v) for k, v in enumerate(categories))
    addCatItem(category_id)
    
    for i in range(len(subset_image_path))[:]:
        print("Dealing:{}".format(subsets[i]))
        parseXmlFiles(subset_image_path[i], subset_anno_path[i])
    
    # 保存JSON
    
    with open(json_path, 'w') as f:
        json.dump(coco, f)
    print(f"Converted: {len(coco['categories'])} categories, {len(coco['images'])} images, {len(coco['annotations'])} annotations")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO to COCO format converter.")
    parser.add_argument('--anno-path', type=str,default=r'F:\Projects\datasets\Hellen-Apple\3_7_yolo\labels',  required=False, help='Directory path to YOLO annotations')
    parser.add_argument('--image-path', type=str,default=r'F:\Projects\datasets\Hellen-Apple\3_7_yolo\images', required=False, help='Directory path to images')
    parser.add_argument('--save-path', type=str,default=r'F:\Projects\datasets\Hellen-Apple\3_7_yolo\annotations',  required=False, help='Directory path to save COCO formatted annotation')
    parser.add_argument('--json-name', type=str, default='train.json', help='Output JSON file name')

    args = parser.parse_args()

    parseXmlFiles_root(args.image_path, args.anno_path, args.save_path, args.json_name)
    # 务必确保 classes.txt文件存在于标注文件夹中，并且正确列出所有类别。


# https://blog.csdn.net/heart_warmonger/article/details/142036018#:~:text=COCO%EF%BC%88Com
# 仅限于一个子数据集