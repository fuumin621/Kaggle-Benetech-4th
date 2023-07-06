import pandas as pd
import numpy as np
import cv2
import os
import json
from collections import defaultdict, OrderedDict
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")
# -

json_files = glob("./../input/benetech-making-graphs-accessible/train/annotations/*.json")

box_length = 5

os.makedirs("./../input/bb_images", exist_ok=True)

# # Plot-bbで切り出し&plot点のGT作成

# +
bbox_all = [ ]

for file in tqdm(json_files):
    with open(file) as fp:
        data = json.load(fp)
    id = file.split("/")[-1].split(".")[0]
    
    source = data["source"]
    chart_type = data["chart-type"]
    
    if chart_type != "scatter":
        continue
    
    bbox_list = {}
    bbox_list["bbox"] = []
    
    plot_bb = data["plot-bb"]
    
    bb_x = plot_bb["x0"]
    bb_y = plot_bb["y0"]
    bb_w = plot_bb["width"]
    bb_h = plot_bb["height"]
    
    img = cv2.imread(f"./../input/benetech-making-graphs-accessible/train/images/{id}.jpg")
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bb_img = img[bb_y:bb_y+bb_h, bb_x:bb_x+bb_w]
    
    cv2.imwrite( f"./../input/bb_images/{id}.jpg", bb_img)
        
    points = data["visual-elements"]["scatter points"][0]
    
    for p in points:
        x = p["x"]
        y = p["y"]
        
        x0 = x-box_length/2 - bb_x
        y0 = y-box_length/2 - bb_y
        x1 = x+box_length/2 - bb_x
        y1 = y+box_length/2 - bb_y
        
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(bb_w, x1)
        y1 = min(bb_h, y1)
        
        
        bbox_list["bbox"].append(["point", x0, y0, x1, y1])
    
    bbox_list["id"] = id
    bbox_list["source"] = source
    #plot_annotated_image(id, gt_points)
    
    bbox_all.append(bbox_list)

print(len(bbox_all))

# +
# count = 0

# for sample in bbox_all:
#     if sample["source"] == "generated":
#         continue
    
#     id = sample["id"]
#     img_path = f"./../input/bb_images/{id}.jpg"
#     img = Image.open(img_path)
#     draw = ImageDraw.Draw(img)
#     bbox = sample["bbox"]

#     for box in bbox:
#         #print(box)
#         draw.rectangle([(box[1], box[2]), (box[3], box[4])], outline = "Red")
#     plt.imshow(np.array(img))
#     plt.show()
    
#     count += 1
#     if count == 50:
#         break
# -

# # ICDARデータセットから

# scatter-lineも含める -> task6がなかった・・・
icdar_files = glob("./../input/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/annotations_JSON/scatter*/*.json")
len(icdar_files)

IMG_PATH = "./../input/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/scatter"

point_th = 20000
cat_th = 500
bb_margin = 0

# +
# icdar_bbox

# +
icdar_id = []
icdar_bbox = []
count1 = 0
count2 = 0

for file in tqdm(icdar_files):
    #id = file.split("/")[-1].split(".")[0]
    id = file.split("/")[-1][:-5]
    
    if "scatter-line" in file:
        IMG_PATH = "./../input/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/scatter-line"
    else:
        IMG_PATH = "./../input/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/scatter"
     
    with open(file) as fp:
        data = json.load(fp)
        
    path = os.path.join(IMG_PATH, id + ".jpg")
    
    if not os.path.exists(path):
        print(path)
        continue
    
    if len(data) < 6 or data["task6"] is None:
        count1 += 1
        continue
    #print(data["task6"]["output"]["visual elements"]["scatter points"])
    scatters = data["task6"]["output"]["visual elements"]["scatter points"]
    total = sum([len(s) for s in scatters])
    if total > point_th:
        continue
    category = len(scatters)
    if category > cat_th:
        continue
    
    bbox_list = {}
    bbox_list["bbox"] = []
        
    plot_bb = data["task6"]["input"]["task4_output"]["_plot_bb"]
    
    bb_x = plot_bb["x0"] - bb_margin
    bb_y = plot_bb["y0"] - bb_margin
    bb_w = plot_bb["width"] + bb_margin*2
    bb_h = plot_bb["height"] + bb_margin*2

    if bb_w < 5 or bb_h < 5 or bb_x < 0 or bb_y < 0:
        count2 += 1
        continue
    
    img = cv2.imread(path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bb_img = img[bb_y:bb_y+bb_h, bb_x:bb_x+bb_w]
    
#     plt.imshow(bb_img)
#     plt.show()
    
    cv2.imwrite( f"./../input/bb_images/{id}.jpg", bb_img)
        
    points = data["task6"]["output"]["visual elements"]["scatter points"]
    
    for point_group in points:
        for p in point_group:
            x = p["x"]
            y = p["y"]

            x0 = x-box_length/2 - bb_x
            y0 = y-box_length/2 - bb_y
            x1 = x+box_length/2 - bb_x
            y1 = y+box_length/2 - bb_y

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(bb_w, x1)
            y1 = min(bb_h, y1)


            bbox_list["bbox"].append(["point", x0, y0, x1, y1])
    
    icdar_id.append(id)
    icdar_bbox.append(bbox_list)

# -

print(count1)
print(count2)

len(icdar_id)


# +
# count = 0

# for id, sample in zip(icdar_id, icdar_bbox):
#     print(id)
#     img_path = f"./../input/bb_images/{id}.jpg"
#     img = Image.open(img_path)
#     draw = ImageDraw.Draw(img)
#     bbox = sample["bbox"]

#     for box in bbox:
#         #print(box)
#         draw.rectangle([(box[1], box[2]), (box[3], box[4])], outline = "Red")
#     plt.imshow(np.array(img))
#     plt.show()
    
#     count += 1
#     if count == 50:
#         break
# -

# # bboxのGTデータをcoco形式に変換

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(NumpyEncoder, self).encode(obj)


categories_id = {
    "point":1,
}

df = pd.read_csv("./../input/benetech-split/benetech_split.csv")

val_id = df[df["fold"] == 0]["id"].to_list()
val_id = set(val_id)
len(val_id)

categories = []
for k, v in categories_id.items():
    sample = {}
    sample["id"] = v
    sample["name"] = k
    categories.append(sample)

# train_json

# ## Benetech

# +
images = []
annotations = []

img_id = 1
box_id = 1

for sample in tqdm(bbox_all):
    id = sample["id"]
    
    if id in val_id:
        continue
    
    # image info
    image = {}
    image["id"] = img_id
    image["file_name"] = f"{id}.jpg"
    img = cv2.imread(f"./../input/bb_images/{id}.jpg")
    h, w = img.shape[:2]
    image["width"] = w
    image["height"] = h
    images.append(image)
    
    # bbox info
    bboxes = sample["bbox"]
    
    for box in bboxes:
        annotation = {}
        
        label = box[0]
        x0 = box[1]
        y0 = box[2]
        width = box[3] - box[1]
        height = box[4] - box[2]
        
        annotation["id"] = box_id
        box_id += 1
        annotation["image_id"] = img_id
        annotation["category_id"] = categories_id[label]
        annotation["area"] = width*height
        annotation["iscrowd"] = 0
        annotation["bbox"] = [x0, y0, width, height]
        
        annotations.append(annotation)
        
    img_id += 1
# -

# ## ICDAR

for id, sample in tqdm(zip(icdar_id, icdar_bbox)):
    
    # image info
    image = {}
    image["id"] = img_id
    image["file_name"] = f"{id}.jpg"
    img = cv2.imread(f"./../input/bb_images/{id}.jpg")
    h, w = img.shape[:2]
    image["width"] = w
    image["height"] = h
    images.append(image)
    
    # bbox info
    bboxes = sample["bbox"]
    
    for box in bboxes:
        annotation = {}
        
        label = box[0]
        x0 = box[1]
        y0 = box[2]
        width = box[3] - box[1]
        height = box[4] - box[2]
        
        annotation["id"] = box_id
        box_id += 1
        annotation["image_id"] = img_id
        annotation["category_id"] = categories_id[label]
        annotation["area"] = width*height
        annotation["iscrowd"] = 0
        annotation["bbox"] = [x0, y0, width, height]
        
        annotations.append(annotation)
        
    img_id += 1

train_json = {}
train_json["images"] = images
train_json["annotations"] = annotations
train_json["categories"] = categories

os.makedirs("./../input/yolox_point/",exist_ok = True)
os.makedirs("./../input/yolox_point/annotations/",exist_ok = True)

with open("./../input/yolox_point/annotations/train.json", "w") as fp:
    json.dump(train_json, fp, indent=2, cls=NumpyEncoder)

# val_json

# +
images = []
annotations = []

img_id = 1
box_id = 1

for sample in tqdm(bbox_all):
    id = sample["id"]
    
    if id not in val_id:
        continue
    
    # image info
    image = {}
    image["id"] = img_id
    image["file_name"] = f"{id}.jpg"
    img = cv2.imread(f"./../input/bb_images/{id}.jpg")
    h, w = img.shape[:2]
    image["width"] = w
    image["height"] = h
    images.append(image)
    
    # bbox info
    bboxes = sample["bbox"]
    
    for box in bboxes:
        annotation = {}
        
        label = box[0]
        x0 = box[1]
        y0 = box[2]
        width = box[3] - box[1]
        height = box[4] - box[2]
        
        annotation["id"] = box_id
        box_id += 1
        annotation["image_id"] = img_id
        annotation["category_id"] = categories_id[label]
        annotation["area"] = width*height
        annotation["iscrowd"] = 0
        annotation["bbox"] = [x0, y0, width, height]
        
        annotations.append(annotation)
        
    img_id += 1
# -

val_json = {}
val_json["images"] = images
val_json["annotations"] = annotations
val_json["categories"] = categories

with open("./../input/yolox_point/annotations/val.json", "w") as fp:
    json.dump(val_json, fp, indent=2, cls=NumpyEncoder)

# # 学習画像をYOLOXの指定フォルダへ配置

os.makedirs("./../input/yolox_point/train2017",exist_ok = True)
os.makedirs("./../input/yolox_point/val2017",exist_ok = True)

# +
images = glob("./../input/bb_images/*.jpg")

import shutil

for path in tqdm(images):
    id = path.split("/")[-1].split(".")[0]
    if id in val_id:
        shutil.copy(path, "./../input/yolox_point/val2017")
    else:
        shutil.copy(path, "./../input/yolox_point/train2017")
# -




