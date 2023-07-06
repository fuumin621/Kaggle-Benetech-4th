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

jsons = glob("./../input/benetech-making-graphs-accessible/train/annotations/*.json")


def try_parse(x, id, fn):
    x = x.replace(",", "").replace("%", "").replace("$", "")
    try:
        x = fn(x)
        return True
    except:
        print(id)
        print(x)
        return False


def clean_numeric(x):
    x = x.replace(",", "").replace("%", "").replace("$", "")
    return x


# +
role_set = set()

for file in tqdm(jsons):
    with open(file) as fp:
        data = json.load(fp)
        
    for d in data["text"]:
        role_set.add(d["role"])
        # if d["role"] in ["other", "tick_grouping", "legend_label"]:
            # print(file.split("/")[-1].split(".")[0])
            # print(d["role"])
# print(role_set)

# +
bbox_all = []
count = 0

for file in tqdm(jsons):
    with open(file) as fp:
        data = json.load(fp)
        
    bbox_list = {}
    bbox_list["bbox"] = []
    
    id = file.split("/")[-1].split(".")[0]

        
    source = data["source"]
    
    plot_bb = data["plot-bb"]
    bbox_list["bbox"].append(["plot-bb", plot_bb["x0"], plot_bb["y0"], plot_bb["x0"]+plot_bb["width"], plot_bb["y0"]+plot_bb["height"]])
    
    x_tick_points = []
    y_tick_points = []
    
    for axe in data["axes"]["x-axis"]["ticks"]:
        x = axe["tick_pt"]["x"]
        y = axe["tick_pt"]["y"] + 10
        x_tick_points.append([x, y])
    for axe in data["axes"]["y-axis"]["ticks"]:
        x = axe["tick_pt"]["x"] - 10
        y = axe["tick_pt"]["y"]
        y_tick_points.append([x, y])
    try:
        neigh = NearestNeighbors(n_neighbors=1)
        sample = x_tick_points + y_tick_points
        #print(sample)
        labels = ["x"] * len(x_tick_points) + ["y"] * len(y_tick_points)
        #print(labels)
        neigh.fit(sample)

        x_polygon= defaultdict(list)
        y_polygon = defaultdict(list)
        for d in data["text"]:
            
            if d["role"] in ["axis_title", "chart_title", "other"]:
                bbox_list["bbox"].append([d["role"], d["polygon"]["x0"], d["polygon"]["y0"], d["polygon"]["x2"], d["polygon"]["y2"]])
            
            elif d["role"] in ["tick_label", "tick_grouping"]:
                x = (d["polygon"]["x0"] + d["polygon"]["x2"])//2
                y = (d["polygon"]["y0"] + d["polygon"]["y2"])//2
                #print(x, y)

                idx = neigh.kneighbors([[x, y]], n_neighbors=1, return_distance=False)
                idx = idx[0][0]
                #print(idx)
                label = labels[idx]

                #print(text)
                #print()
                if label == "x":
                    for k, v in d["polygon"].items():
                        if "x" in k:
                            x_polygon["x"].append(v)
                        else:
                            x_polygon["y"].append(v)
                else:
                    for k, v in d["polygon"].items():
                        if "x" in k:
                            y_polygon["x"].append(v)
                        else:
                            y_polygon["y"].append(v)
    except Exception as e:
        print(id)
        print(e)
        print(sample)
        print(x, y)
        continue
        
    if len(x_polygon["x"]) == 0 or len(y_polygon["x"]) == 0:
        continue
    
    # x軸ボックス
    x_axis_x_min = np.min(x_polygon["x"])
    x_axis_y_min = np.min(x_polygon["y"])
    x_axis_x_max = np.max(x_polygon["x"])
    x_axis_y_max = np.max(x_polygon["y"])
    
    xbox = [x_axis_x_min, x_axis_y_min, x_axis_x_max, x_axis_y_max]
    
    # y軸ボックス
    y_axis_x_min = np.min(y_polygon["x"])
    y_axis_y_min = np.min(y_polygon["y"])
    y_axis_x_max = np.max(y_polygon["x"])
    y_axis_y_max = np.max(y_polygon["y"])
    
    ybox = [y_axis_x_min, y_axis_y_min, y_axis_x_max, y_axis_y_max]
    
    bbox_list["id"] = id
    bbox_list["bbox"].append(["x_label", *xbox])
    bbox_list["bbox"].append(["y_label", *ybox])
#     bbox_list["xbox"] = xbox
#     bbox_list["ybox"] = ybox
    bbox_list["source"] = source
    
    bbox_all.append(bbox_list)
    
#     count += 1
#     if count == 10:
#         break

# +
# count = 0

# for sample in bbox_all:
#     if sample["source"] == "generated":
#         continue
    
#     id = sample["id"]
#     img_path = f"./../input/benetech-making-graphs-accessible/train/images/{id}.jpg"
#     img = Image.open(img_path)
#     draw = ImageDraw.Draw(img)
#     bbox = sample["bbox"]

#     for box in bbox:
#         print(box)
#         draw.rectangle([(box[1], box[2]), (box[3], box[4])], outline = "Red")
#     plt.imshow(np.array(img))
#     plt.show()
    
#     count += 1
#     if count == 5:
#         break
# -

# # COCO formatへ変換

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
    "plot-bb":1,
    "chart_title":2,
    "axis_title":3,
    "x_label":4,
    "y_label":5,
    "other":6
}

df = pd.read_csv("./../input/benetech-split/benetech_split.csv")

val_id = df[df["fold"] == 0]["id"].to_list()
val_id = set(val_id)
len(val_id)

# train.json

categories = []
for k, v in categories_id.items():
    sample = {}
    sample["id"] = v
    sample["name"] = k
    categories.append(sample)

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
    img = cv2.imread(f"./../input/benetech-making-graphs-accessible/train/images/{id}.jpg")
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

train_json = {}
train_json["images"] = images
train_json["annotations"] = annotations
train_json["categories"] = categories

os.makedirs("./../input/yolox_plotbb/",exist_ok = True)
os.makedirs("./../input/yolox_plotbb/annotations/",exist_ok = True)

with open("./../input/yolox_plotbb/annotations/train.json", "w") as fp:
    json.dump(train_json, fp, indent=2, cls=NumpyEncoder)

# val.json

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
    img = cv2.imread(f"./../input/benetech-making-graphs-accessible/train/images/{id}.jpg")
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

with open("./../input/yolox_plotbb/annotations/val.json", "w") as fp:
    json.dump(val_json, fp, indent=2, cls=NumpyEncoder)

# # ### copy image

images = glob(f"./../input/benetech-making-graphs-accessible/train/images/*.jpg")

os.makedirs("./../input/yolox_plotbb/train2017",exist_ok = True)
os.makedirs("./../input/yolox_plotbb/val2017",exist_ok = True)

# +
import shutil

for path in tqdm(images):
    id = path.split("/")[-1].split(".")[0]
    if id in val_id:
        shutil.copy(path, "./../input/yolox_plotbb/val2017")
    else:
        shutil.copy(path, "./../input/yolox_plotbb/train2017")
# -








