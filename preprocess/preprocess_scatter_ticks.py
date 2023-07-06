import json
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

df = pd.read_csv("./../input/benetech-split/benetech_split.csv")

df["is_val"] = df["fold"] == 0

df = df[["id","type","source","is_val"]]

df.columns = ["image_id","chart_type","source","is_val"]
df = df[df["chart_type"]=="scatter"]


def get_edge_point(image_id):
    annt_fname = f"./../input/benetech-making-graphs-accessible/train/annotations/{image_id}.json"
    img_fname = f"./../input/benetech-making-graphs-accessible/train/images/{image_id}.jpg"
    data = json.load(open(annt_fname))

    text = pd.DataFrame(data["text"])
    x_ticks = pd.DataFrame(data['axes']["x-axis"]["ticks"])
    y_ticks = pd.DataFrame(data['axes']["y-axis"]["ticks"])
    if len(x_ticks)==0 or len(y_ticks)==0:
        return None
    x_ticks = x_ticks.merge(text[["id","text"]],on=["id"],how="left")
    y_ticks = y_ticks.merge(text[["id","text"]],on=["id"],how="left")
    x_min = x_ticks["text"].values[0]
    x_max = x_ticks["text"].values[-1]
    y_min = y_ticks["text"].values[-1]
    y_max = y_ticks["text"].values[0]
    return [x_min,y_min,x_max,y_max]


def round_float(x):
    if isinstance(x, float):
        x = str(x)

        if "." in x:
            integer, decimal = x.split(".")
            if abs(float(integer)) > 1:
                decimal = decimal[:1]
            else:
                decimal = decimal[:4]

            x = float(integer + "." + decimal)
    return x


def cv2_imshow(img_path):
    img = cv2.imread(img_path)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_numeric(x):
    # x = eval(x)
    try:
        x = [re.sub(r'[^0-9.-]', '', str(xi))for xi in x]
        res = [float(xi) for xi in x]
    except  ValueError:
        res = None
    return res


res = []
for image_id in tqdm(df["image_id"].tolist()):
    res.append(get_edge_point(image_id))

df["edge_points"] = res
df = df[df["edge_points"].notnull()].reset_index(drop=True)
df["edge_points_num"] = df["edge_points"].apply(get_numeric)
df = df[df["edge_points_num"].notnull()].reset_index(drop=True)

df["xs"] = df["edge_points_num"].apply(lambda x:[x[0],x[2]])
df["ys"] = df["edge_points_num"].apply(lambda x:[x[1],x[3]])

texts = []
for i in tqdm(range(len(df))):
    xs = df["xs"].values[i]
    ys = df["ys"].values[i]
    xys = []
    for x,y in zip(xs,ys):
        # x = round_float(x)
        # y = round_float(y)
        xys.append(str(x) + "|" + str(y))
    text = "<0x0A>".join(xys) + "</s>"    
    texts.append(text)
df["text"] = texts

df.to_pickle("./../input/gt_for_scatter_edge_ticks_v2.pkl")




