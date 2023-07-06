import os
import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Union, Tuple, Any
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import cv2
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
# -

# ### read_annotations

pd.set_option('display.max_colwidth', 1000)

percent_image_ids = []


# +
def series2cat(series):
    return [str(x) for x in series]

def series2num(series):
    return [float(x) for x in series]


# Must be True for every value in the series
def is_numerical(x):
    """Test whether input can be parsed as a Python float."""
    try:
        float(x)
        return True
    except ValueError:
        return False


def is_numerical_all(series):
    return all([is_numerical(x) for x in series])


# -

def get_gt_icadr_line(data,chart_type,margin = 5):
    df_ds = pd.DataFrame(data["task6"]["output"]['data series'][0]["data"])
    df_ve = pd.DataFrame(data["task6"]["output"]["visual elements"]["lines"][0])
    
    # interplは指定ピクセル分マージンをとっておく
    ve_x = df_ve["x"].tolist()
    ve_y = df_ve["y"].tolist()
    ve_x = [ve_x[0]-margin] + ve_x  + [ve_x[-1]+margin]
    ve_y = [ve_y[0]] + ve_y  + [ve_y[-1]]
    f_interp = interp1d(ve_x, ve_y)

    df_xaxis = pd.DataFrame(data["task4"]["output"]["axes"]["x-axis"])
    if len(df_xaxis)==0:
        return None
    df_text = pd.DataFrame(data["task2"]["output"]['text_blocks'])

    df_text = df_text.rename(columns={"text":"x_plot"})
    df_tgt = df_xaxis.merge(df_text[["id","x_plot"]],on=["id"],how="left")
    df_tgt["x_img"] = df_tgt["tick_pt"].apply(lambda x:x["x"])

    df_tgt = df_tgt[df_tgt["x_img"].between(min(ve_x),max(ve_x))].reset_index(drop=True)
    if len(df_tgt)==0:
        return None
    df_tgt["y_img"] = df_tgt["x_img"].apply(lambda x:f_interp(x))
    y_img2plot = LinearRegression().fit(df_ve["y"].values.reshape(-1,1), df_ds["y"])
    df_tgt["y_plot"] = y_img2plot.predict(df_tgt["y_img"].values.reshape(-1,1))
    all_x = df_tgt["x_plot"].astype(str).tolist()
    all_y = df_tgt["y_plot"].tolist()
    return {
        "x" : all_x,
        "y" : all_y,
    }


def get_gt_icadr_bar(data,chart_type):
    data_series = data["task6"]["output"]['data series'][0]["data"]
    all_x, all_y = [], []
    for d in data_series:
        x = d["x"]
        y = d["y"]
        # Ignore nan values
        if (str(x) == "nan" and isinstance(x, float)) or (
            str(y) == "nan" and isinstance(y, float)
        ):
            continue
        all_x.append(x)
        all_y.append(y)
    if chart_type in ["vertical_bar"]:
        all_x = series2cat(all_x)
        all_y = series2num(all_y)
    elif chart_type == "horizontal_bar":
        all_x,all_y = all_y,all_x # icdarはxとyが逆
        all_x = series2num(all_x)
        all_y = series2cat(all_y)
    else:
        ValueError
    return {
        "x" : all_x,
        "y" : all_y,
    }


def care_percent(data,gt,chart_type):
    df_axis = pd.DataFrame(data["task4"]["output"]["axes"]["y-axis"])
    if len(df_axis)==0:
        return gt
    df_text = pd.DataFrame(data["task2"]["output"]['text_blocks'])
    df_axis = df_axis.merge(df_text[["id","text"]],on=["id"],how="left")
    # ｙ_ticskに%含んでいれば100倍する
    if df_axis["text"].apply(lambda x:"%" in x).sum()> 0:
        percent_image_ids.append(data["image_id"])
        if chart_type in ["line","vertical_bar"]:
            gt["y"] = (np.array(gt["y"])*100).tolist()
        elif chart_type == "horizontal_bar":
            gt["x"] = (np.array(gt["x"])*100).tolist()
    return gt


target_types = ["horizontal_bar","vertical_bar","line"]
filepaths = glob("./../input/ICPR2022*/**/*.json", recursive=True)

gt_all = []
for path in tqdm(filepaths):
    image_id = path.split("/")[-1].split(".json")[0]
    with open(path) as fp:
        data = json.load(fp)
    data["image_id"] = image_id
    task6 = data.get("task6")
    if task6 is None:
        continue
    if task6["output"] is None:
        continue
    data_series = task6["output"]["data series"]
    if len(data_series)==0:# 系列が1つ以外のやつも使う
        continue
    series_len = len(data_series)
    data_series = data_series[0]["data"]
    chart_type = data["task1"]["output"]["chart_type"].replace(" ","_")
    if chart_type not in target_types:
        continue
    if "y" not in data_series[0].keys():
        # print(image_id,chart_type)
        continue
    if chart_type in ["line"]:
        gt = get_gt_icadr_line(data,chart_type)
    elif chart_type in ["horizontal_bar","vertical_bar"]:
        gt = get_gt_icadr_bar(data,chart_type)
    else:
        ValueError
    if gt is None:
        continue
    gt = care_percent(data,gt,chart_type)
    gt_all.append((image_id,gt["x"],gt["y"],chart_type,path,series_len))

df = pd.DataFrame(gt_all,columns = ["image_id","xs","ys","chart_type","json_path","series_len"])

image_paths = glob("./../input/ICPR2022*/**/*.jpg", recursive=True)
id2imgpath = {p.split("/")[-1].split(".jpg")[0]: p for p in image_paths}
df["image_path"] = df["image_id"].map(id2imgpath)

image_ids_to_rm = ['PMC5660682___fgene-08-00145-g0003']
df = df[~df["image_id"].isin(image_ids_to_rm)].reset_index(drop=True)
df["source"] = df["image_path"].apply(lambda x:"icdar2022_test" if "TEST" in x else "icdar2022_train")


# ### GT for DEPLOT

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


texts = []
for i in tqdm(range(len(df))):
    xs = df["xs"].values[i]
    ys = df["ys"].values[i]
    
    if len(xs) < len(ys):
        xs.extend(["nan"] * (len(ys) - len(xs)))
    elif len(xs) > len(ys):
        ys.extend(["nan"] * (len(xs) - len(ys)))

    xys = []
    for x,y in zip(xs,ys):
        x = round_float(x)
        y = round_float(y)
        xys.append(str(x) + "|" + str(y))
    text = "<0x0A>".join(xys) + "</s>"    
    texts.append(text)

df["text"] = texts
df["is_percentile"] = df["image_id"].isin(percent_image_ids)

df.to_csv("./../input/processed_df_icdar_v5.csv",index=False)






