import os
exp_id = "deplot_other"

save_dir = "./../output/" + exp_id.split(".")[0]
os.makedirs(save_dir, exist_ok=True)
import re
import json
from collections import Counter
from itertools import chain
from pathlib import Path
import random
from typing import List, Dict, Union
from glob import glob
from collections import defaultdict
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

from transformers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration, AutoConfig
from datasets import Dataset
from datasets import Image as ds_img
from polyleven import levenshtein # a faster version of levenshtein
import gc

import warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "true"


# -

class CFG:

    debug = False
    num_proc = 2
    num_workers = 14
    gpus = 1 # doesn't work yet with 2 gpus

    # Data
    max_length = 512
    
    model_name = "google/deplot"

    # Training
    epochs = 100
    decay_epoch = 50
    lr = 2e-4
    min_lr = 1e-6
    wd = 1e-5
    seed = 42
    batch_size = 8
    font_path="./../input/arial-font/arial.ttf"
    n_accumulate = 64//batch_size


# +
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# +
def round_float(value: Union[int, float, str]) -> Union[str, float]:
    if isinstance(value, float):
        value = "{:.8f}".format(value)

        if "." in value:
            integer, decimal = value.split(".")
            if abs(float(value)) < 0.001:
                decimal = decimal[:6]
            elif abs(float(value)) < 0.01:
                decimal = decimal[:5]
            elif abs(float(integer)) >= 100:
                return integer
            elif abs(float(integer)) < 100 and abs(float(integer)) >= 10:
                decimal = decimal[:1]
            elif abs(float(integer)) < 10 and abs(float(integer)) >= 1:
                decimal = decimal[:2]
            else:
                decimal = decimal[:4]

            value = integer + "." + decimal
    return value

def is_nan(value: Union[int, float, str]) -> bool:
    """
    Check if a value is NaN (not a number).

    Args:
        value (int, float, str): The value to check

    Returns:
        bool: True if the value is NaN, False otherwise
    """
    return isinstance(value, float) and str(value) == "nan"


# -

def get_gt_string_and_xy(filepath):

    with open(filepath) as fp:
        data = json.load(fp)

    data_series = data["data-series"]
    chart_type = data["chart-type"]
    
    id = filepath.split("/")[-1].split(".")[0]

    all_x, all_y = [], []
        
    for d in data_series:
        x = d["x"]
        y = d["y"]

        # Ignore nan values
        if  str(y) == "nan" and isinstance(y, float):
            y == ""
        
        if chart_type == "horizontal_bar":
            x = round_float(x)
        else:
            y = round_float(y)

        all_x.append(x)
        all_y.append(y)

    x_str =  ";".join(list(map(str, all_x))) 
    y_str = ";".join(list(map(str, all_y)))

    gt_string = ""
    for x, y in zip(all_x, all_y):
        gt_string += f"{x} | {y}<0x0A>"
    gt_string = gt_string[:-6]
    gt_string += processor.tokenizer.eos_token
        
    gt_string = gt_string.replace("\n", " ")
        
    return {
        "ground_truth": gt_string,
        "x": all_x,  # must have all lists be str rather than some str and some float
        "y": all_y,
        "chart-type": chart_type,
        "id": id,
        "source": data["source"],
    }


def change_anno(x):
    x = x.split("<0x0A>")[2:]
    x = "<0x0A>".join(x)
    return x


annotations = glob("./../input/benetech-making-graphs-accessible/train/annotations/*.json")

processor = Pix2StructProcessor.from_pretrained(CFG.model_name)

anno = defaultdict(list)
gt_chart = {}
for file in tqdm(annotations):
    sample = get_gt_string_and_xy(file)
    for k, v in sample.items():
        anno[k].append(v)
    gt_chart[sample["id"]] = sample["chart-type"]

# +
manual_find_percent = [
    "PMC2672396___ijerph-06-00915f2",
    "PMC3118237___1471-2458-11-302-2",
    "PMC3118237___1471-2458-11-302-3",
    "PMC3536649___pgen.1003192.g001_panel_4",
    "PMC3585459___1471-2458-12-1097-1",
    "PMC3877867___1471-2458-13-1236-2",
    "PMC4002537___1471-2458-14-365-1",
    "PMC5316188___12889_2017_4122_Fig1_HTML",
    "PMC5442292___IJPH-46-721-g001",
    "PMC5457578___12889_2017_4457_Fig2_HTML",
    "PMC5486351___ijerph-14-00665-g001",
    "PMC5580613___ijerph-14-00910-g002",
    "PMC5712041___fpubh-05-00299-g001",
    "PMC3766007___1755-8166-6-32-2",
    "PMC5135754___12889_2016_3887_Fig6_HTML",
    "PMC5400281___pgen.1006711.g001",
    "PMC5664653___ijerph-14-01152-g001",
    "PMC5907308___166_HTML",
]

no_numerical_label = [
    "PMC5224621___nanomaterials-06-00140-g008",
    "PMC5295199___nanomaterials-07-00009-g001",
    "PMC5344541___materials-10-00019-g005",
    "PMC5445839___materials-03-02447-g004",
    "PMC5445872___materials-03-02053-g001",
    "PMC5448635___materials-04-01132-g001",
    "PMC5452112___materials-06-00085-g008",
    "PMC5452770___materials-06-05410-g004",
    "PMC5453152___materials-07-00206f1",
    "PMC5455289___materials-08-00575-g001",
    "PMC5455936___materials-07-04321-g004",
    "PMC5456011___materials-07-07010-g002",
    "PMC5456616___materials-09-00795-g005",
    "MC5459190___materials-10-00144-g009",
    "PMC5485772___nanomaterials-07-00125-g009",
    "PMC5513583___materials-02-02337-g004",
    "PMC5706176___materials-10-01229-g004",
    "PMC5706176___materials-10-01229-g005",
]


# -

# いろいろなデータセットの小数点を統一する関数
def round_gt_string(row):
    # PMC3062570___g001はgtに" | "が入っている例外ケース
#     if row["id"] == "PMC3062570___g001":
#         return row["ground_truth"]
    
    gt_string_raw = row["ground_truth"]
    gt_string = gt_string_raw.split(processor.tokenizer.eos_token)[0]
    gt_string = gt_string.split("<0x0A>")
    x = [a.split(" | ")[0] for a in  gt_string]
    y= [a.split(" | ")[1] for a in  gt_string]
    try:
        if row["chart-type"] == "horizontal_bar":
            # %の補正
            if row["id"] in manual_find_percent:
                # print(row["id"])
                # print("correct percent")
                x = [float(xi)*100 for xi in x]
                # print(x)
            x = [round_float(float(xi)) for xi in x]
        else:
            if row["id"] in manual_find_percent:
                y = [float(yi)*100 for yi in y]
            y = [round_float(float(yi)) for yi in y]
    except Exception as e:
        # print(e)
        # print(row["id"])
        # print(row["chart-type"])
        # print(gt_string_raw)
        return gt_string_raw
    
    new_gt_string = []
    for xi, yi in zip(x, y):
        new_gt_string.append(f"{xi} | {yi}")
    new_gt_string = "<0x0A>".join(new_gt_string)
    new_gt_string += processor.tokenizer.eos_token
    return new_gt_string


deplot_df = pd.DataFrame(anno)

fold_df = pd.read_csv("./../input/benetech-split/benetech_split.csv")
fold_df = fold_df[["id", "fold"]]
df = pd.merge(fold_df, deplot_df, how="left", on="id")
df = df[df["chart-type"] != "scatter"].reset_index(drop=True)
df["image_path"] = df["id"].apply(lambda x: f"./../input/benetech-making-graphs-accessible/train/images/{x}.jpg")

syn_ver_df = pd.read_csv("./../input/benetech-synthesis-4th/graph_synthesis_vertical_bar_40000_ver11.csv")
syn_line_ex_df = pd.read_csv("./../input/benetech-synthesis-4th/graph_synthesis_line_40000_ver11.csv")
syn_hol_df = pd.read_csv("./../input/benetech-synthesis-4th/graph_synthesis_horizontal_bar_40000_ver12.csv")
syn_df = pd.concat([syn_line_ex_df, syn_hol_df, syn_ver_df]).reset_index(drop=True)
syn_df["image_path"] = "./." + syn_df["image_path"] # fix path

icdar_df = pd.read_csv("./../input/processed_df_icdar_v5.csv")
icdar_df = icdar_df.drop(["json_path"], axis=1)
icdar_df = icdar_df.rename(columns={'image_id': 'id', "xs":"x", "ys":"y", "chart_type":"chart-type", "text": "ground_truth"})
icdar_df["source"] = "icdar2022-train"
icdar_df = icdar_df[~icdar_df["id"].isin(no_numerical_label)].reset_index(drop=True)
icdar_df["fold"] = -3
icdar_df["ground_truth"] = icdar_df["ground_truth"].apply(lambda x: x.replace("|", " | ").replace("\n", " ").replace("\\n", " "))
icdar_df["ground_truth"] = icdar_df.apply(round_gt_string, axis=1)

# ## extra 500k graph
ex_df = pd.read_csv("./../input/benetech-extra-generated-data/metadata.csv")
ex_df = ex_df.rename(columns={'file_name': 'image_path', "text": "ground_truth", "chart_type":"chart-type"})
ex_df = ex_df[ex_df["chart-type"] != "scatter"].reset_index(drop=True)

ex_df["image_path"] = "./../input/benetech-extra-generated-data/" + ex_df["image_path"] 

ex_df["x"] = ""
ex_df["y"] = ""
ex_df["id"] = ex_df.index
ex_df["fold"] = -4
ex_df["source"] = "ex_500k"
ex_df["ground_truth"] = ex_df["ground_truth"].apply(lambda x: x.replace(" <0x0A> ", "<0x0A>"))
ex_df["ground_truth"] = ex_df.apply(round_gt_string, axis=1)

hist_df = pd.read_csv("./../input/benetech-synthesis-4th/graph_synthesis_histogram_1000_ver1.csv")
hist_df["fold"] = -5

df = df[df["chart-type"] != "scatter"].reset_index(drop=True)
generated_df = df[df["source"] == "generated"].reset_index(drop=True)
extracted_df = df[df["source"] == "extracted"].reset_index(drop=True)

# generatedとsynthesysの割合
icdar_num = 1000
generated_num = 1000
syns_num = 2000
ex_num = 1000
hist_num = 100
df = pd.concat([extracted_df, icdar_df[:icdar_num], generated_df[:generated_num], syn_df[:syns_num], ex_df[:ex_num], hist_df[:hist_num]]).reset_index(drop=True)

df = df[df["chart-type"] != "scatter"].reset_index(drop=True)


# +
def parse_deplot(text, id, chart_type):
    try:
        parse_text = text.split(processor.tokenizer.eos_token)[0]
        parse_text = parse_text.split("<0x0A>")
        x = [a.split(" | ")[0] for a in  parse_text]
        y= [a.split(" | ")[1] for a in  parse_text]
        if "" in y:
            pass
            # print(id)
        if chart_type == "horizontal_bar":
            x = [re.sub(r'[^0-9.-]', '', str(xi))for xi in x]
        else:
            y = [re.sub(r'[^0-9.-]', '', str(yi)) for yi in y if yi != ""]
    except Exception as e:
        # print(e)
        # print(text)
#         print(text)
#         print(parse_text)
        x = []
        y = []
    try:
        x_ = list(map(float, x))
        x = [xi for xi in x if not is_nan(float(xi))]
        x = list(map(str, x))
    except:
        x = list(map(str, x))
    try:
        y_ = list(map(float, y))
        y = [yi for yi in y if not is_nan(float(yi))]
        y = list(map(str, y))
    except:
        y = list(map(str, y))
        
#     x = ";".join(x)
#     y = ";".join(y)
    index = [f"{id}_x" ] + [f"{id}_y"] 
    pred_df = pd.DataFrame(index=index, data={"data_series": [x]+[y], "chart_type": chart_type})
    
    return pred_df


# -

# # First step is to transform the given annotations into a format the model can work with
#
# Since the predictions need to be in the form x1;x2;x3 and y1;y2;y3, let's make that the format we will generate. We also need to add special tokens to predict the chart type and separators to distinguish what is what.

# # Preprocessing function to get pixel values and input_ids
#
# When ever the `_getitem__` function is called for the dataset, it will run this function.

def preprocess(examples):
    pixel_values = []

    text = examples["ground_truth"]
    id = examples["id"]
    chart_type = examples["chart-type"]

    input_ids = processor.tokenizer.tokenize(text, add_special_tokens=False)
    input_ids = processor.tokenizer(
        text,
        add_special_tokens=False,
        max_length=CFG.max_length,
        padding=True,
        truncation=True,
    ).input_ids
    
    img_path = examples["image_path"]
    image = Image.open(img_path)
    sample = processor(images=image, text="Generate underlying data table of the figure below:", return_tensors="pt", font_path=CFG.font_path)
    #print(inputs)
    flattened_patches = sample.flattened_patches.squeeze()
    attention_mask = sample.attention_mask.squeeze()

    return {
        "flattened_patches": flattened_patches,
        "attention_mask": attention_mask,
        "labels": input_ids,
        "id": id,
        "chart-type": chart_type,
    }


class BeneTechDataset(Dataset):
    def __init__(self, df, processor, augments=None):
        self.dataset = df
        self.processor = processor
        self.augments = augments

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        source = self.dataset.iloc[idx]["source"]
        if source == "generated":
            item = generated_df.sample(n=1).squeeze()
        elif source == "synthesis":
            item = syn_df.sample(n=1).squeeze()
        elif source == "ex-500k":
            item = ex_df.sample(n=1).squeeze()
        elif source == "icdar2022-train":
            item = icdar_df.sample(n=1).squeeze()
        elif source == "histgram":
            item = hist_df.sample(n=1).squeeze()
        else:
            item = self.dataset.iloc[idx]
        
        example = preprocess(item)
        
        return example


dataset = BeneTechDataset(df, processor, None)

train_df = df.copy()
valid_df = df[df["fold"] >= 0].reset_index(drop=True)

# # Collate function to make sure the ids are all the same length in a batch

# +
pad_token_id = processor.tokenizer.pad_token_id

def collate_fn(samples):

    batch = {"flattened_patches":[], "attention_mask":[]}
    
    max_length = max([len(x["labels"]) for x in samples])

    # Make a multiple of 8 to efficiently use the tensor cores
    if max_length % 8 != 0:
        max_length = (max_length // 8 + 1) * 8

    labels = [
        x["labels"] + [pad_token_id] * (max_length - len(x["labels"]))
        for x in samples
    ]

    batch["labels"] = torch.tensor(labels)

    for item in samples:
        batch["flattened_patches"].append(item["flattened_patches"])
        batch["attention_mask"].append(item["attention_mask"])

    batch["flattened_patches"] = torch.stack(batch["flattened_patches"])
    batch["attention_mask"] = torch.stack(batch["attention_mask"])
#     batch["flattened_patches"] = samples[0]["flattened_patches"].unsqueeze(0)
#     batch["attention_mask"] = samples[0]["attention_mask"].unsqueeze(0)
    
    batch["id"] = [x["id"] for x in samples]
    batch["chart-type"] = [x["chart-type"] for x in samples]

    return batch


# -

# # Dataloaders
#
# Validation uses generation so it is very slow. That's why I only use a small fraction of the examples.

train_ds = BeneTechDataset(train_df, processor, None)
valid_ds = BeneTechDataset(valid_df, processor, None)

# +

train_dataloader = DataLoader(
    train_ds,
    batch_size=CFG.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    timeout = 10,
    num_workers=CFG.num_workers,
)
val_dataloader = DataLoader(
    valid_ds,
    batch_size=CFG.batch_size,
    shuffle=False,
    timeout = 10,
    collate_fn=collate_fn,
    num_workers=CFG.num_workers,
)

batch = next(iter(train_dataloader))


# -

# # Functions to calculate metrics

# +
def rmse(y_true: List[float], y_pred: List[float]) -> float:
    return np.sqrt(np.mean(np.square(np.subtract(y_true, y_pred))))


def sigmoid(x: float) -> float:
    return 2 - 2 / (1 + np.exp(-x))


def normalized_rmse(y_true: List[float], y_pred: List[float]) -> float:
    numerator = rmse(y_true, y_pred)
    denominator = rmse(y_true, np.mean(y_true))

    # https://www.kaggle.com/competitions/benetech-making-graphs-accessible/discussion/396947
    if denominator == 0:
        if numerator == 0:
            return 1.0
        return 0.0

    return sigmoid(numerator / denominator)


def normalized_levenshtein_score(y_true: List[str], y_pred: List[str]):
    total_distance = np.sum([levenshtein(yt, yp) for yt, yp in zip(y_true, y_pred)])
    length_sum = np.sum([len(yt) for yt in y_true])
    return sigmoid(total_distance / length_sum)


def score_series(
    y_true: List[Union[float, str]], y_pred: List[Union[float, str]]
) -> float:
    if len(y_true) != len(y_pred):
        return 0.0
    if isinstance(y_true[0], str):
        return normalized_levenshtein_score(y_true, y_pred)
    else:
        # Since this is a generative model, there is a chance it doesn't produce a float.
        # In that case, we return 0.0.
        try:
            return normalized_rmse(y_true, list(map(float, y_pred)))
        except:
            return 0.0


def benetech_score(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> float:
    """Evaluate predictions using the metric from the Benetech - Making Graphs Accessible.

    Parameters
    ----------
    ground_truth: pd.DataFrame
        Has columns `[data_series, chart_type]` and an index `id`. Values in `data_series`
        should be either arrays of floats or arrays of strings.

    predictions: pd.DataFrame
    """
    if not ground_truth.index.equals(predictions.index):
        raise ValueError(
            "Must have exactly one prediction for each ground-truth instance."
        )
    if not ground_truth.columns.equals(predictions.columns):
        raise ValueError(f"Predictions must have columns: {ground_truth.columns}.")
    pairs = zip(
        ground_truth.itertuples(index=False), predictions.itertuples(index=False)
    )
    scores = []
    for (gt_series, gt_type), (pred_series, pred_type) in pairs:
        if gt_type != pred_type:  # Check chart_type condition
            scores.append(0.0)
        else:  # Score with RMSE or Levenshtein as appropriate
            scores.append(score_series(gt_series, pred_series))

    ground_truth["score"] = scores

    grouped = ground_truth.groupby("chart_type", as_index=False)["score"].mean()

    chart_type2score = {
        chart_type: score
        for chart_type, score in zip(grouped["chart_type"], grouped["score"])
    }

    return np.mean(scores), chart_type2score


# -

# # Create model and ground truth dataframe

def get_gt(id):
    filepath = f"./../input/benetech-making-graphs-accessible/train/annotations/{id}.json"
    with open(filepath) as fp:
        data = json.load(fp)
        
    all_x = []
    all_y = []
    
    data_series = data["data-series"]
    for d in data_series:
        x = d["x"]
        y = d["y"]
        
        if data["axes"]["x-axis"]["values-type"] == "numerical":
            x = float(x)
            y = y.replace("\n", " ")
        if data["axes"]["y-axis"]["values-type"] == "numerical":
            y = float(y)
            x = x.replace("\n", " ")

        # Ignore nan values
        if (str(x) == "nan" and isinstance(x, float)) or (
            str(y) == "nan" and isinstance(y, float)
        ):
            continue            

        all_x.append(x)
        all_y.append(y)
    
    return [id + "_x", id + "_y"], [all_x, all_y], [data["chart-type"]]*2


gt_id = valid_df["id"].to_list()
ids = []
xy = []
ct = []
for id in gt_id:
    i, x, c = get_gt(id)
    ids.extend(i)
    xy.extend(x)
    ct.extend(c)
gt_df = pd.DataFrame(index=ids, data={"data_series":xy, "chart_type":ct})

gt_df.to_csv(os.path.join(save_dir, "gt_df.csv"))


# # Train!

def train_one_epoch(model, optimizer, scheduler, dataloader):
    global best_score
    global best_line
    global best_vertical
    global best_horizontal
    
    model.train()
    scaler = GradScaler()
    
    dataset_size = 0
    running_loss = 0.0
    
    try:
    
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
        for step, batch in pbar:
    #         if step == 0:
    #             break
            flattened_patches = batch["flattened_patches"].to(device, dtype=torch.float)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device, dtype=torch.long)
            chart_type = batch["chart-type"]

            batch_size = flattened_patches.size(0)

            with autocast(enabled=True):
                outputs = model(flattened_patches=flattened_patches, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / CFG.n_accumulate

            scaler.scale(loss).backward()

            if (step + 1) % CFG.n_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                            lr=f'{current_lr:0.6f}',
                            gpu_mem=f'{mem:0.2f} GB')

            if step in [((i+1)*len(dataloader)//1)-1 for i in range(1)]:

                metrics = valid_one_epoch(model, processor, val_dataloader)

                val_score = metrics["score"]
                val_line = metrics["line"]
                val_vertical = metrics["vertical_bar"]
                val_horizontal = metrics["horizontal_bar"]

                if val_score > best_score:
                    print(f"Valid score improved ({best_score:0.4f} ---> {val_score:0.4f})")
                    best_score    = val_score
                    model.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)
                    print(f"Model Saved")

                model.train()
    except Exception as e:
        print(e)
        metrics = valid_one_epoch(model, processor, val_dataloader)

        val_score = metrics["score"]

        if val_score > best_score:
            print(f"Valid score improved ({best_score:0.4f} ---> {val_score:0.4f})")
            best_score    = val_score
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            print(f"Model Saved")
    
    #run.log({"train loss": epoch_loss, "lr": current_lr})
    gc.collect()


# +
@torch.no_grad()
def valid_one_epoch(model, processor, dataloader):    
    model.eval()
    
    val_outputs = []
    try:
        for step, batch in enumerate(dataloader):
            flattened_patches = batch["flattened_patches"].to(device, dtype=torch.float)
            attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device, dtype=torch.long)
            chart_type = batch["chart-type"]

            batch_size = flattened_patches.shape[0]

            outputs = model.generate(
                flattened_patches=flattened_patches,
                attention_mask=attention_mask,
                max_new_tokens=512,
                #max_length=CFG.max_length,
                use_cache=True,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                num_beams=1,
            )

            for i in range(batch_size):
                predictions = processor.decode(outputs[i], skip_special_tokens=True)
                if step == 0 and i == 0:
                    print(predictions)
                val_outputs.append(parse_deplot(predictions, batch["id"][i], chart_type[i]))
    except Exception as e:
        print(e)
                    
    val_df = pd.concat(val_outputs)
    
    #val_df.to_csv(os.path.join(save_dir, f"val_{epoch}_{train_step}.csv"))
    
#     print(gt_df.loc[val_df.index.values])
    score, metrics = benetech_score(gt_df.loc[val_df.index.values], val_df)
    metrics["score"] = score
    metrics["lr"] = optimizer.param_groups[0]['lr']
    
    print(metrics)
    
    return metrics


# -

model = Pix2StructForConditionalGeneration.from_pretrained(CFG.model_name)
processor = Pix2StructProcessor.from_pretrained(CFG.model_name)
model.to(device)
model.config.text_config.is_decoder=True

from torch.utils.checkpoint import checkpoint
model.encoder.gradient_checkpointing_enable()
model.decoder.gradient_checkpointing_enable()

# +
print(f'#'*15)
print(f'Traning starts')
print(f'#'*15)
    
seed_everything(42)

optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(train_dataloader)*CFG.decay_epoch, eta_min=CFG.min_lr)

best_score = -np.inf
best_line = -np.inf
best_vertical = -np.inf
best_horizontal = -np.inf

for epoch in range(1, CFG.epochs + 1):
    print()
    print(f"Epoch {epoch}")
    train_one_epoch(model, optimizer, scheduler, dataloader=train_dataloader)

    if epoch in [45]:
        tmp_dir = os.path.join(save_dir, f"epoch{epoch}")
        os.makedirs(tmp_dir, exist_ok=True)
        model.save_pretrained(tmp_dir)
        processor.save_pretrained(tmp_dir)
# -






