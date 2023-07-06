# %%
import os
class CFG:
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    debug = False
    num_workers = 12
    gpus = 1 # doesn't work yet with 2 gpus

    # Training
    epochs = 15
    val_check_interval = 1.0  # how many times we want to validate during an epoch
    check_val_every_n_epoch = 1
    gradient_clip_val = 1.0
    lr = 1e-5
    # lr = 1e-4
    seed = 42
    warmup_steps = 300  # 800/8*30/10 10%
    log_steps = 200
    batch_size = 4
    min_valid_epoch = 0
    decay_epoch = 4
    min_lr = 1e-6

# %%
if CFG.debug:
    CFG.exp_id += "_DEBUG"
    CFG.epochs = 1
    CFG.min_valid_epoch = 0



# %%
CFG.output_path = f"./../output/{CFG.exp_id}/"
os.makedirs(CFG.output_path,exist_ok=True)

# %%
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from PIL import Image
import cv2
import albumentations as A

import numpy as np
import torch
import pytorch_lightning as pl
pl.seed_everything(CFG.seed)

# %%
df = pd.read_pickle("./../input/gt_for_scatter_edge_ticks_v2.pkl")
# %%
train_df = df[~df["is_val"]].reset_index(drop=True)
valid_df = df[df["is_val"]].reset_index(drop=True)
train_df = train_df[train_df["source"] == "extracted"].reset_index(drop=True)

# %%
if CFG.debug:
    train_df = train_df.head(100)
    valid_df = valid_df.head(2)

# %%

train_aug = A.Compose([
])

valid_aug = A.Compose([
])



image_dir = './../input/benetech-making-graphs-accessible/train/images/'
MAX_PATCHES = 2048
class ImageCaptioningDataset(Dataset):
    def __init__(self, df, processor,transform):
        self.df = df
        self.processor = processor
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df["image_id"].values[idx]
        path = image_dir + image_id + ".jpg"
        image = cv2.imread(path)
        image = self.transform(image=image)["image"]

        text =self.df["text"].values[idx]
        # encoding = self.processor(images=image, text = "", return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)
        encoding = self.processor(images=image, 
                                  text = "Generate underlying data table of the figure below:",
                                  return_tensors="pt", add_special_tokens=True, max_patches=MAX_PATCHES)        
        encoding = {k:v.squeeze() for k,v in encoding.items()}
        encoding["text"] = text
        return encoding

# %%
def collator(batch):
  new_batch = {"flattened_patches":[], "attention_mask":[]}
  texts = [item["text"] for item in batch]
  
  text_inputs = processor.tokenizer(text=texts, padding="max_length", return_tensors="pt", add_special_tokens=True, max_length=128,truncation=True)
  
  new_batch["labels"] = text_inputs.input_ids
  
  for item in batch:
    new_batch["flattened_patches"].append(item["flattened_patches"])
    new_batch["attention_mask"].append(item["attention_mask"])
  
  new_batch["flattened_patches"] = torch.stack(new_batch["flattened_patches"])
  new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])

  return new_batch

# %%
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
processor = AutoProcessor.from_pretrained("google/matcha-base")
model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-base")
# processor = AutoProcessor.from_pretrained("google/deplot")
# model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
model.encoder.gradient_checkpointing_enable()
model.decoder.gradient_checkpointing_enable()

# %%
train_dataset = ImageCaptioningDataset(train_df, processor,transform=train_aug)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=collator,num_workers=CFG.num_workers,pin_memory=True)
valid_dataset = ImageCaptioningDataset(valid_df, processor,transform=valid_aug)
valid_dataloader = DataLoader(valid_dataset, shuffle=True, batch_size=2, collate_fn=collator,num_workers=CFG.num_workers,pin_memory=True)

# %%
def prepare_gt_df(df):
    gt_x = df[["image_id","xs"]].copy()
    gt_x["id"] = gt_x["image_id"] + "_x"
    gt_x = gt_x.rename(columns={"xs":"data_series"})

    gt_y = df[["image_id","ys"]].copy()
    gt_y["id"] = gt_y["image_id"] + "_y"
    gt_y = gt_y.rename(columns={"ys":"data_series"})


    gt_df = pd.concat([gt_x,gt_y])
    gt_df = gt_df.merge(df[["image_id","chart_type","source"]],on=["image_id"],how="left")
    gt_df = gt_df[["id","data_series","chart_type","source"]]
    gt_df = gt_df.set_index("id").drop(columns=['source'])
    return gt_df

gt_df = prepare_gt_df(df)


# %%
import numpy as np
from typing import List, Dict, Union
from polyleven import levenshtein # a faster version of levenshtein

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
        scores.append(score_series(gt_series, pred_series))
        # if gt_type != pred_type:  # Check chart_type condition
        #     scores.append(0.0)
        # else:  # Score with RMSE or Levenshtein as appropriate
        #     scores.append(score_series(gt_series, pred_series))

    ground_truth["score"] = scores

    grouped = ground_truth.groupby("chart_type", as_index=False)["score"].mean()

    chart_type2score = {
        chart_type: score
        for chart_type, score in zip(grouped["chart_type"], grouped["score"])
    }

    return np.mean(scores), chart_type2score,scores



# %%
def postprocess_string(pred_string):
    chart_type = "line"
    pred_string = pred_string.split("</s>")[0]
    data = pred_string.split("<0x0A>")
    data = [d for d in data if "|" in d]
    xs = [d.split("|")[0] for d in data]
    ys = [d.split("|")[1] for d in data]
    return chart_type,xs,ys

# %%


# %%
def validation_metrics(val_outputs, val_ids, gt_df):
    pred_triplets = []

    for example_output in val_outputs:
        pred_triplets.append(postprocess_string(example_output))
    
    pred_df = pd.DataFrame(
        data={
            "id":[f"{id_}_x" for id_ in val_ids] + [f"{id_}_y" for id_ in val_ids],
            "data_series": [x[1] for x in pred_triplets]
            + [x[2] for x in pred_triplets],
            "chart_type": [x[0] for x in pred_triplets] * 2,
        },
    )
    pred_df = pred_df.set_index("id")

    _, chart_type2score,scores = benetech_score(
        gt_df.loc[pred_df.index.values], pred_df
    )
    pred_df["output"] =  val_outputs * 2
    pred_df["score"] = scores
    pred_df["chart_type"] = gt_df.loc[pred_df.index.values]["chart_type"]
    pred_df["data_series_gt"] = gt_df.loc[pred_df.index.values]["data_series"]
    overall_score = pred_df[pred_df["chart_type"]!="dot"]["score"].mean() #全体スコアはdot以外で計算
    pred_df = pred_df.reset_index()
    pred_df["image_id"] = pred_df["id"].apply(lambda x:x.split("_")[0])
    pred_df["len_pred"] = pred_df["data_series"].apply(lambda x:len(x))
    pred_df["len_gt"] = pred_df["data_series_gt"].apply(lambda x:len(x))
    pred_df = pred_df[['id','image_id', 'data_series', 'data_series_gt',"len_pred","len_gt",'chart_type', 'output', 'score']]
    metrics = {
        "val_score": overall_score,
        **{f"{k}_score": v for k, v in chart_type2score.items()},
    }
    return metrics,pred_df


device = "cuda" if torch.cuda.is_available() else "cpu"
class DeplotPLModule(pl.LightningModule):
    def __init__(self, processor, model):
        super().__init__()
        self.processor = processor
        self.model = model
        self.best_score = -1

    def training_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        flattened_patches = batch.pop("flattened_patches")
        attention_mask = batch.pop("attention_mask")
        

        outputs = self.model(flattened_patches=flattened_patches,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        labels = batch.pop("labels")
        flattened_patches = batch.pop("flattened_patches")
        attention_mask = batch.pop("attention_mask")
        batch_size = labels.shape[0]

        outputs = self.model(flattened_patches=flattened_patches,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        self.log("val_loss", loss, batch_size=2)
        return loss

    # def on_validation_start(self) -> None:
    #     pass

    def on_validation_epoch_end(self):
        if self.current_epoch  >= CFG.min_valid_epoch:
            val_outputs,val_ids = [],[]

            for idx in tqdm(range(len(valid_df))):
                image_id = valid_df["image_id"].values[idx]
                path = image_dir + image_id + ".jpg"
                image = cv2.imread(path)
                image = valid_aug(image=image)["image"]
                inputs = processor(images=image, 
                                text="Generate underlying data table of the figure below:", 
                                return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                predictions = self.model.generate(**inputs, max_new_tokens=128,
                    num_beams=5,top_k=1,early_stopping=True)
                val_ids.append(image_id)
                val_outputs.append(processor.decode(predictions[0], skip_special_tokens=True))
            print(val_outputs[:3])
            metrics, pred_df = validation_metrics(val_outputs, val_ids, gt_df)
            print("\n", metrics)
            self.log_dict(metrics)

            if metrics["val_score"] > self.best_score:
                self.best_score = metrics["val_score"]
                pred_df.to_pickle(CFG.output_path + "pred_df_best.pkl")
                self.model.save_pretrained(CFG.output_path)
                self.processor.save_pretrained(CFG.output_path)

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.Adam(self.parameters(), lr=CFG.lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=len(train_dataloader)*CFG.decay_epoch, eta_min=CFG.min_lr)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# %%
def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith("__"))

# %%
trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=CFG.epochs,
        val_check_interval=CFG.val_check_interval,
        check_val_every_n_epoch=CFG.check_val_every_n_epoch,
        gradient_clip_val=CFG.gradient_clip_val,
        precision=16, # we'll use mixed precision
        num_sanity_val_steps=0,
        callbacks=[], 
)

model_module = DeplotPLModule(processor, model)
trainer.fit(model_module, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)