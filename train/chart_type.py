import os
class CFG:
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    debug = False
    num_workers = 12
    gpus = 1 # doesn't work yet with 2 gpus

    # Training
    epochs = 20
    lr = 1e-3
    seed = 42
    batch_size = 16
    model_name = 'efficientnet_b0'

if CFG.debug:
    CFG.exp_id += "_DEBUG"
    CFG.epochs = 1
    
CFG.output_path = f"./../output/{CFG.exp_id}/"
os.makedirs(CFG.output_path,exist_ok=True)
# -

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
import timm
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

# +

# %%
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# %%
label_map = {'dot': 0, 'horizontal_bar' : 1, 'vertical_bar': 2, 'line': 3, 'scatter': 4}
num_classes = 5


# %%
image_dir = './../input/benetech-making-graphs-accessible/train/images/'


# +

# %%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # img_path = image_dir + self.df["image_id"].values[index] + ".jpg"
        img_path = self.df["image_path"].values[index]
        target = self.df["label"].values[index]
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        target = torch.tensor(target) 
        return img, target


# -

df = pd.read_csv("./../input/benetech-split/benetech_split.csv")

df["chart_type"] = df["type"]
df["image_path"] = df["id"].apply(lambda x: f"./../input/benetech-making-graphs-accessible/train/images/{x}.jpg")

ext_df = df[df["source"]=="extracted"].reset_index(drop=True)
gen_df = df[df["source"]=="generated"].reset_index(drop=True)

icdar_df = pd.read_csv("./../input/processed_df_icdar_v5.csv")

syn_ver_df = pd.read_csv("./../input/benetech-synthesis-4th/graph_synthesis_vertical_bar_40000_ver11.csv")
syn_line_ex_df = pd.read_csv("./../input/benetech-synthesis-4th/graph_synthesis_line_40000_ver11.csv")
syn_hol_df = pd.read_csv("./../input/benetech-synthesis-4th/graph_synthesis_horizontal_bar_40000_ver12.csv")
syn_df = pd.concat([syn_line_ex_df, syn_hol_df, syn_ver_df]).reset_index(drop=True)
syn_df["image_path"] = "./." + syn_df["image_path"] # fix path
syn_df["chart_type"] = syn_df["chart-type"]

icdar_df = icdar_df[icdar_df["series_len"]==1].reset_index(drop=True)
valid_df = icdar_df[icdar_df["source"] == "icdar2022_test"].reset_index(drop=True)
icdar_df = icdar_df[icdar_df["source"] == "icdar2022_train"].reset_index(drop=True)

# +
ext_df = df[df["source"]=="extracted"].reset_index(drop=True)
gen_df = df[df["source"]=="generated"].reset_index(drop=True)
dot_df = gen_df[gen_df["chart_type"]=="dot"].sample(1000)
hbar_df = pd.concat([
    ext_df[ext_df["chart_type"]=="horizontal_bar"],
    icdar_df[icdar_df["chart_type"]=="horizontal_bar"],
    syn_df[syn_df["chart_type"]=="horizontal_bar"].sample(724),
                    ])
vbar_df = pd.concat([
    ext_df[ext_df["chart_type"]=="vertical_bar"],
    icdar_df[icdar_df["chart_type"]=="vertical_bar"],
    gen_df[gen_df["chart_type"]=="vertical_bar"].sample(40),
    syn_df[syn_df["chart_type"]=="vertical_bar"].sample(25),
                    ])
scatter_df = pd.concat([
    ext_df[ext_df["chart_type"]=="scatter"],
    gen_df[gen_df["chart_type"]=="scatter"].sample(835),
                    ])
line_df = pd.concat([
    ext_df[ext_df["chart_type"]=="line"],
    icdar_df[icdar_df["chart_type"]=="line"],
    gen_df[gen_df["chart_type"]=="line"].sample(150),
    syn_df[syn_df["chart_type"]=="line"].sample(157),
                    ])

train_df = pd.concat([dot_df,hbar_df,vbar_df,scatter_df,line_df],ignore_index=True)

# -

train_df["label"] = train_df["chart_type"].map(label_map)
valid_df["label"] = valid_df["chart_type"].map(label_map)
print(train_df["chart_type"].value_counts())
print(valid_df["chart_type"].value_counts())


# +

train_df = train_df[["image_id","label","source","image_path"]]
valid_df = valid_df[["image_id","label","source","image_path"]]
data = MyDataset(train_df)
plt.imshow(data[0][0].permute((1,2,0)))

# %%
if CFG.debug:
    train_df = train_df.sample(500).reset_index(drop=True)

# %%
train_dataset = MyDataset(train_df)
val_dataset = MyDataset(valid_df)
train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True, num_workers=CFG.num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=CFG.num_workers)

# %%
class MyEstimation(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(CFG.model_name, pretrained=True, num_classes=5)
        self.criterion = nn.CrossEntropyLoss()
        self.val_pred,self.val_gt = [],[]
        self.best_score = -1

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x) #.squeeze()
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, on_epoch=True)
        self.val_pred += [torch.argmax(y_hat, 1).detach().cpu().numpy()]
        self.val_gt += [y.detach().cpu().numpy()]

    def on_validation_epoch_end(self):
        val_pred = np.concatenate(self.val_pred)
        label = np.concatenate(self.val_gt)
        score = (val_pred == label).mean()
        self.val_pred,self.val_gt = [],[]
        self.log('val_acc', score, on_epoch=True)
        print(f'ACC: {score:.4f}')
        
        if score > self.best_score:
            self.best_score = score
            torch.save(self.model.state_dict(), f'{CFG.output_path}{CFG.model_name}_chart_type.pth')
            pred_df = valid_df.copy()
            pred_df["pred"] = val_pred
            pred_df["label"] = label
            pred_df.to_csv(f'{CFG.output_path}pred_df.csv',index=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=CFG.lr)
        return optimizer

# %%
def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith("__"))

# %%
model = MyEstimation()
trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=CFG.epochs,
        precision=16, # we'll use mixed precision
        num_sanity_val_steps=0,
        callbacks=[], 
)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
# -


