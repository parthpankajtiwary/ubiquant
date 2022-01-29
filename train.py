import pickle
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import List

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

pl.utilities.seed.seed_everything(seed=2022)

BATCHSIZE = 8192
CLASSES = 1
EPOCHS = 15
DIR = os.getcwd()

def pearson_loss(x, y):
    xd = x - x.mean()
    yd = y - y.mean()
    nom = (xd*yd).sum()
    denom = ((xd**2).sum() * (yd**2).sum()).sqrt()
    return 1 - nom / denom

def swish(x):
    return x * torch.sigmoid(x)

def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2**(n + 1 - j)))
    return np.average(a, weights=w)

class UbiquantData(Dataset):
    def __init__(self, data: pd.core.frame.DataFrame, categorical=False):
        self.target = data[['target']].values
        self.data = data.drop(['row_id', 'time_id', 'investment_id', 'target'], axis=1).values
        self.investment_ids = data.investment_id.values
        self.categorical = categorical

    def __getitem__(self, idx):
        x_cont = self.data[idx]
        target = self.target[idx]
        x_cat = self.investment_ids[idx]
        if self.categorical:
            return torch.tensor(x_cont).float(), x_cat, torch.tensor(target).float()
        else:
            return torch.tensor(x_cont).float(), torch.tensor(target).float()

    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self,
                 dropout_mlp: float,
                 dropout_emb: float,
                 output_dims: List[int],
                 cat_dims: List[int],
                 emb_output: int,
                 categorical=False):

        super().__init__()
        mlp_layers: List[nn.Module] = []
        emb_layers: List[nn.Module] = []
        self.categorical = categorical
        input_dim: int = 300
        cat_input_dim: int = 3774

        emb_layers.append(nn.Embedding(cat_input_dim, emb_output))

        if categorical:
                cat_input_dim = emb_output
                for cat_output_dim in cat_dims:
                    emb_layers.append(nn.Linear(cat_input_dim, cat_output_dim))
                    emb_layers.append(nn.ReLU())
                    emb_layers.append(nn.Dropout(dropout_emb))
                    cat_input_dim = cat_output_dim

                input_dim += cat_output_dim

                for output_dim in output_dims:
                    mlp_layers.append(nn.Linear(input_dim, output_dim))
                    mlp_layers.append(nn.ReLU())
                    mlp_layers.append(nn.Dropout(dropout_mlp))
                    input_dim = output_dim
        else:
            for output_dim in output_dims:
                mlp_layers.append(nn.Linear(input_dim, output_dim))
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout_mlp))
                input_dim = output_dim

        mlp_layers.append(nn.Linear(input_dim, 1))

        # self.layers: nn.Module = nn.Sequential(*layers)
        self.emb_nn: nn.Module = nn.Sequential(*emb_layers)
        self.mlp_nn: nn.Module = nn.Sequential(*mlp_layers)

    def forward(self, x_cont, x_cat):
        x_cat = self.emb_nn(x_cat)
        concat = torch.cat([x_cat, x_cont], 1)
        output = self.mlp_nn(concat)
        return output

class UbiquantModel(pl.LightningModule):
    def __init__(self,
                 dropout_mlp: float,
                 dropout_emb: float,
                 output_dims: List[int],
                 emb_dims: List[int],
                 emb_output: int,
                 l_rate: float,
                 categorical: bool):
        super().__init__()
        self.model = Net(dropout_mlp,
                         dropout_emb,
                         output_dims,
                         emb_dims,
                         emb_output,
                         categorical=categorical)
        self.l_rate = l_rate
        self.categorical = categorical

    def forward(self, x_cont, x_cat):
        return self.model(x_cont, x_cat)

    def train_dataloader(self):
        train_dataset = UbiquantData(train_data, categorical=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE)
        return train_loader

    def val_dataloader(self):
        val_dataset = UbiquantData(val_data, categorical=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCHSIZE)
        return val_loader

    def training_step(self, batch, batch_idx):
        if self.categorical:
            x_cont, x_cat, y = batch
        else:
            x_cont, y = batch
        logits = self.forward(x_cont, x_cat)
        loss = pearson_loss(logits, y)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        if self.categorical:
            x_cont, x_cat, y = batch
        else:
            x_cont, y = batch
        scores_df = pd.DataFrame(index=range(len(y)), columns=['targets', 'preds'])
        logits = self.forward(x_cont, x_cat)
        scores_df.targets = y.cpu().numpy()
        scores_df.preds = logits.cpu().numpy()
        pearson = scores_df['targets'].corr(scores_df['preds'])
        pearson = np.array(pearson)
        pearson = torch.from_numpy(pearson)
        self.log('pearson', pearson)
        return {'val_loss': pearson}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('mean pearson correlation on validation set: ', avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.l_rate)

if __name__ == '__main__':
    scores = dict()
    df = pd.read_csv('input/train.csv')

    print('data loaded...')

    with open('input/folds.pickle', 'rb') as handle:
        fold_indexes = pickle.load(handle)

    fold = 0

    remove_fields = ['target', 'row_id', 'time_id', 'investment_id']
    target_fields = ['target']

    train_data = df.iloc[fold_indexes[fold]['train']].sort_values(by=['time_id'])
    val_data = df.iloc[fold_indexes[fold]['test']].sort_values(by=['time_id'])

    checkpoint_callback = ModelCheckpoint(
        monitor="pearson",
        dirpath="models",
        filename="fold-" + str(fold) + "-ubiquant-mlp-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="max",
    )

    model = UbiquantModel(dropout_mlp=0.4217199217221381,
                          dropout_emb=0.4250209544891712,
                          output_dims=[508, 405],
                          emb_dims=[245, 238, 230],
                          emb_output=56,
                          l_rate=0.00026840511349794486,
                          categorical=True)

    trainer = Trainer(max_epochs=EPOCHS,
                      fast_dev_run=False,
                      callbacks=[checkpoint_callback],
                      gpus=1)
    trainer.fit(model)

    score_list = list()
    for fold in scores:
        score_list.append(max(scores[fold]))

    print('final weighted correlation for the experiment: ', weighted_average(score_list))