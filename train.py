import os
import warnings
import pandas as pd
import numpy as np
from typing import List
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from data.GroupTimeSeriesSplit import GroupTimeSeriesSplit

warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BATCHSIZE = 30000
CLASSES = 1
EPOCHS = 15
DIR = os.getcwd()


def pearson_loss(x, y):
    xd = x - x.mean()
    yd = y - y.mean()
    nom = (xd * yd).sum()
    denom = ((xd ** 2).sum() * (yd ** 2).sum()).sqrt()
    return 1 - nom / denom


def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2 ** (n + 1 - j)))
    return np.average(a, weights=w)


class UbiquantData(Dataset):
    def __init__(self, data: pd.core.frame.DataFrame, categorical=False):
        self.target = data[['target']].values
        self.time_id = data.time_id.values
        self.data = data.drop(['index', 'row_id', 'time_id', 'investment_id', 'target'], axis=1).values
        # self.data = data.drop(['row_id', 'time_id', 'investment_id', 'target'], axis=1).values
        self.investment_ids = data.investment_id.values
        self.categorical = categorical

    def __getitem__(self, idx):
        x_cont = self.data[idx]
        target = self.target[idx]
        x_cat = self.investment_ids[idx]
        time_ids = self.time_id[idx]
        if self.categorical:
            return torch.tensor(x_cont).float(), x_cat, torch.tensor(target).float(), time_ids
        else:
            return torch.tensor(x_cont).float(), [], torch.tensor(target).float(), time_ids

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
        self.categorical = categorical
        mlp_layers: List[nn.Module] = []
        input_dim: int = 300

        if categorical:
            cat_input_dim: int = 3774
            emb_layers: List[nn.Module] = [nn.Embedding(cat_input_dim, emb_output)]
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

        if self.categorical:
            self.emb_nn: nn.Module = nn.Sequential(*emb_layers)
        self.mlp_nn: nn.Module = nn.Sequential(*mlp_layers)

    def forward(self, x_cont, x_cat):
        if self.categorical:
            x_cat = self.emb_nn(x_cat)
            concat = torch.cat([x_cat, x_cont], 1)
            output = self.mlp_nn(concat)
        else:
            output = self.mlp_nn(x_cont)
        return output


def pearson_coef(data):
    return data.corr()['targets']['preds']


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
        self.train_scores = list()
        self.valid_scores = list()

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
            x_cont, x_cat, y, time_ids = batch
            logits = self.forward(x_cont, x_cat)
        else:
            x_cont, _, y, time_ids = batch
            logits = self.forward(x_cont, [])
        scores_df = pd.DataFrame(index=range(len(y)), columns=['time_id', 'targets', 'preds'])
        scores_df.time_id, scores_df.targets, scores_df.preds = time_ids.cpu().numpy(), \
                                                                y.cpu().numpy(), \
                                                                logits.detach().cpu().numpy()
        self.train_scores.append(scores_df)
        pearson = np.mean(pd.concat([scores_df]).groupby(['time_id']).apply(pearson_coef))
        pearson = torch.from_numpy(np.array(pearson))
        p_loss = pearson_loss(logits, y)
        self.log("fold-{}/train/step/loss".format(fold), p_loss)
        self.log("fold-{}/train/step/pearson".format(fold), pearson)
        return {'loss': p_loss, 'train_corr': pearson}

    def training_epoch_end(self, outputs):
        avg_corr = np.mean(pd.concat(self.train_scores).groupby(['time_id']).apply(pearson_coef))
        print('mean pearson correlation on training set: ', avg_corr)
        self.train_scores = list()
        self.log("fold-{}/train/epoch/pearson".format(fold), avg_corr)

    def validation_step(self, batch, batch_idx):
        if self.categorical:
            x_cont, x_cat, y, time_ids = batch
            logits = self.forward(x_cont, x_cat)
        else:
            x_cont, _, y, time_ids = batch
            logits = self.forward(x_cont, [])
        scores_df = pd.DataFrame(index=range(len(y)), columns=['time_id', 'targets', 'preds'])
        scores_df.time_id, scores_df.targets, scores_df.preds = time_ids.cpu().numpy(), \
                                                                y.cpu().numpy(), \
                                                                logits.cpu().numpy()
        self.valid_scores.append(scores_df)
        pearson = np.mean(pd.concat([scores_df]).groupby(['time_id']).apply(pearson_coef))
        pearson = torch.from_numpy(np.array(pearson))
        self.log("fold-{}/valid/step/pearson".format(fold), pearson)
        return {'val_corr': pearson}

    def validation_epoch_end(self, outputs):
        if self.current_epoch == 9:
            oof.append(pd.concat(self.valid_scores))
        avg_corr = np.mean(pd.concat(self.valid_scores).groupby(['time_id']).apply(pearson_coef))
        print('mean pearson correlation on validation set: ', avg_corr)
        self.log("fold-{}/valid/epoch/pearson".format(fold), avg_corr)
        self.valid_scores = list()
        return {'avg_val_corr': avg_corr}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.l_rate)


if __name__ == '__main__':
    neptune_logger = NeptuneLogger(
        project="kaggle.collaboration/ubiquant",
        tags=["5-fold", "training"],
    )

    pl.utilities.seed.seed_everything(seed=2022)

    df = pd.read_csv('input/train.csv')
    df = df[(df.time_id <= 355) | (df.time_id >= 420)].reset_index()
    print('data loaded...')

    oof = list()

    gtss = GroupTimeSeriesSplit(n_folds=1, holdout_size=100, groups=df['time_id'])

    for fold, (train_indexes, val_indexes) in enumerate(gtss.split(df)):

        scores_valid, scores_train = list(), list()

        train_data = df.iloc[train_indexes].sort_values(by=['time_id'])
        val_data = df.iloc[val_indexes].sort_values(by=['time_id'])

        checkpoint_callback = ModelCheckpoint(
            monitor="fold-{}/valid/epoch/pearson".format(fold),
            dirpath="models",
            filename="fold-" + str(fold) + "-ubiquant-mlp-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="max",
        )

        model = UbiquantModel(dropout_mlp=0.4217199217221381,
                              dropout_emb=0.4250209544891712,
                              output_dims=[508, 405],
                              emb_dims=[245, 238, 230],
                              emb_output=56,
                              l_rate=0.00026840511349794486,
                              categorical=True)

        print(model)

        trainer = Trainer(max_epochs=EPOCHS,
                          logger=neptune_logger,
                          fast_dev_run=False,
                          callbacks=[checkpoint_callback],
                          gpus=1)
        trainer.fit(model)

oof_pearson_coef = pd.concat(oof).groupby(['time_id']).apply(pearson_coef)
neptune_logger.experiment['pearson_timeid'].log(oof_pearson_coef.values.tolist())
print('oof score pearson coef per time id: ', np.mean(oof_pearson_coef))
