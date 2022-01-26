import pickle
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

epochs = 5
l_rate = 1e-3
mse_loss = nn.MSELoss()

def swish(x):
    return x * torch.sigmoid(x)

def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2**(n + 1 - j)))
    return np.average(a, weights = w)

class UbiquantData(Dataset):
    def __init__(self, data: pd.core.frame.DataFrame):
        self.target = data[['target']].values
        self.data = data.drop(['row_id', 'time_id', 'investment_id', 'target'], axis=1).values
        self.investment_ids = data.investment_id.values

    def __getitem__(self, idx):
        x_cont = self.data[idx]
        target = self.target[idx]
        x_cat = self.investment_ids[idx]
        return torch.tensor(x_cont).float(), x_cat, torch.tensor(target).float()

    def __len__(self):
        return len(self.data)

class UbiquantModel(pl.LightningModule):
    def __init__(self):
        super(UbiquantModel, self).__init__()
        self.emb = nn.Embedding(3774, 64)
        self.emb_drop = nn.Dropout(0.1)
        self.bn1 = nn.BatchNorm1d(300)
        self.fc1 = nn.Linear(64+300, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x_cont, x_cat):
        x1 = self.emb(x_cat)
        x1 = self.emb_drop(x1)

        x = torch.cat([x1, x_cont], 1)

        x = swish(self.fc1(x))
        x = swish(self.fc2(x))
        x = swish(self.fc3(x))
        x = swish(self.fc4(x))
        x = self.fc5(x)
        return x

    def train_dataloader(self):
        train_dataset = UbiquantData(train_data)
        train_loader = DataLoader(dataset=train_dataset, batch_size=4096)
        return train_loader

    def val_dataloader(self):
        val_dataset = UbiquantData(val_data)
        val_loader = DataLoader(dataset=val_dataset, batch_size=4096)
        return val_loader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=l_rate)

    def training_step(self, batch, batch_idx):
        x_cont, x_cat, y = batch
        logits = self.forward(x_cont, x_cat)
        loss = mse_loss(logits, y)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x_cont, x_cat, y = batch
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
        if fold not in scores:
            scores[fold] = [avg_loss]
        else:
            scores[fold].append(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

if __name__ == '__main__':
    scores = dict()
    df = pd.read_csv('input/train.csv')

    print('data loaded...')

    with open('input/folds.pickle', 'rb') as handle:
        fold_indexes = pickle.load(handle)

    for fold in fold_indexes:
        remove_fields = ['target', 'row_id', 'time_id', 'investment_id']
        target_fields = ['target']

        train_data = df.iloc[fold_indexes[fold]['train']]
        val_data = df.iloc[fold_indexes[fold]['test']]

        checkpoint_callback = ModelCheckpoint(
            monitor="pearson",
            dirpath="models",
            filename="fold-" + str(fold) + "-ubiquant-mlp-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="max",
        )

        model = UbiquantModel()
        trainer = Trainer(max_epochs=epochs,
                          fast_dev_run=False,
                          callbacks=[checkpoint_callback],
                          gpus=1)
        trainer.fit(model)

    score_list = list()
    for fold in scores:
        score_list.append(max(scores[fold]))

    print('final weighted correlation for the experiment: ', weighted_average(score_list))