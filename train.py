import pickle
import pandas as pd
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import warnings
warnings.filterwarnings("ignore")

epochs = 10
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

class UbiquantModel(pl.LightningModule):
    def __init__(self):
        super(UbiquantModel, self).__init__()
        self.fc1 = nn.Linear(300, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)

    def forward(self, x):
        x = swish(self.fc1(x))
        x = swish(self.fc2(x))
        x = swish(self.fc3(x))
        x = swish(self.fc4(x))
        x = self.fc5(x)
        return x

    def train_dataloader(self):
        train_dataset = TensorDataset(torch.tensor(train_features.values).float(),
                                      torch.tensor(train_targets[['target']].values).float())
        train_loader = DataLoader(dataset=train_dataset, batch_size=4096)
        return train_loader

    def val_dataloader(self):
        validation_dataset = TensorDataset(torch.tensor(validation_features.values).float(),
                                           torch.tensor(validation_targets[['target']].values).float())
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=4096)
        return validation_loader

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=l_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = mse_loss(logits, y)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores_df = pd.DataFrame(index=range(len(y)), columns=['targets', 'preds'])
        logits = self.forward(x)
        scores_df.targets = y.numpy()
        scores_df.preds = logits.numpy()
        pearson = scores_df['targets'].corr(scores_df['preds'])
        pearson = np.array(pearson)
        pearson = torch.from_numpy(pearson)
        return {'val_loss': pearson}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print('mean pearson correlation on validation set: ', avg_loss)
        if fold not in scores and avg_loss > 0.05:
            scores[fold] = avg_loss
        elif avg_loss > 0.05:
            scores[fold] += avg_loss
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

if __name__ == '__main__':
    scores = dict()
    df = pd.read_csv('input/train.csv')

    with open('input/folds.pickle', 'rb') as handle:
        fold_indexes = pickle.load(handle)

    for fold in fold_indexes:
        remove_fields = ['target', 'row_id', 'time_id', 'investment_id']
        target_fields = ['target']

        train_data = df.iloc[fold_indexes[fold]['train']]
        validation_data = df.iloc[fold_indexes[fold]['test']]

        train_features, train_targets = train_data.drop(remove_fields, axis=1), train_data[target_fields]
        validation_features, validation_targets = validation_data.drop(remove_fields, axis=1), validation_data[
            target_fields]

        model = UbiquantModel()
        trainer = Trainer(max_epochs=epochs, fast_dev_run=False)
        trainer.fit(model)

    score_list = list()
    for fold in scores:
        score_list.append(scores[fold]/epochs)

    print('final weighted correlation for the experiment: ', weighted_average(score_list))