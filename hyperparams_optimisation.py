import optuna
import argparse
import pandas as pd
import numpy as np
from typing import List
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from data.GroupTimeSeriesSplit import GroupTimeSeriesSplit
from optuna.integration import PyTorchLightningPruningCallback
import neptune.new.integrations.optuna as optuna_utils

import neptune.new as neptune

import warnings

warnings.filterwarnings("ignore")

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BATCHSIZE = 8192
CLASSES = 1
EPOCHS = 15
DIR = os.getcwd()

def pearson_loss(x, y):
    xd = x - x.mean()
    yd = y - y.mean()
    nom = (xd * yd).sum()
    denom = ((xd ** 2).sum() * (yd ** 2).sum()).sqrt()
    return 1 - nom / denom

class UbiquantData(Dataset):
    def __init__(self, data: pd.core.frame.DataFrame, categorical=False):
        self.target = data[['target']].values
        self.data = data.drop(['index', 'row_id', 'time_id', 'investment_id', 'target'], axis=1).values
        # self.data = data.drop(['row_id', 'time_id', 'investment_id', 'target'], axis=1).values
        self.investment_ids = data.investment_id.values
        self.categorical = categorical

    def __getitem__(self, idx):
        x_cont = self.data[idx]
        target = self.target[idx]
        x_cat = self.investment_ids[idx]
        if self.categorical:
            return torch.tensor(x_cont).float(), x_cat, torch.tensor(target).float()
        else:
            return torch.tensor(x_cont).float(), [], torch.tensor(target).float()

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
            logits = self.forward(x_cont, x_cat)
        else:
            x_cont, _, y = batch
            logits = self.forward(x_cont, [])
        loss = pearson_loss(logits, y)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        if self.categorical:
            x_cont, x_cat, y = batch
            logits = self.forward(x_cont, x_cat)
        else:
            x_cont, _, y = batch
            logits = self.forward(x_cont, [])
        scores_df = pd.DataFrame(index=range(len(y)), columns=['targets', 'preds'])
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

def objective(trial: optuna.trial.Trial) -> float:
    n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 5)
    n_emb_layers = trial.suggest_int("n_emb_layers", 1, 5)
    dropout_mlp = trial.suggest_float("dropout_mlp", 0.1, 0.5)
    dropout_emb = trial.suggest_float("dropout_emb", 0.1, 0.5)
    output_dims = [
        trial.suggest_int("n_units_mlp{}".format(i), 32, 512, log=True) for i in range(n_mlp_layers)
    ]
    emb_dims = [
        trial.suggest_int("n_units_emb{}".format(i), 32, 512, log=True) for i in range(n_emb_layers)
    ]

    emb_output = trial.suggest_int("emb_output", 5, 100)

    l_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

    model = UbiquantModel(dropout_mlp=dropout_mlp,
                          dropout_emb=dropout_emb,
                          output_dims=output_dims,
                          emb_dims=emb_dims,
                          emb_output=emb_output,
                          l_rate=l_rate,
                          categorical=False)

    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        gpus=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="pearson")],
    )
    hyperparameters = dict(n_mlp_layers=n_mlp_layers,
                           n_emb_layers=n_emb_layers,
                           dropout_mlp=dropout_mlp,
                           dropout_emb=dropout_emb,
                           output_dims=output_dims,
                           emb_dims=emb_dims,
                           emb_output=emb_output,
                           l_rate=l_rate)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model)

    return trainer.callback_metrics["pearson"].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ubiquant hyperparameters optimisation.")
    parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
             "trials at the early stages of training.",
    )
    args = parser.parse_args()

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    run = neptune.init(
        project="kaggle.collaboration/ubiquant"
    )

    df = pd.read_csv("input/train.csv")
    df = df[(df.time_id <= 355) | (df.time_id >= 420)].reset_index()

    gtss = GroupTimeSeriesSplit(n_folds=5, holdout_size=150, groups=df['time_id'])
    neptune_callback = optuna_utils.NeptuneCallback(run)

    for fold, (train_indexes, val_indexes) in enumerate(gtss.split(df)):
        if fold == 4:
            train_data = df.iloc[train_indexes].sort_values(by=['time_id'])
            val_data = df.iloc[val_indexes].sort_values(by=['time_id'])

            study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler())
            study.optimize(objective, n_trials=20, callbacks=[neptune_callback])

            print("Number of finished trials: {}".format(len(study.trials)))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: {}".format(trial.value))

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))

            break
        else:
            pass
