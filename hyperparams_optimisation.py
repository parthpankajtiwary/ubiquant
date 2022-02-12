import numpy as np

import argparse
import os
from typing import List

import pandas as pd

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

from data.GroupTimeSeriesSplit import GroupTimeSeriesSplit

BATCHSIZE = 8192
CLASSES = 1
EPOCHS = 6
DIR = os.getcwd()

def pearson_loss(x, y):
    xd = x - x.mean()
    yd = y - y.mean()
    nom = (xd*yd).sum()
    denom = ((xd**2).sum() * (yd**2).sum()).sqrt()
    return 1 - nom / denom

bce_with_logits_loss  = nn.BCEWithLogitsLoss()#reduction = 'mean'


def swish(x):
    return x * torch.sigmoid(x)

class UbiquantData(Dataset):
    def __init__(self, data: pd.core.frame.DataFrame, categorical=False):
        self.target = data[['target']].values.astype(float)
        self.target_labels = data['target'].apply(lambda x: 1 if x>0 else 0).values.astype(float)
        self.data = data.drop(['index', 'row_id', 'time_id', 'investment_id', 'target'], axis=1).values.astype(float)
        # self.data = data.drop(['row_id', 'time_id', 'investment_id', 'target'], axis=1).values
        self.investment_ids = data.investment_id.values
        self.categorical = categorical

    def __getitem__(self, idx):
        x_cont = self.data[idx]
        target = self.target[idx]
        target_label = self.target_labels[idx]
        x_cat = self.investment_ids[idx]
        if self.categorical:
            return torch.tensor(x_cont).float(), x_cat, torch.tensor(target).float(),torch.tensor(target_label).float()
        else:
            return torch.tensor(x_cont).float(), [], torch.tensor(target).float(),torch.tensor(target_label).float()

    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self,
                 dropout_mlp: float,
                 dropout_emb: float,
                 output_dims: List[int],
                 cat_dims: List[int],
                 emb_output: int,
                 dropout_mlp_cls: float,
                 output_dims_cls: List[int],
                 with_classification = False,
                 categorical=False):

        super().__init__()
        self.categorical = categorical
        self.with_classification = with_classification
        mlp_layers: List[nn.Module] = []
        mlp_layers_cls: List[nn.Module] = []
        mlp_layers_reg: List[nn.Module] = []
        input_dim: int = 300
#         input_dim_cls: int = 300

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
            input_dim_cls+=cat_output_dim

        for output_dim in output_dims:
            mlp_layers.append(nn.Linear(input_dim, output_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout_mlp))
            input_dim = output_dim

        
        mlp_layers_reg.append(nn.Linear(input_dim, 1))
        
        if with_classification:
            input_dim_cls = input_dim
            for output_dim in output_dims_cls:
                mlp_layers_cls.append(nn.Linear(input_dim_cls, output_dim))
                mlp_layers_cls.append(nn.ReLU())
                mlp_layers_cls.append(nn.Dropout(dropout_mlp_cls))
                input_dim_cls = output_dim
            
            mlp_layers_cls.append(nn.Linear(input_dim_cls, 1))
            

        if self.categorical:
            self.emb_nn: nn.Module = nn.Sequential(*emb_layers)
        self.base_nn: nn.Module = nn.Sequential(*mlp_layers)
        self.reg_nn: nn.Module = nn.Sequential(*mlp_layers_reg)
        if with_classification:
            self.mlp_cls_nn: nn.Module = nn.Sequential(*mlp_layers_cls)
            

    def forward(self, x_cont, x_cat):
        if self.categorical:
            x_cat = self.emb_nn(x_cat)
            x_cont = torch.cat([x_cat, x_cont], 1) 
        
        output = self.base_nn(x_cont)
        output1 = self.reg_nn(output)
        output2 = None
        if self.with_classification:
            output2 = self.mlp_cls_nn(output)
        
        return output1,output2

class UbiquantModel(pl.LightningModule):
    def __init__(self,
                 dropout_mlp: float,
                 dropout_emb: float,
                 output_dims: List[int],
                 emb_dims: List[int],
                 emb_output: int,
                 l_rate: float,
                 dropout_mlp_cls: float,
                 output_dims_cls: List[int],
                 with_classification:bool,
                 categorical: bool):
        super().__init__()
        self.model = Net(dropout_mlp,
                         dropout_emb,
                         output_dims,
                         emb_dims,
                         emb_output,
                         dropout_mlp_cls,
                         output_dims_cls,
                         with_classification = with_classification,
                         categorical=categorical)
        self.l_rate = l_rate
        self.categorical = categorical
        self.with_classification = with_classification

    def forward(self, x_cont, x_cat):
        return self.model(x_cont, x_cat)

    def train_dataloader(self):
        train_dataset = UbiquantData(train_data, categorical=self.categorical)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHSIZE)
        return train_loader

    def val_dataloader(self):
        val_dataset = UbiquantData(val_data, categorical=self.categorical)
        val_loader = DataLoader(dataset=val_dataset, batch_size=BATCHSIZE)
        return val_loader

    def training_step(self, batch, batch_idx):
        logs = {}
        if self.categorical:
            x_cont, x_cat, y, y_label = batch
            logits,logits_cls = self.forward(x_cont, x_cat)
        else:
            x_cont, _, y, y_label = batch
            logits,logits_cls = self.forward(x_cont, [])
        
        loss_p = pearson_loss(logits, y)
#         print(loss_p)
        logs['loss_p'] = loss_p
        if self.with_classification:
            loss_cls = bce_with_logits_loss(logits_cls,y_label.unsqueeze(dim=1))
#             print(loss_cls)
            
            loss = loss_p+loss_cls
            
            logs['loss_cls'] = loss_cls
        
        else:
            loss = loss_p
        
        logs['loss'] = loss
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        if self.categorical:
            x_cont, x_cat, y, y_label = batch
            logits,logits_cls = self.forward(x_cont, x_cat)
        else:
            x_cont, _, y, y_label = batch
            logits,logits_cls = self.forward(x_cont, [])
        
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
        if fold not in scores:
            scores[fold] = [avg_loss]
        else:
            scores[fold].append(avg_loss)
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.l_rate)


# +
def objective(trial: optuna.trial.Trial) -> float:
    n_mlp_layers = trial.suggest_int("n_mlp_layers", 1, 5)
#     n_emb_layers = trial.suggest_int("n_emb_layers", 1, 5)
    n_cls_layers = trial.suggest_int("n_cls_layers", 1, 5)
    dropout_mlp = trial.suggest_float("dropout_mlp", 0.1, 0.5)
#     dropout_emb = trial.suggest_float("dropout_emb", 0.1, 0.5)
    dropout_mlp_cls = trial.suggest_float("dropout_mlp_cls", 0.1, 0.5)
    
    output_dims = [
        trial.suggest_int("n_units_mlp{}".format(i), 32, 512, log=True) for i in range(n_mlp_layers)
    ]
#     emb_dims = [
#         trial.suggest_int("n_units_emb{}".format(i), 32, 512, log=True) for i in range(n_emb_layers)
#     ]

#     emb_output = trial.suggest_int("emb_output", 5, 100)

    output_dims_cls = [
        trial.suggest_int("n_units_mlp_cls{}".format(i), 32, 512, log=True) for i in range(n_cls_layers)
    ]

    l_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)

    model = UbiquantModel(dropout_mlp=dropout_mlp,
                          dropout_emb=0,
                          output_dims=output_dims,
                          emb_dims=[],
                          emb_output=1,
                          l_rate=l_rate,
                          dropout_mlp_cls = dropout_mlp_cls,
                          output_dims_cls = output_dims_cls,
                          with_classification = True,
                          categorical=False)

    print(model)

    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=EPOCHS,
        gpus=1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="pearson")],
    )
    hyperparameters = dict(n_mlp_layers=n_mlp_layers,
#                            n_emb_layers=n_emb_layers,
                           dropout_mlp=dropout_mlp,
#                            dropout_emb=dropout_emb,
                           output_dims=output_dims,
#                            emb_dims=emb_dims,
#                            emb_output=emb_output,
                           dropout_mlp_cls = dropout_mlp_cls,
                           output_dims_cls = output_dims_cls,
                           l_rate=l_rate)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model)

    return trainer.callback_metrics["pearson"].item()


# -

BASE_DIR = '/sharedHDD/rohit/timeseries_learning/ubiquant/'
DATA_DIR = BASE_DIR+'data/parquet/'
INPUT_DIR = BASE_DIR+'input/'
WEIGHTS_DIR = BASE_DIR + 'weights/'

# +
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

    scores = dict()
    
    df = pd.read_parquet(DATA_DIR+'train_low_mem.parquet')

    df = df[(df.time_id <= 355) | (df.time_id >= 420)].reset_index()
    
    print('data loaded...')

    gtss = GroupTimeSeriesSplit(n_folds=5, holdout_size=150, groups=df['time_id'])
    
    for fold, (train_indexes, val_indexes) in enumerate(gtss.split(df)):
        print(len(train_indexes), len(val_indexes))

        train_data = df.iloc[train_indexes].sort_values(by=['time_id'])
        val_data = df.iloc[val_indexes].sort_values(by=['time_id'])
        
        break

#     train_data = pd.read_csv('input/fold_0_train.csv')
#     val_data = pd.read_csv('input/fold_0_val.csv')

    study = optuna.create_study(direction="maximize", pruner=pruner, sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100, timeout=86400)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
