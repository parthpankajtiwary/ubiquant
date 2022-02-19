import os
import warnings
from typing import List
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from data.GroupTimeSeriesSplit import GroupTimeSeriesSplit

from data.data import UbiquantData
from models.model_mlp import MLP
from utils.utils import *

warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BATCHSIZE = 30000
CLASSES = 1
EPOCHS = 25
DIR = os.getcwd()


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
        self.model = MLP(dropout_mlp,
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
        scores_df = create_scores_df(time_ids.detach().cpu().numpy(),
                                     y.cpu().detach().numpy(),
                                     logits.detach().cpu().numpy())
        self.train_scores.append(scores_df)
        pearson = calc_pearson(scores_df)
        p_loss = pearson_loss(logits, y) + mse_loss(logits, y)
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
        scores_df = create_scores_df(time_ids.cpu().numpy(),
                                     y.cpu().numpy(),
                                     logits.cpu().numpy())
        self.valid_scores.append(scores_df)
        pearson = calc_pearson(scores_df)
        self.log("fold-{}/valid/step/pearson".format(fold), pearson)
        return {'val_corr': pearson}

    def validation_epoch_end(self, outputs):
        if self.current_epoch == EPOCHS - 1:
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
        tags=["1-fold", "combined loss", "embedding"],
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
            dirpath="checkpoints",
            filename="fold-" + str(fold) + "-ubiquant-mlp-{epoch:02d}-{val_loss:.2f}",
            save_top_k=15,
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
coef = np.mean(oof_pearson_coef)
neptune_logger.experiment['pearson_coeff'].log(coef)
print('oof score pearson coef per time id: ', np.mean(oof_pearson_coef))
