import torch
import pandas as pd
from torch.utils.data import Dataset

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
