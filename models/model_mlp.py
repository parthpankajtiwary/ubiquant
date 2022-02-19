import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
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
