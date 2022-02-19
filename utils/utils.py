import torch
import numpy as np
import pandas as pd


def pearson_loss(x, y):
    xd = x - x.mean()
    yd = y - y.mean()
    nom = (xd * yd).sum()
    denom = ((xd ** 2).sum() * (yd ** 2).sum()).sqrt()
    return 1 - nom / denom


def mse_loss(x, y):
    diff_squared = ((x - y) ** 2)
    return diff_squared.mean()


def weighted_average(a):
    w = []
    n = len(a)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2 ** (n + 1 - j)))
    return np.average(a, weights=w)


def pearson_coef(data):
    return data.corr()['targets']['preds']


def calc_pearson(df_list):
    pearson = np.mean(pd.concat([df_list]).groupby(['time_id']).apply(pearson_coef))
    pearson = torch.from_numpy(np.array(pearson))
    return pearson


def create_scores_df(time_ids, y, logits):
    scores_df = pd.DataFrame(index=range(len(y)),
                             columns=['time_id', 'targets', 'preds'])
    scores_df.time_id, scores_df.targets, scores_df.preds = time_ids, y, logits
    return scores_df
