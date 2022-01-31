import pickle
import pandas as pd
from data.grouptimeseriessplit import \
    GroupTimeSeriesSplit

# read the dataframe
DATA_DIR = 'data/parquet/'
df = pd.read_parquet(DATA_DIR+'train_low_mem.parquet')

investment_ids = set(df.investment_id)

fold_indexes = dict()

for investment_id in investment_ids:
    print('generating fold indexes for investment id: ', investment_id)
    x_train = df[df.investment_id == investment_id]
    x_train = x_train.set_index('time_id')
    # filter for investment id's with not enough time steps
    if x_train.shape[0] > 5:
        fold = 0
        for train_idx, test_idx in GroupTimeSeriesSplit().split(x_train, groups=x_train.index):
            # print(train_idx,test_idx)
            train_indexes = df[(df.investment_id == investment_id) &
                               (df.time_id.isin(x_train.iloc[train_idx].index))].index
            test_indexes = df[(df.investment_id == investment_id) &
                              (df.time_id.isin(x_train.iloc[test_idx].index))].index
            if fold not in fold_indexes:
                fold_indexes[fold] = {'train': list(train_indexes),
                                      'test': list(test_indexes)}
            else:
                fold_indexes[fold]['train'].extend(list(train_indexes))
                fold_indexes[fold]['test'].extend(list(test_indexes))
            fold += 1
    else:
        pass
    
    # break
    

# with open('input/folds.pickle', 'wb') as handle:
#     pickle.dump(fold_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)

