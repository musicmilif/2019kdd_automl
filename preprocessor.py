import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin
from util import timeit, reduce_mem_usage
SEED = 1


class DateSorting(object):
    def __init__(self, config):
        self.config = config

    def sorting(self, X, y=None):
        if self.config['time_col'] in X.columns:
            X = X.sort_values(self.config['time_col'])
            y = y[X.index].reset_index(drop=True) if isinstance(y, pd.Series) else y
            X = X.reset_index(drop=True)
        
        return X, y


class Spliter(object):
    def __init__(self, nbags):
        self.nbags = nbags

    def train_valid_split(self, X, y):
        return train_test_split(X, y, test_size=.1, shuffle=False)

    def kfold_split(self, X, y=None, train=True):
        """1 to nbags: Training data; 0: Testing data
        """
        if isinstance(y, pd.Series) and train:
            X['Fold'] = np.nan
            skf = StratifiedKFold(n_splits=self.nbags, shuffle=True, random_state=SEED)
            for i, (_, valid_idx) in enumerate(skf.split(X.index, y)):
                X['Fold'].iloc[valid_idx] = i + 1
        elif train:
            X['Fold'] = 0
        else:
            X['Fold'] = -1

        return X


class ReduceMemory(BaseEstimator, TransformerMixin):
    def fit(self, X):
        self.dtypes = None
        return self
    
    @timeit
    def transform(self, X):
        if not isinstance(self.dtypes, dict):
            X = reduce_mem_usage(X)
            self.dtypes = X.dtypes
        else:
            X = X.astype(self.dtypes)
        
        return X
