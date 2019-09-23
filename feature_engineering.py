import datetime
from util import timeit
from CONSTANT import NUMERICAL_PREFIX, CATEGORY_PREFIX, MULTI_CAT_PREFIX, TIME_PREFIX

import gc
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from encoding import BaysianTargetEncoder


class NumericalEngineer(BaseEstimator, TransformerMixin):
    @timeit
    def fit(self, X):
        self.feats = [col for col in X if col.startswith(NUMERICAL_PREFIX)]
        # Get skewed features' name
        self.num_cols = [col for col in self.feats if np.abs(skew(X[col])) > 3]
        self.feats += [f'{col}_square' for col in self.feats]
        self.feats += [f'{col}_log' for col in self.num_cols]
        
        return self

    @timeit
    def transform(self, X):
        for col in [col for col in X if col.startswith(NUMERICAL_PREFIX)]:
            X[f'{col}_square'] = X[col].apply(lambda x: x**2)
        for col in self.num_cols:
            sgn = (X[col] >= 0).astype(np.int8) * 2 - 1
            X[f'{col}_log'] = sgn * np.log1p(np.abs(X[col]))
        
        return X


class CategoricalEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fe_cols = dict()
        self.bte_cols = []

    @timeit
    def fit(self, X, y):
        self.cat_feats = [col for col in X if col.startswith(CATEGORY_PREFIX)]
        for col in self.cat_feats:
            # Drop too rare categories
            val_cnts = X[col].value_counts()
            # Frequent Encoding
            if 1 < len(val_cnts) <= 10:
                self.fe_cols[col] = {k: v+1 for k, v in zip(val_cnts.index, range(len(val_cnts)))}
            # Baysian Target Encoding
            X[col] = X[col].fillna('0')
            if 10 < len(val_cnts):
                self.bte_cols.append(col)
        self.bte = BaysianTargetEncoder(y)
        self.bte.fit(X[self.bte_cols])

        self.feats = [col for col in self.fe_cols]
        self.feats += [f'{NUMERICAL_PREFIX}{col}_median' for col in self.bte_cols]

        return self
    
    @timeit
    def transform(self, X):
        if self.bte_cols:
            X[self.bte_cols] = X[self.bte_cols].fillna('0')
            bte_X = self.bte.transform(X[self.bte_cols])
            bte_X.index = X.index
            X = pd.concat([X, bte_X], axis=1)
        # Label Encoding
        for col in self.fe_cols:
            X[col] = X[col].map(self.fe_cols[col]).fillna(0)

        return X


class MultiCatEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, max_ncols=3):
        self.max_ncols = max_ncols
        self.fe_cols = dict()

    @timeit
    def fit(self, X):
        self.feats = []
        self.mul_feats = [col for col in X if col.startswith(MULTI_CAT_PREFIX)]
        for i, col in enumerate(self.mul_feats):
            if i < self.max_ncols*2:
                self.feats += [f'{col}_cnt']
            if i < self.max_ncols:
                self.feats += [f'{CATEGORY_PREFIX}{col}_fst', 
                               f'{CATEGORY_PREFIX}{col}_lst']

        return self

    @timeit
    def transform(self, X):
        # Count number of elements in each columns
        for i, col in enumerate(self.mul_feats):
            if i < self.max_ncols*2:
                # Calculate counts in feature
                cnt_col = f'{col}_cnt'
                X[cnt_col] = X[col].str.count(',') + 1
            if i < self.max_ncols:
                # Get first and last category in feature
                fst_col = f'{CATEGORY_PREFIX}{col}_fst'
                lst_col = f'{CATEGORY_PREFIX}{col}_lst'
                X[fst_col] = X[col].apply(self.first_trans)
                X[lst_col] = X[col].apply(self.last_trans)
                if col not in self.fe_cols:
                    val_cnts = pd.concat([X[fst_col], X[lst_col]], axis=0).value_counts()
                    min_cnt = len(X)/1000
                    val_cnts = val_cnts[val_cnts >= min_cnt]
                    self.fe_cols[col] = {k: v+1 for k, v in zip(val_cnts.index, range(len(val_cnts)))}
                X[fst_col] = X[fst_col].map(self.fe_cols[col]).fillna(0)
                X[lst_col] = X[lst_col].map(self.fe_cols[col]).fillna(0)

        return X
    
    @staticmethod
    def first_trans(string):
        idx = string.find(',')
        return string[:idx] if idx >= 0 else string

    @staticmethod
    def last_trans(string):
        return string[string.rfind(',')+1:]
    

class DatetimeEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.weeks_cols = []
        self.days_cols = []
        self.hours_cols = []
        self.minutes_cols = []
        self.feats = []

    @timeit
    def fit(self, X):
        self.date_feats = [col for col in X if col.startswith(TIME_PREFIX)]
        for col in self.date_feats:
            time_diff = X[col].max() - X[col].min()
            if time_diff.days//7 > 1:
                self.weeks_cols.append(col)
                self.feats += [f'{col}_year', f'{col}_month', f'{col}_DoW', f'{col}_WoY', f'{col}_hour']

        return self

    @timeit
    def transform(self, X):
        for col in self.weeks_cols:
            X[f'{col}_year'] = X[col].dt.year
            X[f'{col}_month'] = X[col].dt.month
            X[f'{col}_DoW'] = X[col].dt.dayofweek
            X[f'{col}_WoY'] = X[col].dt.weekofyear
            X[f'{col}_hour'] = X[col].dt.hour

        X.drop(self.date_feats, axis=1, inplace=True); gc.collect()
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        # Numerical features
        self.num_engineer = NumericalEngineer()
        self.num_engineer.fit(X)
        # Categorical features
        self.cat_engineer = CategoricalEngineer()
        self.cat_engineer.fit(X, y)
        # Multi-categorical features
        self.mul_engineer = MultiCatEngineer()
        self.mul_engineer.fit(X)
        # Datetime features
        self.date_engineer = DatetimeEngineer()
        self.date_engineer.fit(X)

        # Make sure order of columns in dataframe are the same
        self.col_order = self.num_engineer.feats + self.cat_engineer.feats + \
                         self.mul_engineer.feats + self.date_engineer.feats    
        return self
    
    def transform(self, X):
        X = self.num_engineer.transform(X)
        X = self.cat_engineer.transform(X)
        X = self.mul_engineer.transform(X)
        X = self.date_engineer.transform(X)

        return X[self.col_order]

