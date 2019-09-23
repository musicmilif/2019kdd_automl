import gc
import numpy as np
import pandas as pd
from util import timeit
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb

from automl import data_sample
from util import Config, log, timeit, TimeShiftSplit
from CONSTANT import TIME_PREFIX


class TimeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, skip, nbags=5, threshold=None, show_cv=True):
        self.skip = skip
        self.nbags = nbags
        self.threshold = threshold
        self.show_cv = show_cv
        self.params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'num_threads': 4,
                'learning_rate': 0.01,
                'max_depth': 8,
                'num_leaves': 2**5,
                'feature_fraction': 0.66,
            }

    @timeit
    def fit(self, X, y):
        X = X.drop(self.skip, axis=1)
        iter_num = []
        self.feats = []
        tss = TimeShiftSplit(n_splits=self.nbags)

        for train_idx, valid_idx in tss.split(X.index, y):
            # Sample and dealing with imbalance
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_train, y_train = data_sample(X_train, y_train, 60000, balance=True)
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            X_valid, y_valid = data_sample(X_valid, y_valid, 40000)

            # Fit Model to get feature importance
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            clr = lgb.train(self.params, train_data, 500, valid_data, 
                            early_stopping_rounds=50, verbose_eval=False)

            # Drop Time Feature if iteration varies a lot or stop very early
            feat_name, feat_imp = clr.feature_name(), clr.feature_importance()
            col_imp = [(n, v) for n, v in zip(feat_name, feat_imp) if n.startswith(TIME_PREFIX)] 
            if  clr.best_iteration < 10:
                self.feats +=  [n for n, v in col_imp if v > 0]
            elif clr.best_iteration < 100:
                self.feats += [n for n, v in col_imp if v/sum(feat_imp) > 0.1]
        
        self.feats = list(set(self.feats))

        return self

    @timeit
    def transform(self, X):
        self.feats = [col for col in X.columns if col in self.feats]
        return X.drop(self.feats, axis=1) if self.feats else X


class ModelSelector(BaseEstimator, TransformerMixin):
    def __init__(self, skip, nbags=5, threshold=None, show_cv=True):
        self.skip = skip
        self.nbags = nbags
        self.threshold = threshold
        self.show_cv = show_cv
        self.params = {
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'num_threads': 4,
                'learning_rate': 0.01,
                'max_depth': 8,
                'num_leaves': 2**5,
                'feature_fraction': 0.66,
            }

    @timeit
    def fit(self, X, y):
        X = X.drop(self.skip, axis=1)
        tss = TimeShiftSplit(n_splits=self.nbags)
        self.feats = set(X.columns)
        coef_var = dict()

        for train_idx, valid_idx in tss.split(X.index, y):
            # Sample and dealing with imbalance
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_train, y_train = data_sample(X_train, y_train, 60000, balance=True)
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            X_valid, y_valid = data_sample(X_valid, y_valid, 40000)

            # Fit Model to get feature importance
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            clr = lgb.train(self.params, train_data, 200, valid_data, 
                            early_stopping_rounds=200, verbose_eval=False)
            # Get column name which cumulative sum of importance larger than threshold
            col_imp = [(n, v) for n, v in zip(clr.feature_name(), clr.feature_importance())]
            drop = set([n for (n, v) in col_imp if v < 30])
            self.feats = self.feats.intersection(drop)

            # Save importance to dictionary so that I can compute CV
            for col, imp in col_imp:
                coef_var[col] = coef_var.get(col, []) + [imp]

        # Drop features with high coefficent Varience (truncate on threshold)
        self.threshold = self.threshold if self.threshold else int(0.1*(len(X.columns)-len(self.feats)))+1
        coef_var = {k: np.std(v)/(np.mean(v)+1e-5) for k, v in coef_var.items()}
        drop = sorted([(k, v) for k, v in coef_var.items() if v > 1], key=lambda x: -x[1])[:self.threshold]
        drop = set([k for (k, v) in drop])
        self.feats = self.feats.union(drop)

        if self.show_cv:
            import operator
            print(sorted(coef_var.items(), key=operator.itemgetter(1), reverse=True))
            print('\n', self.feats)
        log(f'ModelSelector: {self.feats}')

        return self

    @timeit
    def transform(self, X):
        self.feats = [col for col in X.columns if col in self.feats]
        return X.drop(self.feats, axis=1) if self.feats else X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.skip_cols = []
        self.time_selector = TimeSelector(skip=self.skip_cols)
        self.model_selector = ModelSelector(skip=self.skip_cols, show_cv=False)
        
    @timeit
    def fit(self, X, y):
        self.time_selector.fit(X, y)
        drop_time = self.time_selector.feats
        self.model_selector.fit(X.drop(drop_time, axis=1), y)

        return self

    @timeit
    def transform(self, X):
        X = self.time_selector.transform(X)
        X = self.model_selector.transform(X)

        return X

