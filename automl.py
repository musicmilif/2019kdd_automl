from typing import Dict, List

import hyperopt
import lightgbm as lgb

import numpy as np
import pandas as pd
from tqdm import tqdm
from hyperopt import STATUS_OK, Trials, hp, space_eval, tpe
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from feature_engineering import CategoricalEngineer
from util import Config, log, timeit
SEED = 1


def data_sample(X: pd.DataFrame, y: pd.Series, nrows: int=5000, balance: bool=False):
    # -> (pd.DataFrame, pd.Series):
    # Imbalanced data sample
    if balance:
        y_dist = y.value_counts()
        larger_cat, less_cat = y_dist.index
        ratio = y_dist[less_cat]/y_dist[larger_cat]
        balance = balance and (ratio < .05)

    if balance:
        sample_frac = max(ratio*20, .05)
        sample_cnt = int(sample_frac*len(y))
        less_idx = y.loc[y==less_cat].index
        larger_idx = y.loc[y==larger_cat].iloc[-int(sample_cnt*.3):].index
        larger_idx = larger_idx.append(y.loc[y==larger_cat].iloc[:-int(sample_cnt*.3)].\
                                         sample(frac=sample_frac*.7).index)
        y_sample = y.loc[less_idx.append(larger_idx)].sample(frac=1)
        X_sample = X.loc[y_sample.index]
        X_sample = X_sample.reset_index(drop=True)
        y_sample = y_sample.reset_index(drop=True)
    else:
        X_sample = X
        y_sample = y

    # Random sample to reduce data size
    if nrows and len(X_sample) > nrows:
        sss = StratifiedShuffleSplit(test_size=nrows)
        _, sample_idx = next(sss.split(X_sample, y_sample))
        X_sample = X_sample.iloc[sample_idx]
        y_sample = y_sample.iloc[sample_idx]

    return X_sample, y_sample


class LGBM(BaseEstimator, ClassifierMixin):
    def __init__(self, config, kfolds):
        self.config = config
        self.kfolds = kfolds
        self.balance_sample = False

    @timeit
    def fit(self, X, y, show_imp=False):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': False,
            'verbosity': -1,
            'seed': SEED,
            'num_threads': 4
        }

        # Hyper parameters search
        X_sample, y_sample = data_sample(X, y, 150000, balance=False)
        hyperparams = self.hyper_opt(X_sample, y_sample, params)

        # Imbalanced sampling and bagging
        max_iters, early_iters = 1000, 50
        self.aucs = []
        i, total_iterations = 0, 0
        skf = StratifiedKFold(n_splits=self.kfolds, shuffle=True, random_state=SEED)
        skf = skf.split(X, y)
        while total_iterations < max_iters and i < self.kfolds:
            train_idx, valid_idx = next(skf)
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_train, y_train = data_sample(X_train, y_train, None, balance=self.balance_sample)
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            # Model Training
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            self.config[f'lgbm_{i}'] = lgb.train({**params, **hyperparams},
                                                 train_data,
                                                 max_iters,
                                                 valid_data,
                                                 early_stopping_rounds=early_iters,
                                                 verbose_eval=200)
            self.aucs.append(self.config[f'lgbm_{i}'].best_score['valid_0'][params['metric']])
            total_iterations += min(self.config[f'lgbm_{i}'].best_iteration+early_iters, max_iters)
            i += 1
        self.nbags = i

        if show_imp:
            clr = self.config[f'lgbm_0']
            feat_imp = sorted([p for p in zip(clr.feature_name(), clr.feature_importance())], 
                              key=lambda x: -x[-1])
            log(f'Feature Importance: {feat_imp}')
        log(f'Average bagging AUC: {np.mean(self.aucs)}+{np.std(self.aucs)}')

    @timeit
    def predict(self, X: pd.DataFrame, method: str='geo') -> List:
        weight_sum = 0
        bagging_weight = self.aucs / max(self.aucs)

        if method == 'geo':
            pred_y = np.ones(len(X))
            for i in tqdm(range(self.nbags)):
                if bagging_weight[i] > 0.95:
                    pred_y *= self.config[f'lgbm_{i}'].predict(X)**bagging_weight[i]
                    weight_sum += bagging_weight[i]

            pred_y = pred_y**(1/weight_sum)
        else:
            pred_y = np.zeros(len(X))
            for i in tqdm(range(self.nbags)):
                if bagging_weight[i] > 0.95:
                    pred_y += self.config[f'lgbm_{i}'].predict(X)*bagging_weight[i]
    
            pred_y = pred_y / weight_sum

        return pred_y

    @timeit
    def hyper_opt(self, X, y, params: Dict):
        sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=SEED)
        train_idx, valid_idx = next(sss.split(X, y))
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_train, y_train = data_sample(X_train, y_train, None, balance=self.balance_sample)
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)

        space = {
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.5)),
            'max_depth': hp.choice('max_depth', [-1, 3, 5, 8, 13]),
            'num_leaves': hp.choice('num_leaves', 2**np.arange(3, 11)),
            'feature_fraction': hp.quniform('feature_fraction', 0.3, 0.8, 0.1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 2),
            'reg_lambda': hp.uniform('reg_lambda', 0, 2),
            'min_child_weight': hp.uniform('min_child_weight', 0.5, 10),
        }

        def objective(hyperparams):
            model = lgb.train({**params, **hyperparams}, train_data, 500,
                            valid_data, early_stopping_rounds=50, verbose_eval=0)
            score = model.best_score['valid_0'][params['metric']]

            return {'loss': -score, 'status': STATUS_OK}

        trials = Trials()
        best = hyperopt.fmin(fn=objective, space=space, trials=trials,
                            algo=tpe.suggest, max_evals=7, verbose=1,
                            rstate=np.random.RandomState(1))

        hyperparams = space_eval(space, best)
        log(f"auc = {-trials.best_trial['result']['loss']:0.4f} {hyperparams}")

        return hyperparams


class AutoML(BaseEstimator, ClassifierMixin):
    def __init__(self, config, kfolds):
        self.config = config
        self.lgbm = LGBM(self.config, kfolds=kfolds)
    
    def fit(self, X, y):
        self.lgbm.fit(X, y, show_imp=True)
        return self

    def predict(self, X):
        preds_lgbm = self.lgbm.predict(X)
        return preds_lgbm
