from CONSTANT import NUMERICAL_PREFIX, CATEGORY_PREFIX, MULTI_CAT_PREFIX, TIME_PREFIX
from util import log, timeit

import gc
import warnings
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit
from multiprocessing import cpu_count, Pool
warnings.filterwarnings('ignore')


class FillNull(BaseEstimator, TransformerMixin):
    @timeit
    def fit(self, X):
        self.feats = [col for col in X if col.startswith(MULTI_CAT_PREFIX)]
        return self

    @timeit
    def transform(self, X):
        X[self.feats] = X[self.feats].fillna('')
        return X


class CleanCategory(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop_cols = []

    @timeit
    def fit(self, X, ths_hold=0.1):
        cnt = int(len(X)*ths_hold)
        cat_feats = [col for col in X if col.startswith(CATEGORY_PREFIX)]
        nuniques = [(col, X[col].nunique()) for col in cat_feats]
        # Hash high cardinality and drop singular
        self.drop_cols += [col for (col, nuniq) in nuniques if nuniq > cnt]
        self.drop_cols += [col for (col, nuniq) in nuniques if nuniq == 1]

        return self

    @timeit
    def transform(self, X):
        self.drop_cols = [col for col in X.columns if col in self.drop_cols]

        return X.drop(self.drop_cols, axis=1) if self.drop_cols else X


class CleanDensity(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.remain_cats = dict()
    
    @timeit
    def fit(self, X, y):
        # Drop heavily time dependency columns
        cat_feats = [col for col in X if col.startswith(CATEGORY_PREFIX)]
        drop_tuple = self.parallel_loop(self.calc_density, X, y, cat_feats)
        drop_cols = [col for col, drop in drop_tuple if drop]
        log(f'CleanDensity: {drop_cols}')

        # Drop rare categories
        for col in cat_feats:
            val_cnts = X[col].value_counts()
            idx = int(len(val_cnts)*0.99)
            min_cnt = len(X)/1000 if col in drop_cols else max(3, val_cnts.iloc[idx])
            keep_cats = val_cnts[val_cnts > min_cnt].index
            self.remain_cats[col] = keep_cats

        return self

    @timeit
    def transform(self, X):
        for col in self.remain_cats:
            X.loc[~X[col].isin(self.remain_cats[col]), col] = np.nan

        return X

    def calc_density(self, df):
        cnt = 0
        n_splits = 5
        # Calculate the difference between shifted training duration
        thrs = 3
        col = list(set(df.columns) - set(['target']))[0]
        tss = TimeSeriesSplit(n_splits=n_splits)
        total_cnt = 0
        for idx, (train_idx, valid_idx) in enumerate(tss.split(df)):
            if idx > 1:
                train_len = int(len(valid_idx)*(idx+1)/2)
                train_idx = pd.Index(np.random.choice(train_idx, train_len, replace=False))
            # Training target mean
            train_vc = df[col].iloc[train_idx].value_counts()
            train_vc = train_vc.loc[train_vc > thrs].index
            series = df.iloc[train_idx]
            series = series.loc[series[col].isin(train_vc)]
            train_mean = series.groupby(col)['target'].agg({'avg': 'mean', 'cnt': 'count'})
            # Validation target mean
            valid_vc = df[col].iloc[valid_idx].value_counts()
            valid_vc = valid_vc.loc[valid_vc > thrs].index
            series = df.iloc[valid_idx]
            series = series.loc[series[col].isin(valid_vc)]
            valid_mean = series.groupby(col)['target'].agg({'avg': 'mean', 'cnt': 'count'})
            # Compare two distribution with p-value
            prop = pd.merge(train_mean, valid_mean, how='inner', left_index=True, right_index=True)
            total_cnt += len(prop)
            for _, cat in prop.iterrows():
                p_value = self.t_test(cat)
                if p_value < 0.01 or p_value > 0.99:
                    cnt += 1
                if cnt > total_cnt/10:
                    return (col, True)

        return (col, False)

    @staticmethod
    def t_test(cat):
        p = cat['avg_x']*cat['cnt_x']+cat['avg_y']*cat['cnt_y']
        p = p/(cat['cnt_x']+cat['cnt_y'])
        p += 1e-8
        z = (cat['avg_x']-cat['avg_y']) / np.sqrt((p*(1-p)*(1/cat['cnt_x']+1/cat['cnt_y'])))
        p_value = t.sf(abs(z), df=cat['cnt_x']+cat['cnt_y']-2)*2

        return p_value

    @staticmethod
    def parallel_loop(func, X, y, cat_feats, cores=None):
        cores = min(len(X.columns), cpu_count()) if not cores else cores
        with Pool(cores) as pool:
            y.name = 'target'
            X = pd.concat([X, y], axis=1)
            params = [X[[col, 'target']] for col in cat_feats]
            prob_tuple = pool.map(func, params)

        return prob_tuple


class CleanDatetime(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.drop = []

    @timeit
    def fit(self, X):
        cyc_feats = ['_hour', '_minute', '_second']
        date_feats = [col for col in X for c in cyc_feats if col.endswith(c)]
        for col in date_feats:
            cyc_rate = np.sum(X[col].diff() < 0) + 1
            if cyc_rate < 3:
                self.drop.append(col)

        log(f'CleanDatetime: {self.drop}')

        return self

    @timeit
    def transform(self, X):
        self.drop = [col for col in X.columns if col in self.drop]
        return X.drop(self.drop, axis=1) if self.drop else X


class CleanCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=.95):
        self.feats = []
        self.threshold = threshold

    @timeit
    def fit(self, X):
        corr = X.sample(frac=.5).corr().abs()
        corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
        self.feats += [col for col in corr.columns if any(corr[col] > self.threshold)]
        log(f'CleanCorrelation: {self.feats}')

        return self

    @timeit
    def transform(self, X):
        self.feats = [col for col in X.columns if col in self.feats]
        return X.drop(self.feats, axis=1) if self.feats else X


class CleanDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, prev):
        self.prev = prev
        self.fill_null = FillNull()
        self.clean_dense = CleanDensity()
        self.clean_cat = CleanCategory()
        self.clean_corr = CleanCorrelation()
        self.clean_date = CleanDatetime()

    @timeit
    def fit(self, X, y):
        if self.prev:
            self.fill_null.fit(X)
            self.clean_cat.fit(X)
            skip_col = self.clean_cat.drop_cols
            self.clean_dense.fit(X.drop(skip_col, axis=1), y)
        else:
            # self.clean_date.fit(X)
            self.clean_corr.fit(X)

        return self

    @timeit
    def transform(self, X):
        if self.prev:
            X = self.fill_null.transform(X)
            X = self.clean_cat.transform(X)
            X = self.clean_dense.transform(X)
        else:
            # X = self.clean_date.transform(X)
            X = self.clean_corr.transform(X)

        return X

