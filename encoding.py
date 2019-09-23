import gc
import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from multiprocessing import cpu_count, Pool
from CONSTANT import NUMERICAL_PREFIX


class BaysianTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, y):
        self.y = y
        self.col_stats = dict()
    
    def get_baysian_stats(self, encode_stat, alpha, beta):
        if encode_stat == 'mean':
            num = alpha
            dem = alpha + beta
        elif encode_stat == 'mode':
            num = alpha - 1
            dem = alpha + beta - 2
        elif encode_stat == 'median':
            num = alpha - 1/3
            dem = alpha + beta - 2/3
        elif encode_stat == 'var':
            num = alpha * beta
            dem = (alpha + beta)**2 * (alpha + beta + 1)
        elif encode_stat == 'skewness':
            num = 2*(beta - alpha) * np.sqrt(alpha + beta + 1)
            dem = (alpha + beta + 2) * np.sqrt(alpha * beta)
        elif encode_stat == 'kurtosis':
            num = 6*(alpha - beta)**2 * (alpha + beta + 1) - alpha * beta * (alpha + beta + 2)
            dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)
        else:
            num = self.prior_mean
            dem = np.ones_like(self.N_prior)
        
        return num / dem

    def fit(self, df):
        df['target'] = self.y
        self.prior_mean = df['target'].mean()

        for col in df.columns.difference(pd.Index(['target'])):
            stats = df[[col, 'target']].groupby(col)
            stats = stats.agg(['sum', 'count'])['target']
            stats.rename(columns={'sum': 'n', 'count': 'N'}, inplace=True)
            stats.reset_index(level=0, inplace=True)           
            self.col_stats[col] = stats

        return self
        
    def transform(self, df):
        all_df = self.parallelize_apply(df)

        return all_df

    def func(self, series, col, stats='median', N_min=100):
        feat_stats = pd.merge(series.to_frame(), self.col_stats[col], how='left')
        n, N = feat_stats['n'].copy(), feat_stats['N'].copy()
        # Fill missing values
        n.fillna(self.prior_mean, inplace=True)
        N.fillna(1.0, inplace=True)
        # Prior parameters
        self.N_prior = np.maximum(N_min-N, 0)
        alpha_prior = self.prior_mean * self.N_prior
        beta_prior = (1 - self.prior_mean) * self.N_prior
        # posterior parameters
        alpha_post = alpha_prior + n
        beta_post =  beta_prior + N - n
        
        res = self.get_baysian_stats(stats, alpha_post, beta_post)
        res = pd.Series(res, name=f'{NUMERICAL_PREFIX}{col}_{stats}')
        return res.fillna(np.nanmedian(res))
        
    def parallelize_apply(self, df, cores=4):
        cores = min(len(df.columns), cpu_count()) if not cores else cores
        with Pool(cores) as pool:
            cols = [col for col in df.columns]
            serieses = [df[col] for col in cols]
            list_series = pool.starmap(self.func, zip(serieses, cols))

            return pd.concat(list_series, axis=1)

