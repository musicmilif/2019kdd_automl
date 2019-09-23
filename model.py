import os
os.system('pip3 install hyperopt')
os.system('pip3 install lightgbm')
os.system('pip3 install pandas==0.24.2')

import gc
import copy
import numpy as np
import pandas as pd

from automl import AutoML
from CONSTANT import MAIN_TABLE_NAME
from merge import merge_table
from preprocessor import DateSorting, ReduceMemory
from feature_engineering import FeatureEngineer
from cleaner import CleanDataFrame
from selector import FeatureSelector
from util import Config, timeit


class Model:
    def __init__(self, info, kfolds=5):
        self.config = Config(info)
        self.tables = None
        self.date_sorting = DateSorting(self.config)
        self.prev_cleaning = CleanDataFrame(prev=True)
        self.post_cleaning = CleanDataFrame(prev=False)
        self.reduce_memory = ReduceMemory()
        self.feature_engineer = FeatureEngineer()
        self.selector = FeatureSelector(self.config)
        self.automl = AutoML(self.config, kfolds=kfolds)

    @timeit
    def fit(self, Xs, y, time_ramain):
        self.tables = copy.deepcopy(Xs)
        self.tables.pop(MAIN_TABLE_NAME, None)
        X = merge_table(Xs, self.config)
        del Xs; gc.collect()
        # Preprocessing
        X, y = self.date_sorting.sorting(X, y)
        # Feature Engineering and Feature Selection
        X = self.prev_cleaning.fit_transform(X, y)
        X = self.reduce_memory.fit_transform(X)
        X = self.feature_engineer.fit_transform(X, y)
        X = self.post_cleaning.fit_transform(X, y)
        X = self.selector.fit_transform(X, y)
        # Modeling
        self.automl.fit(X, y)

    @timeit
    def predict(self, X_test, time_remain):
        Xs = self.tables
        Xs[MAIN_TABLE_NAME] = X_test
        X = merge_table(Xs, self.config)
        del Xs; gc.collect()
        # Feature Engineering and Feature Selection
        X = self.prev_cleaning.transform(X)
        X = self.reduce_memory.transform(X)
        X = self.feature_engineer.transform(X)
        X = self.post_cleaning.transform(X)
        X = self.selector.transform(X)
        # Modeling
        result = self.automl.predict(X)

        return pd.Series(result)

