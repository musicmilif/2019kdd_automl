import os
import pickle
import time
from typing import Any

import CONSTANT

nesting_level = 0
is_start = None

class Timer:
    def __init__(self):
        self.start = time.time()
        self.history = [self.start]

    def check(self, info):
        current = time.time()
        log(f"[{info}] spend {current - self.history[-1]:0.2f} sec")
        self.history.append(current)

def timeit(method, start_log=None):
    def timed(*args, **kw):
        global is_start
        global nesting_level

        if not is_start:
            print()

        is_start = True
        log(f"Start [{method.__name__}]:" + (start_log if start_log else ""))
        nesting_level += 1

        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()

        nesting_level -= 1
        log(f"End   [{method.__name__}]. Time elapsed: {end_time - start_time:0.2f} sec.")
        is_start = False

        return result

    return timed


def log(entry: Any):
    global nesting_level
    space = "-" * (4 * nesting_level)
    print(f"{space}{entry}")

def show_dataframe(df):
    if len(df) <= 30:
        print(f"content=\n"
              f"{df}")
    else:
        print(f"dataframe is too large to show the content, over {len(df)} rows")

    if len(df.dtypes) <= 100:
        print(f"types=\n"
              f"{df.dtypes}\n")
    else:
        print(f"dataframe is too wide to show the dtypes, over {len(df.dtypes)} columns")


class Config:
    def __init__(self, info):
        self.data = {
            "start_time": time.time(),
            **info
        }
        self.data["tables"] = {}
        for tname, ttype in info['tables'].items():
            self.data['tables'][tname] = {}
            self.data['tables'][tname]['type'] = ttype

    @staticmethod
    def aggregate_op(col):
        ops = {
            CONSTANT.NUMERICAL_TYPE: ['mean'],
            CONSTANT.CATEGORY_TYPE: ['count'],
        }
        if col.startswith(CONSTANT.NUMERICAL_PREFIX):
            return ops[CONSTANT.NUMERICAL_TYPE]
        if col.startswith(CONSTANT.CATEGORY_PREFIX):
            return ops[CONSTANT.CATEGORY_TYPE]

    def time_left(self):
        return self["time_budget"] - (time.time() - self["start_time"])

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __contains__(self, key):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)


import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable, validation
from multiprocessing import cpu_count, Pool

class TimeShiftSplit(_BaseKFold):
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = validation._num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1

        indices = np.arange(n_samples)
        test_size = (n_samples // n_folds)
        train_starts = range(0, n_samples - 2*test_size, test_size)
        for train_start in train_starts:
            valid_start = train_start + test_size
            yield (indices[train_start:valid_start],
                   indices[valid_start:(valid_start+test_size)])


def parallelize_apply(func, df, cores=4):
    cores = min(len(df.columns), cpu_count()) if not cores else cores
    with Pool(cores) as pool:
        series = [df[col] for col in df.columns]
        list_series = pool.map(func, series)

        return list_series


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.select_dtypes(include=['number']).columns:
        c_min = df[col].min()
        c_max = df[col].max()
        if 'int' in str(df[col].dtype) and c_min < 0:
            if np.iinfo(np.int8).min <= c_min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif np.iinfo(np.int16).min <= c_min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif np.iinfo(np.int32).min <= c_min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif np.iinfo(np.int64).min <= c_min and c_max <= np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)  
        elif 'int' in str(df[col].dtype) and c_min >= 0:
            if c_max <= np.iinfo(np.uint8).max:
                df[col] = df[col].astype(np.uint8)
            elif c_max <= np.iinfo(np.uint16).max:
                df[col] = df[col].astype(np.uint16)
            elif c_max <= np.iinfo(np.uint32).max:
                df[col] = df[col].astype(np.uint32)
            elif c_max <= np.iinfo(np.uint64).max:
                df[col] = df[col].astype(np.uint64)  
        else:
            if np.finfo(np.float16).min < c_min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif np.finfo(np.float32).min < c_min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        red_mem = 100 * (start_mem - end_mem) / start_mem
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, red_mem))
    
    return df

