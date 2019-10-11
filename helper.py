import math
import time

import hyperopt
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset


class LargeTabularDataset(IterableDataset):
    def __init__(self, data_path, cont_cols, cat_cols, output_col, chunksize, shuffle=False, is_hdf=False):
        self.data_path = data_path
        if is_hdf:
            with pd.HDFStore(data_path) as store:
                self.nb_samples = store.get_storer('data').nrows
        else:
            # Assume it is csv
            # self.nb_samples = pd.read_csv(data_path, usecols=[0]).shape[0]
            with open(data_path) as f:
                self.nb_samples = max(sum(1 for line in f if line) - 1, 0)

        self.cont_cols = cont_cols
        self.cat_cols = cat_cols
        self.output_col = output_col
        self.chunksize = chunksize
        self.shuffle = shuffle
        self.is_hdf = is_hdf

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.nb_samples
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.nb_samples / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.nb_samples)
        return LargeTabularDatesetIterator(self, iter_start, iter_end)


class LargeTabularDatesetIterator:

    def __init__(self, tabular_dataset, start_row, end_row):
        self._tabular_dataset = tabular_dataset

        if self._tabular_dataset.is_hdf:
            self._pd_chunk_iter = iter(
                pd.read_hdf(
                    self._tabular_dataset.data_path,
                    start=start_row, stop=end_row,
                    chunksize=self._tabular_dataset.chunksize))
        else:
            # Assume it is csv
            self._pd_chunk_iter = pd.read_csv(
                self._tabular_dataset.data_path,
                skiprows=range(1, start_row + 1),
                nrows=end_row - start_row,
                chunksize=self._tabular_dataset.chunksize)

    def __next__(self):
        x = next(self._pd_chunk_iter)
        if self._tabular_dataset.shuffle:
            x = x.sample(frac=1)
        cont_x = x[self._tabular_dataset.cont_cols].astype(np.float32).squeeze(axis=0).values
        cat_x = x[self._tabular_dataset.cat_cols].astype(np.int64).squeeze(axis=0).values
        # 'y' is a vector of scalar
        y = x[self._tabular_dataset.output_col].astype(np.int64).values

        return (cont_x, cat_x), y


class BiDirectionalDict:

    def __init__(self, d=None, none_value=None):
        if d is None:
            d = {}
        self.none_value = none_value
        self._forward_dict = {}
        self._backward_dict = {}
        self.update(d)

    def __len__(self):
        return len(self._forward_dict)

    def __str__(self):
        return 'FORWARD: ' + str(self._forward_dict) + '\nBACKWARD: ' + str(self._backward_dict)

    def add(self, first_key, second_key):
        if first_key in self._forward_dict:
            prev_second_key = self._forward_dict[first_key]
            if second_key != prev_second_key:
                del self._backward_dict[prev_second_key]
        if second_key in self._backward_dict:
            prev_first_key = self._backward_dict[second_key]
            if first_key != prev_first_key:
                del self._forward_dict[prev_first_key]

        self._forward_dict[first_key] = second_key
        self._backward_dict[second_key] = first_key

    def forward(self, first_key):
        if first_key not in self._forward_dict:
            return self.none_value
        return self._forward_dict[first_key]

    def backward(self, second_key):
        if second_key not in self._backward_dict:
            return self.none_value
        return self._backward_dict[second_key]

    def update(self, dictionary, forward=True):
        if forward:
            for first_key, second_key in dictionary.items():
                self.add(first_key, second_key)
        else:
            for second_key, first_key in dictionary.items():
                self.add(first_key, second_key)

    def first_keys(self):
        return self._forward_dict.keys()

    def second_keys(self):
        return self._backward_dict.keys()

    def items(self, forward=True):
        if forward:
            return self._forward_dict.items()
        else:
            return self._backward_dict.items()


class NaN(float):

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, float('nan'))

    def __hash__(self):
        return np.nan.__hash__()

    def __eq__(self, other):
        return np.isnan(other)

    def __bool__(self):
        return False


nan = NaN()


# credit to @guiferviz for the memory reduction
def memory_usage_mb(df, *args, **kwargs):
    """Dataframe memory usage in MB. """
    return df.memory_usage(*args, **kwargs).sum() / 1024 ** 2


def reduce_memory_usage(df, deep=True, verbose=True):
    # All types that we want to change for "lighter" ones.
    # int8 and float16 are not include because we cannot reduce
    # those data types.
    # float32 is not include because float16 has too low precision.
    numeric2reduce = ["int16", "int32", "int64", "float64"]
    start_mem = 0
    if verbose:
        start_mem = memory_usage_mb(df, deep=deep)

    for col, col_type in df.dtypes.iteritems():
        best_type = None
        if col_type in numeric2reduce:
            downcast = "integer" if "int" in str(col_type) else "float"
            df[col] = pd.to_numeric(df[col], downcast=downcast)
            best_type = df[col].dtype.name
        # Log the conversion performed.
        if verbose and best_type is not None and best_type != str(col_type):
            print(f"Column '{col}' converted from {col_type} to {best_type}")

    if verbose:
        end_mem = memory_usage_mb(df, deep=deep)
        diff_mem = start_mem - end_mem
        percent_mem = 100 * diff_mem / start_mem
        print(f"Memory usage decreased from"
              f" {start_mem:.2f}MB to {end_mem:.2f}MB"
              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")

    return df


def safe_del(var_list, local_context):
    for v in var_list:
        if v in local_context:
            del local_context[v]


class DataEpochGenerator:
    def __init__(self, chunk_loader, batch_size, epoch=100):
        self.chunk_loader = chunk_loader
        self.batch_size = batch_size
        self.epoch = epoch

    def __iter__(self):
        return data_epoch_generator(self.chunk_loader, self.batch_size, self.epoch)


def data_epoch_generator(chunk_loader, batch_size, epoch=100):
    for _ in range(epoch):
        for (cont_chunk, cat_chunk), target_chunk in chunk_loader:
            # Fix the shapes from (1 x N x F) -> (N x F) and (1 x N) -> (N)
            (cont_chunk, cat_chunk), target_chunk = \
                (cont_chunk.view(-1, cont_chunk.shape[-1]),
                 cat_chunk.view(-1, cat_chunk.shape[-1])), \
                target_chunk.view(target_chunk.shape[-1])
            # Read batches from chunks and yield them
            chunk_size = target_chunk.shape[0]
            start_index = 0
            while start_index < chunk_size:
                end_index = min(start_index + batch_size, chunk_size)
                yield (cont_chunk[start_index:end_index, :],
                       cat_chunk[start_index:end_index, :]), \
                      target_chunk[start_index:end_index]
                start_index = end_index


def train_eval_split_hdf5(train_eval_path, train_path, eval_path, train_ratio=0.9, processed_size=50000):
    with pd.HDFStore(train_eval_path) as store:
        nb_samples = store.get_storer('data').nrows
    nb_train_samples = round(int(nb_samples) * train_ratio)
    chunk_iter = iter(pd.read_hdf(train_eval_path, chunksize=processed_size))
    with pd.HDFStore(train_path, mode='w') as train_f, pd.HDFStore(eval_path, mode='w') as eval_f:
        for chunk in chunk_iter:
            chunk = chunk.sample(frac=1)
            last_train_index = round(train_ratio * len(chunk))
            train_f.append('data', chunk.iloc[:last_train_index, :], format='table', expectedrows=nb_train_samples)
            eval_f.append('data', chunk.iloc[last_train_index:, :], format='table',
                          expectedrows=nb_samples - nb_train_samples)


class LGBMEvaluator:
    def __init__(self, X_train, X_test, y_train, y_test, cat_cols, prior_params, quantized_param_names,
                 invert_loss=False):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.cat_cols = cat_cols
        self.params = prior_params
        self.quantized_param_names = quantized_param_names
        self.invert_loss = invert_loss
        assert isinstance(prior_params, dict)
        self.best_model = None
        self.best_params = None
        self.best_loss = None
        self.iter_count = 0

    def __call__(self, params):
        d_train = lgb.Dataset(self.X_train, label=self.y_train, categorical_feature=self.cat_cols)
        d_test = lgb.Dataset(self.X_test, label=self.y_test, reference=d_train, categorical_feature=self.cat_cols)

        self.params.update(params)
        for p in self.quantized_param_names:
            self.params[p] = int(self.params[p])

        start = time.time()
        model = lgb.train(self.params, d_train, valid_sets=[d_test], verbose_eval=False)
        training_time = time.time() - start
        # pred_y_test = model.predict(self.X_test)
        # loss = -(self.y_test * np.log(pred_y_test) + (1 - self.y_test) * np.log(1 - pred_y_test)).mean()
        last_metric = next(iter(model.best_score['valid_0'].values()))
        loss = 1.0 - last_metric if self.invert_loss else last_metric

        if (self.best_loss is not None and loss < self.best_loss) or self.best_loss is None:
            self.best_model = model
            self.best_params = self.params.copy()
            self.best_loss = loss
        self.iter_count += 1
        return {'status': hyperopt.STATUS_OK, 'loss': loss, 'params': params, 'training_time': training_time,
                'iter_count': self.iter_count}
