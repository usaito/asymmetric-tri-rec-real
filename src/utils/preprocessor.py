"""
Codes for preprocessing datasets used in the real-world experiments
in the paper "Asymmetric Tri-training for Debiasing Missing-Not-At-Random Explicit Feedback".
"""

import codecs
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_datasets(data: str, seed: int = 0) -> Tuple:
    """Load and preprocess raw datasets (Yahoo! R3 or Coat)."""
    if data == 'yahoo':
        with codecs.open(f'../data/{data}/train.txt', 'r', 'utf-8', errors='ignore') as f:
            data_train = pd.read_csv(f, delimiter='\t', header=None)
            data_train.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
        with codecs.open(f'../data/{data}/test.txt', 'r', 'utf-8', errors='ignore') as f:
            data_test = pd.read_csv(f, delimiter='\t', header=None)
            data_test.rename(columns={0: 'user', 1: 'item', 2: 'rate'}, inplace=True)
        for _data in [data_train, data_test]:
            _data.user, _data.item = _data.user - 1, _data.item - 1
    elif data == 'coat':
        col = {'level_0': 'user', 'level_1': 'item', 2: 'rate', 0: 'rate'}
        with codecs.open(f'../data/{data}/train.ascii', 'r', 'utf-8', errors='ignore') as f:
            data_train = pd.read_csv(f, delimiter=' ', header=None)
            data_train = data_train.stack().reset_index().rename(columns=col)
            data_train = data_train[data_train.rate.values != 0].reset_index(drop=True)
        with codecs.open(f'../data/{data}/test.ascii', 'r', 'utf-8', errors='ignore') as f:
            data_test = pd.read_csv(f, delimiter=' ', header=None)
            data_test = data_test.stack().reset_index().rename(columns=col)
            data_test = data_test[data_test.rate.values != 0].reset_index(drop=True)

    test = data_test.values
    train, val = train_test_split(data_train.values, test_size=0.1, random_state=seed)
    num_users, num_items = train[:, 0].max() + 1, train[:, 1].max() + 1

    return train, val, test, num_users, num_items
