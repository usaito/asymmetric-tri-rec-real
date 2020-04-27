"""Tools for summarizing experimental results."""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from utils.preprocessor import preprocess_datasets


datasets = ['yahoo', 'coat']
metrics = ['mae (w/o at)', 'mae (w/ at)', 'mse (w/o at)', 'mse (w/ at)', 'ndcg (w/o at)', 'ndcg (w/ at)']
model_names = ['uniform', 'uniform-at', 'user', 'user-at', 'item', 'item-at',
               'both', 'both-at', 'nb', 'nb-at', 'nb_true', 'nb_true-at']
stats_idx = ['#User', '#Item', '#Rating', 'Sparsity',
             'Avg. rating of train', 'Avg. rating of test', 'KL div']


def calc_kl_div(train: np.ndarray, test: np.ndarray) -> float:
    """Estimate KL divergence of rating distributions between training and test sets."""
    p = np.unique(train[:, 2], return_counts=True)[1] / \
        np.unique(train[:, 2], return_counts=True)[1].sum()
    q = np.unique(test[:, 2], return_counts=True)[1] / \
        np.unique(test[:, 2], return_counts=True)[1].sum()
    return np.round(np.sum(np.where(p != 0, p * np.log(p / q), 0)), 4)


def summarize_data_statistics() -> None:
    """Save dataset statistics with Tex Table Format."""
    stat_data_list = []
    Path('../paper_results').mkdir(exist_ok=True)
    for data in datasets:
        train, _, test, num_users, num_items = preprocess_datasets(data=data)
        num_data = train.shape[0]
        spasity = f'{100 * (num_data / (num_users * num_items)).round(4)}%'
        avg_train, avg_test = train[:, 2].mean().round(3), test[:, 2].mean().round(3)
        kl = calc_kl_div(train, test)
        stat_data = DataFrame(data=[num_users, num_items, num_data, spasity, avg_train, avg_test, kl],
                              index=stats_idx, columns=[data]).T
        stat_data_list.append(stat_data)
    pd.concat(stat_data_list).to_csv('../paper_results/data_stat.csv', sep='&')


model_names = ['uniform', 'uniform-at', 'user', 'user-at', 'item', 'item-at',
               'both', 'both-at', 'nb', 'nb-at', 'nb_true', 'nb_true-at']


def summarize_experimental_results(data: str) -> None:
    """Summarize results with Tex Table format."""
    raw_results_path = Path(f'../logs/{data}')
    paper_results_path = Path(f'../paper_results/{data}')
    paper_results_path.mkdir(exist_ok=True, parents=True)
    results_mse_dict = {}
    results_mae_dict = {}
    results_ndcg_dict = {}
    results_mse_dict_at = {}
    results_mae_dict_at = {}
    results_ndcg_dict_at = {}

    for model_name in model_names:
        results_ = pd.read_csv(str(raw_results_path / f'{model_name}/results.csv'), index_col=0)
        if '-at' in model_name:
            results_mse_dict_at[model_name[:-3]] = results_['MSE']
            results_mae_dict_at[model_name[:-3]] = results_['MAE']
            results_ndcg_dict_at[model_name[:-3]] = results_['nDCG@3']
        else:
            results_mse_dict[model_name] = results_['MSE']
            results_mae_dict[model_name] = results_['MAE']
            results_ndcg_dict[model_name] = results_['nDCG@3']
    results_mae = DataFrame(results_mae_dict).describe().round(5).T
    results_mse = DataFrame(results_mse_dict).describe().round(5).T
    results_ndcg = DataFrame(results_ndcg_dict).describe().round(5).T
    results_mae_at = DataFrame(results_mae_dict_at).describe().round(5).T
    results_mse_at = DataFrame(results_mse_dict_at).describe().round(5).T
    results_ndcg_at = DataFrame(results_ndcg_dict_at).describe().round(5).T
    results_list = [results_mae, results_mae_at, results_mse, results_mse_at, results_ndcg, results_ndcg_at]
    results_dict = {}
    for results, metric in zip(results_list, metrics):
        results_dict[metric] = results['mean']
    DataFrame(results_dict).to_csv(str(paper_results_path / 'overall_results.csv'))
