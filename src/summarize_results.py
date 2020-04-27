"""
Code for summarizing experimental results for the paper
"Asymmetric Tri-training for Debiasing Missing-Not-At-Random Rating Feedback".
"""
import argparse
import yaml

from utils.preprocessor import preprocess_datasets
from utils.result_tools import (summarize_data_statistics, summarize_experimental_results)

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', '-d', type=str, nargs='*', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    num_sims = yaml.safe_load(open('../config.yaml', 'rb'))['num_sims']

    summarize_data_statistics()
    for data in args.datasets:
        train, _, test, _, _ = preprocess_datasets(data=data)
        summarize_experimental_results(data=data)
