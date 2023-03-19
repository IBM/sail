# Implementation of Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline (2016, arXiv) in PyTorch.
# The scripts are provided by https://github.com/aybchan/time-series-classification
import os
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sail.models.torch.data import Data


def download_datasets(datasets):
    """
    Download the time series datasets used in the paper
    from http://www.timeseriesclassification.com
    NonInvThorax1 and NonInvThorax2 are missing
    """
    subprocess.call("mkdir -p data".split())
    datasets_pbar = tqdm(datasets)
    print(datasets_pbar)
    for dataset in datasets_pbar:
        datasets_pbar.set_description("Downloading {0}".format(dataset))
        subprocess.call(
            "curl http://www.timeseriesclassification.com/Downloads/{0}.zip -o data/{0}.zip".format(
                dataset
            ).split()
        )
        datasets_pbar.set_description("Extracting {0} data".format(dataset))
        subprocess.call("unzip data/{0} -d data/".format(dataset).split())
        assert os.path.exists("data/{}_TRAIN.arff".format(dataset)), dataset


def data_dictionary(dataset_path, datasets):
    """
    Create a dictionary of train/test DataLoaders for
    each of the datasets downloaded
    """
    dataset_dict = {}
    pbar = tqdm(datasets)
    print(pbar)
    for dataset in pbar:
        pbar.set_description("Processing {}".format(dataset))
        train_set, test_set = Data(
            os.path.join(dataset_path, dataset), False
        ), Data(os.path.join(dataset_path, dataset), True)
        batch_size = min(16, len(train_set) // 10)

        dataset_dict[dataset] = {}
        dataset_dict[dataset]["train"] = DataLoader(
            train_set, batch_size=batch_size
        )
        dataset_dict[dataset]["test"] = DataLoader(
            test_set, batch_size=batch_size
        )

    return dataset_dict
