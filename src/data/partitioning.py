import os
import tempfile
import itertools
from pathlib import Path

import torch
from torchvision import datasets as vision_datasets
from torch.utils.data import ConcatDataset
import numpy as np
import pandas as pd


def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    np.random.shuffle(idx_k)
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    proportions = np.array(
        [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    idx_batch = [
        idx_j + idx.tolist()
        for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min(len(idx_j) for idx_j in idx_batch)

    return idx_batch, min_size


def dirichlet_split(
        labels, num_clients, seed, min_size_of_dataset, alpha_value=0.5
):
    np.random.seed(seed)
    net_dataidx_map = {}

    idx_batch = [[] for _ in range(num_clients)]
    N = len(labels)

    min_size = 0
    while min_size < min_size_of_dataset:
        idx_batch = [[] for _ in range(num_clients)]
        for k in np.unique(labels):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(labels == k)[0]
            idx_batch, min_size = partition_class_samples_with_dirichlet_distribution(
                N, alpha_value, num_clients, idx_batch, idx_k
            )

    for i in range(num_clients):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    assert len(labels) == sum(len(indices) for _, indices in net_dataidx_map.items()), \
        f"Some samples were not assigned to a client." \
        f" {len(labels)} != {sum(len(indices) for _, indices in net_dataidx_map.items())}"
    # assert that there is no intersection between clients indices!
    assert all((set(p0).isdisjoint(set(p1))) for p0, p1 in
               itertools.combinations([indices for _, indices in net_dataidx_map.items()], 2))

    return net_dataidx_map



def download_data(data_home_folder, dataset_name):
    interim_data_folder = os.path.join(data_home_folder, dataset_name)
    if os.path.exists(interim_data_folder):
        print("Data downloaded already. Skipping...")
        return

    dataset_class = {
        "cifar10": vision_datasets.CIFAR10,
        "mnist": vision_datasets.MNIST
    }[dataset_name]

    # laod the data
    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        train_dataset = dataset_class(temp_dir, True, None, download=True)
        test_dataset = dataset_class(temp_dir, False, None, download=True)
        dataset = ConcatDataset([train_dataset, test_dataset])

        targets = []

        Path(interim_data_folder).mkdir(parents=True, exist_ok=False)

        for idx, (image, target) in enumerate(dataset):
            file_name = f"{idx}.jpg"
            image_file = os.path.join(interim_data_folder, file_name)
            image.save(image_file)
            targets.append((file_name,target))

    df = pd.DataFrame(targets, columns=["filename", "label"])
    df.to_csv(f"{interim_data_folder}/metadata.csv", index=False)


def _save_configs(trainsets, testsets, save_folder):
    for idx, (partition_train_df, partition_test_df) in enumerate(zip(trainsets, testsets)):
        save_train_file = f"{save_folder}/partition_{idx}_train.csv"
        save_test_file = f"{save_folder}/partition_{idx}_test.csv"
        partition_train_df.to_csv(save_train_file, index=False)
        partition_test_df.to_csv(save_test_file, index=False)


def _generate_dirichlet_partition(raw_data_folder, partitions_home_folder, dataset_name,
                                  num_clients, seed, test_percentage, min_size_of_dataset, alpha):

    df = pd.read_csv(f"{raw_data_folder}/{dataset_name}/metadata.csv")
    df["idx"] = df["filename"].str.split(".").str[0].astype(int)
    subsets = dirichlet_split(df["label"],
                              num_clients=num_clients,
                              seed=seed,
                              min_size_of_dataset=min_size_of_dataset,
                              alpha_value=alpha)
    partition_folder = f"{partitions_home_folder}/{dataset_name}/"\
                       f"dirichlet_{num_clients}clients_{seed}seed"\
                       f"_{alpha}alpha_{test_percentage}test"
    if os.path.exists(partition_folder):
        print("Partitioning exists already. Returning...")
        return partition_folder
    Path(partition_folder).mkdir(parents=True, exist_ok=False)

    trainsets, testsets = [], []
    for subset in subsets.values():
        partition_df = df[df["idx"].isin(subset)]
        test_size = int(partition_df.shape[0] * test_percentage)
        partition_test_df = partition_df.iloc[:test_size]
        partition_train_df = partition_df.iloc[test_size:]

        trainsets.append(partition_train_df)
        testsets.append(partition_test_df)
    _save_configs(trainsets, testsets, partition_folder)
    return partition_folder


def _generate_shards_partition(raw_data_folder, partitions_home_folder,
                               dataset_name, num_clients, test_percentage, seed):
    df = pd.read_csv(f"{raw_data_folder}/{dataset_name}/metadata.csv")

    shards = []
    shard_size = int(df.shape[0] / num_clients / 2)

    partition_folder = f"{partitions_home_folder}/{dataset_name}/" \
                       f"shard_{num_clients}clients_{seed}seed_{test_percentage}test"
    if os.path.exists(partition_folder):
        print("Partitioning exists already. Returning...")
        return partition_folder
    Path(partition_folder).mkdir(parents=True, exist_ok=False)

    sorted_target_idxs = np.argsort(df["label"].values)
    for i in range(2 * num_clients):
        shards.append(sorted_target_idxs[i * shard_size: (i + 1) * shard_size])
    idxs_list = torch.randperm(
        num_clients * 2, generator=torch.Generator().manual_seed(seed)
    ).numpy()
    datasets = [
        np.hstack([shards[idxs_list[i*2]], shards[idxs_list[i*2+1]]]) for i in range(0, num_clients)
    ]

    # randomly split in train and test set
    trainsets, testsets = [], []
    testset_size = int(shard_size * test_percentage)
    for i, ds in enumerate(datasets):
        permuted_idxs = \
            ds[torch.randperm(len(ds), generator=torch.Generator().manual_seed(i + seed)).numpy()]
        testsets.append(df.iloc[permuted_idxs[:testset_size]])
        trainsets.append(df.iloc[permuted_idxs[testset_size:]])
    _save_configs(trainsets, testsets, partition_folder)
    return partition_folder


def generate_partition(partition_method, **kwargs):
    return {
        "dirichlet": _generate_dirichlet_partition,
        "shard": _generate_shards_partition
    }[partition_method](**kwargs)
