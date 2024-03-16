import os
import random
import tempfile
import itertools
from pathlib import Path

import torch
from torchvision import datasets as vision_datasets
from torch.utils.data import ConcatDataset
import numpy as np
import pandas as pd

from src.helper.data_partitioning_configuration import (
    PartitioningConfig,
    DirichletPartitioning,
    ShardsPartitioning,
    FDPartitioning,
    IIDPartitioning
)
from src.data.dataset_partition import DatasetPartition


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
        "cifar100": vision_datasets.CIFAR100,
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


def _save_configs(trainsets, valsets, testsets, save_folder):
    for idx, (partition_train_df, partition_val_df, partition_test_df) in enumerate(zip(trainsets, valsets, testsets)):
        for partition, df in [
            (DatasetPartition.TRAIN, partition_train_df),
            (DatasetPartition.VAL, partition_val_df),
            (DatasetPartition.TEST, partition_test_df)
        ]:
            save_file = f"{save_folder}/partition_{idx}_{partition.value}.csv"
            df.to_csv(save_file, index=False)


def _partition_client_dataset(client_partitions, partitioning_config):
    trainsets, valsets, testsets = [], [], []
    for partition_df in client_partitions.values():
        test_size = int(partition_df.shape[0] * partitioning_config.test_percentage)
        val_size = int(partition_df.shape[0] * partitioning_config.val_percentage)
        partition_test_df = partition_df.iloc[:test_size]
        ho_size = test_size + val_size
        partition_val_df = partition_df.iloc[test_size:ho_size]
        if partitioning_config.fixed_training_set_size > 0:
            partition_train_df = \
                partition_df.iloc[ho_size:ho_size+partitioning_config.fixed_training_set_size]
        else:
            partition_train_df = partition_df.iloc[ho_size:]
        trainsets.append(partition_train_df)
        valsets.append(partition_val_df)
        testsets.append(partition_test_df)
    return trainsets, valsets, testsets


def _generate_dirichlet_partition(partitioning_config: DirichletPartitioning, metadata_df):

    metadata_df["idx"] = metadata_df["filename"].str.split(".").str[0].astype(int)
    subsets = dirichlet_split(metadata_df["label"],
                              num_clients=partitioning_config.n_clients,
                              seed=partitioning_config.seed,
                              min_size_of_dataset=partitioning_config.min_size_of_dataset,
                              alpha_value=partitioning_config.alpha)
    subsets = {
        k: metadata_df.iloc[v] for k, v in subsets.items()
    }
    return _partition_client_dataset(subsets, partitioning_config)


def _generate_IID_partitions(partitioning_config: IIDPartitioning, metadata_df: pd.DataFrame):
    np.random.seed(partitioning_config.seed)
    random.seed(partitioning_config.seed)
    client_partitions = {k: [] for k in range(partitioning_config.n_clients)}

    for _, group_df in metadata_df.groupby("label"):
        subgroups = np.array_split(range(group_df.shape[0]), partitioning_config.n_clients)
        for i in range(partitioning_config.n_clients):
            client_partitions[i].append(group_df.iloc[subgroups[i]])
    for i in range(partitioning_config.n_clients):
        client_partitions[i] = pd.concat(client_partitions[i], ignore_index=True)
        client_partitions[i] = client_partitions[i].sample(frac=1.0).reset_index(drop=True)

    return _partition_client_dataset(client_partitions, partitioning_config)


def _generate_shards_partition(partitioning_config: ShardsPartitioning, metadata_df):

    shards = []
    shard_size = \
        int(metadata_df.shape[0] / partitioning_config.n_clients / partitioning_config.n_shards)

    sorted_target_idxs = np.argsort(metadata_df["label"].values)
    for i in range(partitioning_config.n_shards * partitioning_config.n_clients):
        shards.append(sorted_target_idxs[i * shard_size: (i + 1) * shard_size])
    idxs_list = torch.randperm(
        partitioning_config.n_clients * partitioning_config.n_shards,
        generator=torch.Generator().manual_seed(partitioning_config.seed)
    ).numpy()

    datasets = []
    for client_idx in range(partitioning_config.n_clients):
        client_shards = []
        for shard_idx in range(partitioning_config.n_shards):
            client_shards.append(
                shards[idxs_list[client_idx*partitioning_config.n_shards + shard_idx]]
            )
        datasets.append(np.hstack(client_shards))

    # randomly split in train and test set
    trainsets, testsets = [], []
    testset_size = int(shard_size * partitioning_config.test_percentage)
    for i, ds in enumerate(datasets):
        permuted_idxs = ds[torch.randperm(
            len(ds),
            generator=torch.Generator().manual_seed(i + partitioning_config.seed)
        ).numpy()]
        testsets.append(metadata_df.iloc[permuted_idxs[:testset_size]])
        trainsets.append(metadata_df.iloc[permuted_idxs[testset_size:]])
    return trainsets, testsets


def _generate_partitions_as_FD(partitioning_config: FDPartitioning, metadata_df):

    partition_sizes = [2000] * partitioning_config.n_clients

    # shuffle rows of the dataframe
    metadata_df = metadata_df \
        .sample(frac=1.0, replace=False, random_state=partitioning_config.seed) \
        .reset_index(drop=True)

    partitions = []
    start_idx = 0
    for size in partition_sizes:
        partitions.append(
            metadata_df.iloc[start_idx : start_idx+size].reset_index(drop=True).copy()
        )
        start_idx += size

    reduced_partitions = {"train": [], "test": []}
    np.random.seed(partitioning_config.seed)

    for df in partitions:
        # sample a target label and retain only 5 values
        target_label = np.random.randint(0, 10)
        label_idx = df[df["label"] == target_label].index.to_list()
        drop = np.random.choice(label_idx, len(label_idx) - 5, replace=False)
        reduced = df.drop(index=drop).reset_index(drop=True).copy()
        assert (reduced["label"] == target_label).sum() == 5

        test_idxs = np.random.choice(
            reduced.shape[0],
            size=int(reduced.shape[0] * partitioning_config.test_percentage),
            replace=False
        )
        reduced_partitions["test"].append(reduced.loc[test_idxs].reset_index(drop=True))
        reduced_partitions["train"].append(reduced.drop(index=test_idxs).reset_index(drop=True))

    return reduced_partitions["train"], reduced_partitions["test"]


def generate_partition(partitioning_config: PartitioningConfig):
    partition_folder = partitioning_config.get_partition_folder()
    if os.path.exists(partition_folder):
        print("Exists aready. Returning...")
        return partition_folder
    Path(partition_folder).mkdir(parents=True, exist_ok=False)
    metadata_df = pd.read_csv(os.path.join(
        partitioning_config.raw_data_folder,
        partitioning_config.dataset_name,
        "metadata.csv")
    )

    public_metadata_df = metadata_df.sample(n=partitioning_config.holdout_set_size)
    metadata_df = metadata_df.drop(public_metadata_df.index).reset_index(drop=True)
    public_metadata_df.to_csv(f"{partition_folder}/public_dataset.csv", index=False)

    partitioning_fn = {
        "dirichlet": _generate_dirichlet_partition,
        "shard": _generate_shards_partition,
        "fd": _generate_partitions_as_FD,
        "iid": _generate_IID_partitions,
    }[partitioning_config.partitioning_method]
    trainsets, valsets, testsets = partitioning_fn(partitioning_config, metadata_df)
    _save_configs(trainsets, valsets, testsets, partition_folder)

    return partition_folder
