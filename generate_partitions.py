import os
import json
import argparse

from dotenv import load_dotenv

from src.data.partitioning import download_data, generate_partition
from src.helper.data_partitioning_configuration import (
    DirichletPartitioning,
    ShardsPartitioning,
    FDPartitioning,
    IIDPartitioning
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["mnist", "cifar10", "cifar100", "cinic"],
                        help="Name of the dataset to be used")
    parser.add_argument("--n_clients", type=int, required=True,
                        help="Number of clients to be generated")
    parser.add_argument("--holdout_set_size", type=int, required=False, default=0,
                        help="Number of data point to exclude and keep as 'public dataset'")
    parser.add_argument("--seed", type=int, required=True,
                        help="Seed for reproducibility")
    parser.add_argument("--test_percentage", type=float, required=True,
                        help="Percentage of data that every clients reserves as test set")
    parser.add_argument("--val_percentage", type=float, required=True,
                        help="Percentage of data that every clients reserves as val set")
    parser.add_argument("--partitioning_method", type=str, required=True, choices=["dirichlet", "shard", "fd", "iid"],
                        help="Partitioning algorithm to be used. Use dirichlet with high alpha (100) for iid")
    parser.add_argument("--alpha", type=float, required=False,
                        help="Parameter for the dirichlet distribution")
    parser.add_argument("--min_size_of_dataset", type=int, required=False,
                        help="Parameter for the dirichlet distribution")
    parser.add_argument("--n_shards", type=int, required=False,
                        help="Number of classes each client should possess")
    parser.add_argument("--fixed_training_set_size", type=int, required=False, default=-1,
                        help="Fixed size of the training dataset")

    args = parser.parse_args()
    args = {k: v for k, v in vars(args).items() if v is not None}

    data_home_folder = os.environ.get("COLEXT_DATA_HOME_FOLDER")
    partitions_home_folder = "./data/partitions"
    assert os.path.isdir(data_home_folder), f"Folder {data_home_folder} does not exist"
    download_data(data_home_folder, args["dataset_name"])

    args["raw_data_folder"] = data_home_folder
    args["partitions_home_folder"] = partitions_home_folder
    partition_config_class = {
        "dirichlet": DirichletPartitioning,
        "shard": ShardsPartitioning,
        "fd": FDPartitioning,
        "iid": IIDPartitioning
    }[args["partitioning_method"]]
    partition_config = partition_config_class(**args)

    partition_folder = generate_partition(partition_config)
    print(f"Partitioning generated and saved in {partition_folder}")

    with open(f"{partition_folder}/generation_config.json", "w") as fp:
        json.dump(args, fp, indent=4)


if __name__ == "__main__":
    load_dotenv()
    main()
