import os
import copy
import json
from glob import glob
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T

from src.models.training_procedures import train
from src.models.evaluation_procedures import test_accuracy
from src.models.helper import init_model_from_string
from src.helper.optimization_config import OptimizationConfig
from src.helper.commons import read_json, set_seed
from src.data.cv_dataset import CustomDataset
from src.helper.environment_variables import EnvironmentVariables as EV


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_folder", required=True, type=str)
    parser.add_argument("--filter_client", required=False, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--lr", required=False, type=float, default=0.05)
    parser.add_argument("--seed", required=False, type=int, default=1)
    parser.add_argument("--patience", required=False, type=float, default=10)
    args = parser.parse_args()

    data_home_folder = os.environ.get(EV.DATA_HOME_FOLDER)

    dataset_name = os.path.normpath(args.partition_folder).split(os.sep)[-2]
    dataset_config = read_json("config/data/data_configuration.json", [dataset_name])
    norm_params = dataset_config["normalization_parameters"]
    transforms = [
        T.ToTensor(),
        T.Normalize(mean=norm_params["mean"], std=norm_params["std"])
    ]

    dataloaders = {}
    for partition in ["train", "test", "val"]:

        if args.filter_client is None:
            csvs = glob(f"{args.partition_folder}/partition**_{partition}.csv")
        else:
            csvs = [f"{args.partition_folder}/partition_{args.filter_client}_{partition}.csv"]

        if partition == "train":
            transforms_ = transforms + [T.RandomHorizontalFlip()]
        else:
            transforms_ = transforms

        dataset = CustomDataset(f"{data_home_folder}/{dataset_name}", csvs, T.Compose(transforms_))
        print(partition, len(dataset), transforms_)
        dataloaders[partition] = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    set_seed(args.seed * 10 + 19)
    model = init_model_from_string(args.model, dataset_config["n_classes"], 1.0, device)
    optim_config = \
        OptimizationConfig(model, dataloaders["train"], args.lr, 1, "sgd", device)

    loop = tqdm(range(args.epochs))
    best_acc, no_improvement_idx, best_model = -1, 0, None

    for _ in loop:
        train(optim_config)
        acc = test_accuracy(model, dataloaders["val"], device)
        loop.set_postfix({"acc": acc})
        if acc > best_acc:
            best_acc = acc
            no_improvement_idx = 0
            best_model = copy.deepcopy(model)

        else:
            no_improvement_idx += 1
        if no_improvement_idx > args.patience:
            print("Converged")
            break

    final_acc = test_accuracy(best_model, dataloaders["test"], device)
    print(f"Final acc {args.partition_folder} {args.filter_client}: {final_acc}")
    with open("data/interim/test_centralized.json", "r") as fp:
        accs = json.load(fp)
    if dataset_name not in accs:
        accs[dataset_name] = {}
        accs[dataset_name]["clients"] = []
        accs[dataset_name]["centralized"] = []

    if args.filter_client is None:
        accs[dataset_name]["centralized"].append(final_acc)
    else:
        accs[dataset_name]["clients"].append(final_acc)

    with open("data/interim/test_centralized.json", "w") as fp:
        json.dump(accs, fp, indent=4)



if __name__ == "__main__":
    run()
