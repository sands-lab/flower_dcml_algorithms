import os
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
from src.helper.commons import read_json
from src.data.cv_dataset import CustomDataset


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_folder", required=True, type=str)
    parser.add_argument("--filter_client", required=False, type=int)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--lr", required=False, type=float, default=0.05)
    args = parser.parse_args()

    data_home_folder = os.environ.get("COLEXT_DATA_HOME_FOLDER")

    dataset_name = os.path.normpath(args.partition_folder).split(os.sep)[-2]
    if args.filter_client is None:
        train_csvs = glob(f"{args.partition_folder}/partition**_train.csv")
        test_csvs = glob(f"{args.partition_folder}/partition**_test.csv")
    else:
        train_csvs = [f"{args.partition_folder}/partition_{args.filter_client}_train.csv"]
        test_csvs = [f"{args.partition_folder}/partition_{args.filter_client}_test.csv"]

    norm_params = read_json("config/data/data_configuration.json", [dataset_name, "normalization_parameters"])
    transforms = [
        T.ToTensor(),
        T.Normalize(mean=norm_params["mean"], std=norm_params["std"])
    ]
    print(transforms)
    trainset = CustomDataset(f"{data_home_folder}/{dataset_name}", train_csvs, T.Compose(transforms + [T.RandomHorizontalFlip()]))
    testset = CustomDataset(f"{data_home_folder}/{dataset_name}", test_csvs, T.Compose(transforms))

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, pin_memory=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, pin_memory=True, num_workers=2)

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    model = init_model_from_string(args.model, 10, 1.0, device)
    optim_config = \
        OptimizationConfig(model, trainloader, args.lr, 1, "sgd", device)

    loop = tqdm(range(args.epochs))
    accs = []
    patience = 50
    for epoch in loop:
        train(optim_config)
        acc = test_accuracy(model, testloader, device)
        loop.set_postfix({"accuracy": acc})
        accs.append(acc)
        if len(accs) > 2 * patience and np.max(accs[-2 * patience:-patience]) > np.max(accs[-patience:]):
            print("Converged")
            break
    print(f"Final acc: {np.mean(accs[-20])}")


if __name__ == "__main__":
    run()
