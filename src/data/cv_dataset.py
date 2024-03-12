import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CustomDataset(Dataset):
    def __init__(self, images_folder, partition_csv, transforms=None):
        super().__init__()
        if isinstance(partition_csv, list):
            self.df = pd.concat([pd.read_csv(file) for file in partition_csv])
        else:
            self.df = pd.read_csv(partition_csv)
        self.images_folder = images_folder
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row["label"]
        image_file = row["filename"]
        image = Image.open(f"{self.images_folder}/{image_file}")
        if self.transforms:
            image = self.transforms(image)
        return image, target


class UnlabeledDataset(Dataset):
    def __init__(self, dataset_name, dataset_size) -> None:
        super().__init__()
        data_home_folder = os.environ.get("COLEXT_DATA_HOME_FOLDER")
        dataset_home_folder = f"{data_home_folder}/{dataset_name}"
        self.filepaths = pd.read_csv(f"{dataset_home_folder}/metadata.csv")["filename"]\
            .sample(dataset_size, replace=False).to_list()
        self.dataset_home_folder = dataset_home_folder
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)),
            T.RandomHorizontalFlip()
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(f"{self.dataset_home_folder}/{self.filepaths[idx]}")
        image = self.transforms(image)
        return image
