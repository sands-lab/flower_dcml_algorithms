import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, images_folder, partition_csv, transforms=None, metadata=None):
        super().__init__()
        self.df = pd.read_csv(partition_csv)
        if metadata is not None:
            assert len(metadata) == self.df.shape[0]
            assert isinstance(metadata, np.ndarray)
            self.metadata = metadata
        else:
            self.metadata = None
        self.images_folder = images_folder
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target = row["label"]
        image_idx = row["idx"]
        image = Image.open(f"{self.images_folder}/{image_idx}.jpg")
        if self.transforms:
            image = self.transforms(image)
        if self.metadata is None:
            return image, target
        else:
            mtd = torch.from_numpy(self.metadata[idx])
            return image, target, mtd