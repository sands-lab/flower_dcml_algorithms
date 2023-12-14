import os
from logging import INFO

import torchvision.transforms as T
import flwr as fl
from flwr.common.logger import log
import torch
from torch.utils.data import DataLoader

from src.helper.parameters import get_parameters, set_parameters
from src.models.training_procedures import train
from src.models.evaluation_procedures import test_accuracy
from src.models.convnet import ConvNet
from src.data.cv_dataset import CustomDataset
from src.helper.commons import set_seed, sync_rng_state


class BaseClient(fl.client.NumPyClient):

    @sync_rng_state
    def __init__(self, idx, images_folder, partition_folder, seed, experiment_folder, model_name):
        super().__init__()
        self.idx = idx
        self.client_working_folder = f"{experiment_folder}/{idx}"
        self.dataset_name = os.path.split(images_folder)[-1]
        if not os.path.isdir(self.client_working_folder):
            os.mkdir(self.client_working_folder)
        set_seed(seed)
        log(INFO, f"Initializing client {idx}...")
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model_name = model_name
        self.n_classes = 10
        self.model = self._init_model()
        self.images_folder = images_folder
        self.partition_folder = partition_folder

    def get_parameters(self, config):
        return get_parameters(self.model)

    def set_parameters(self, model, parameters):
        set_parameters(model, parameters)

    @sync_rng_state
    def _init_model(self):
        model = {
            "convnet": ConvNet
        }[self.model_name](self.n_classes)
        model.to(self.device)
        return model

    def _init_dataloader(self, train, batch_size, metadata=None):
        if train:
            transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip()])
        else:
            transforms = T.ToTensor()

        partition = "train" if train else "test"
        partition_file = f"{self.partition_folder}/partition_{self.idx}_{partition}.csv"
        dataset = CustomDataset(self.images_folder,
                                partition_csv=partition_file,
                                transforms=transforms,
                                metadata=metadata)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train)  # if training shuffle the data
        return dataloader

    @sync_rng_state
    def evaluate(self, parameters, config):
        self.set_parameters(self.model, parameters)
        testloader = self._init_dataloader(train=False, batch_size=32)
        accuracy = test_accuracy(self.model, testloader, self.device)
        return accuracy, len(testloader.dataset), {"accuracy": accuracy, "client_id": self.idx}