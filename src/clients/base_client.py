import os
import time
from logging import INFO

import flwr as fl
from flwr.common.logger import log
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.helper.parameters import get_parameters, set_parameters
from src.helper.commons import set_seed, save_rng_state_if_not_exists, sync_rng_state
from src.helper.optimization_config import OptimizationConfig
from src.models.evaluation_procedures import test_accuracy
from src.models.helper import init_model
from src.data.cv_dataset import CustomDataset


class BaseClient(fl.client.NumPyClient):

    def __init__(
            self,
            cid,
            images_folder,
            partition_folder,
            seed,
            experiment_folder,
            client_capacity,
            stateful_client,
            strict_load=True
    ):
        super().__init__()
        self.cid = cid
        self.strict_load = strict_load
        self.stateful_client = stateful_client
        self.client_working_folder = f"{experiment_folder}/{cid}"
        self.dataset_name = os.path.split(images_folder)[-1]
        self.model_save_file = \
            f"{self.client_working_folder}/model.pth" if stateful_client else None
        if not os.path.isdir(self.client_working_folder):
            os.mkdir(self.client_working_folder)
        set_seed(seed)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.n_classes = 10
        self.client_capacity = client_capacity
        self.images_folder = images_folder
        self.partition_folder = partition_folder
        self._init_model()
        self.measure_accuracy_on_public_dataset = False  # only for reproducibility purposes
        save_rng_state_if_not_exists(self.client_working_folder)
        log(INFO, "Initialed client %s [model %s] on %s...", cid,
            self.model.__class__.__name__, self.device)

    def get_parameters(self, config):
        assert not self.stateful_client, "Should not call get_parameters on stateful clients!!"
        return get_parameters(self.model)

    def set_parameters(self, model, parameters):
        if not self.stateful_client:
            set_parameters(model, parameters)

    def _init_model(self):
        self.model = init_model(
            client_capacity=self.client_capacity,
            n_classes=self.n_classes,
            device=self.device,
            dataset=self.dataset_name
        )

        if self.stateful_client:
            try:
                load_dict = torch.load(self.model_save_file)
                self.model.load_state_dict(load_dict, strict=self.strict_load)
            except:
                self.save_model_to_disk()  # actually, we might not need this

    def _init_dataloader(self, train, batch_size, metadata=None):
        dataset = self._init_dataset(train, metadata)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train, pin_memory=True)
        return dataloader

    def _init_dataset(self, train, metadata):
        transforms = [T.ToTensor()]
        if self.dataset_name == "cifar10":
            transforms.append(T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261)))
        elif self.dataset_name == "mnist":
            transforms.append(T.Normalize((0.1307,), (0.3081,)))
        elif self.dataset_name == "cinic":
            transforms.append(T.Normalize(mean=(0.4788, 0.4722, 0.4304), std=(0.242, 0.238, 0.259)))
        else:
            raise Exception()
        if train and self.dataset_name not in {"mnist"}:
            transforms.append(T.RandomHorizontalFlip())
        transforms = T.Compose(transforms)

        partition = "train" if train else "test"
        if not train and self.measure_accuracy_on_public_dataset:
            filename = "public_dataset.csv"
        else:
            filename = f"partition_{self.cid}_{partition}.csv"
        partition_file = f"{self.partition_folder}/{filename}"
        dataset = CustomDataset(self.images_folder,
                                partition_csv=partition_file,
                                transforms=transforms,
                                metadata=metadata)
        return dataset

    @sync_rng_state
    def evaluate(self, parameters, config):
        if not self.stateful_client:
            self.set_parameters(self.model, parameters)
        testloader = self._init_dataloader(train=False, batch_size=32)
        accuracy = test_accuracy(self.model, testloader, self.device)
        return accuracy, len(testloader.dataset), {"accuracy": accuracy, "client_id": self.cid}

    def save_model_to_disk(self):
        assert self.stateful_client, \
            "Saving model to disk is possible only if using stateful clients!!"
        torch.save(self.model.state_dict(), self.model_save_file)

    def get_optimization_config(self, trainloader, config, **kwargs):
        return OptimizationConfig(
            model=self.model,
            dataloader=trainloader,
            optimizer_name=config["optimizer"],
            epochs=config["local_epochs"],
            lr=config["lr"],
            device=self.device,
            **kwargs
        )
