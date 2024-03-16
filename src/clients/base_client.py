import os
from logging import INFO

import flwr as fl
from flwr.common.logger import log
import torch
from torch.utils.data import DataLoader

from src.helper.parameters import get_parameters, set_parameters
from src.helper.commons import set_seed, save_rng_state_if_not_exists, sync_rng_state
from src.helper.optimization_config import OptimizationConfig
from src.data.helper import init_dataset
from src.models.evaluation_procedures import test_accuracy
from src.models.helper import init_model
from src.data.dataset_partition import DatasetPartition


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
            strict_load=True,
            separate_val_test_sets = False
    ):
        super().__init__()
        self.cid = cid
        self.seed = seed
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
        self.n_classes = 100 if self.dataset_name == "cifar100" else 10
        self.client_capacity = client_capacity
        self.images_folder = images_folder
        self.partition_folder = partition_folder
        self._init_model()
        self.separate_val_test_sets = separate_val_test_sets
        self.measure_accuracy_on_public_dataset = False  # only for reproducibility purposes
        save_rng_state_if_not_exists(self.client_working_folder)
        log(INFO, "Initialed client %s [model %s %d] on %s...", cid,
            self.model.__class__.__name__, self.get_model_size(), self.device)

    def get_model_size(self):
        return sum(p.numel() for p in self.model.parameters())

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

    def _init_dataloader(self, dataset_partition, batch_size):
        dataset = self._init_dataset(dataset_partition)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=dataset_partition == DatasetPartition.TRAIN,
            pin_memory=self.device.type == "cuda",
            num_workers=2
        )
        return dataloader

    def _init_dataset(self, dataset_partition):
        return init_dataset(
            cid=self.cid,
            dataset_partition=dataset_partition,
            dataset_name=self.dataset_name,
            partition_folder=self.partition_folder,
            images_folder=self.images_folder
        )

    @sync_rng_state
    def evaluate(self, parameters, config):
        if not self.stateful_client:
            self.set_parameters(self.model, parameters)
        return self._evaluate()

    def get_accuracy_and_dataset_size(self, partition):
        dataloader = self._init_dataloader(partition, batch_size=32)
        return test_accuracy(self.model, dataloader, self.device), len(dataloader.dataset)

    def _evaluate(self):
        dataset_size = None
        out_dict = {"client_id": self.cid, "client_capacity": self.client_capacity}

        if self.separate_val_test_sets:
            # compute accuracy on both validation and test datasets
            out_dict["accuracy"], dataset_size = \
                self.get_accuracy_and_dataset_size(DatasetPartition.VAL)
            out_dict["test_accuracy"] = \
                self.get_accuracy_and_dataset_size(DatasetPartition.TEST)[0]
        else:
            # compute the accuracy only on the test set
            out_dict["accuracy"], dataset_size = \
                self.get_accuracy_and_dataset_size(DatasetPartition.TEST)

        return out_dict["accuracy"], dataset_size, out_dict


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

    def get_properties(self, config):
        return {
            "client_capacity": self.client_capacity
        }
