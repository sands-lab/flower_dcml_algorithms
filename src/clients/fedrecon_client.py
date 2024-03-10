from collections import OrderedDict

import torch
from torch.utils.data import random_split, DataLoader

from src.clients.lg_fedavg_client import LgFedAvgClient
from src.models.training_procedures import train_model_layers
from src.helper.optimization_config import OptimizationConfig
from src.helper.parameters import get_parameters, set_parameters
from src.models.evaluation_procedures import test_accuracy


class FedReconClient(LgFedAvgClient):

    def __init__(self, support_set_perc, reconstruction_config, **kwargs):
        super().__init__(**kwargs, stateful_client=False)
        self.reconstruction_config = reconstruction_config
        self.support_set_perc = support_set_perc

    def set_parameters(self, model, parameters):
        set_parameters(model[1], parameters)
        self.reconstruct_model()

    def reconstruct_model(self):
        support_dataloader, _ = \
            self._get_support_query_dataloaders(self.reconstruction_config.batch_size)
        private_model_keys = set(self.model[0].state_dict().keys())

        # reconstruct the model parameters
        optimization_config = OptimizationConfig(
            model=self.model,
            dataloader=support_dataloader,
            lr=self.reconstruction_config.lr,
            epochs=2,
            optimizer_name=self.reconstruction_config.optimizer_name,
            device=self.device
        )
        train_model_layers(optimization_config, train_layers=private_model_keys,
                           gd_steps=self.reconstruction_config.gd_steps)

    def _get_support_query_dataloaders(self, batch_size):
        dataset = self._init_dataset(True, None)

        support_set_size = int(self.support_set_perc * len(dataset))
        query_set_size = len(dataset) - support_set_size
        support_dataset, query_dataset = \
            random_split(dataset, [support_set_size, query_set_size])
        support_dataloader = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)
        query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)
        return support_dataloader, query_dataloader

    def fit(self, parameters, config):
        self.set_parameters(self.model, parameters)

        _, query_dataloader = \
            self._get_support_query_dataloaders(config["batch_size"])

        # train global parameters
        train_layers = set(self.model.state_dict().keys())
        train_model_layers(self.get_optimization_config(query_dataloader, config),
                           train_layers=train_layers, gd_steps=None)

        return self.get_parameters({}), len(query_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(self.model, parameters)
        return self._evaluate()


def client_fn(cid, **kwargs) -> FedReconClient:
    return FedReconClient(cid=int(cid), **kwargs)
