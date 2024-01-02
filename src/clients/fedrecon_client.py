from collections import OrderedDict

import torch
from torch.utils.data import random_split, DataLoader

from src.clients.plft import PLFT
from src.models.training_procedures import train_model_layers
from src.helper.optimization_config import OptimizationConfig
from src.models.evaluation_procedures import test_accuracy


class FedReconClient(PLFT):

    def __init__(self, support_set_perc, joint_train_params, reconstruction_config, **kwargs):
        super().__init__(**kwargs, stateful_client=False)
        self.reconstruction_config = reconstruction_config
        self.support_set_perc = support_set_perc
        self.joint_train_params = joint_train_params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def split_model_layer_names(self, model_names):
        return model_names[:-self.n_public_layers], model_names[-self.n_public_layers:]

    def reconstruct_model(self):
        support_dataloader, _ = \
            self._get_support_query_dataloaders(self.reconstruction_config.batch_size)
        private_model_dict, _ = self.split_model(self.model)
        private_model_keys = list(private_model_dict.keys())

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
            random_split(dataset, [support_set_size, query_set_size],
                         generator=torch.Generator().manual_seed(42))
        support_dataloader = DataLoader(support_dataset, batch_size=batch_size, shuffle=True)
        query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)
        return support_dataloader, query_dataloader

    def fit(self, parameters, config):
        _, query_dataloader = \
            self._get_support_query_dataloaders(self.reconstruction_config.batch_size)
        self.set_parameters(self.model, parameters)
        private_model_dict, public_model_dict = self.split_model(self.model)
        self.reconstruct_model()

        # train global parameters
        if self.joint_train_params:
            train_params = OrderedDict(
                list(private_model_dict.items()) +
                list(public_model_dict.items())
            )
        else:
            train_params = public_model_dict
        train_layers = list(train_params.keys())

        train_model_layers(self.get_optimization_config(query_dataloader, config),
                           train_layers=train_layers, gd_steps=None)

        return self.get_parameters({}), len(query_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(self.model, parameters)
        self.reconstruct_model()
        testloader = self._init_dataloader(train=False, batch_size=32)
        accuracy = test_accuracy(self.model, testloader, self.device)
        return accuracy, len(testloader.dataset), {"accuracy": accuracy, "client_id": self.cid}


def client_fn(cid, **kwargs) -> FedReconClient:
    return FedReconClient(cid=int(cid), **kwargs)
