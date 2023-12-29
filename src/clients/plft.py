from collections import OrderedDict

import torch

from src.clients.base_client import BaseClient
from src.models.helper import init_model
from src.models.training_procedures import train
from src.helper.commons import sync_rng_state

# pylint: disable=W0105
"""
Base class for Partially Local Federated Training algorithms
"""

class PLFT(BaseClient):

    def __init__(self, n_public_layers, **kwargs):
        self.n_public_layers = n_public_layers
        super().__init__(**kwargs, stateful_client=True)

    def split_model(self, model):
        """Return a tuple with private and public part of the model as OrderedDict"""
        raise NotImplementedError("The split model must be implemented")

    def _init_model(self):
        self.model = init_model(
            client_capacity=self.client_capacity,
            n_classes=self.n_classes,
            device=self.device,
            dataset=self.dataset_name
        )
        try:
            self.model.load_state_dict(torch.load(self.model_save_file), strict=False)
        except:
            private_model_dict, _ = self.split_model(self.model)
            torch.save(private_model_dict, self.model_save_file)

    def get_parameters(self, config):
        _, public_model = self.split_model(self.model)
        return [val.cpu().numpy() for _, val in public_model.items()]

    def set_parameters(self, model, parameters):
        _, public_model = self.split_model(model)
        params_dict = zip(public_model.keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=False)

    @sync_rng_state
    def fit(self, parameters, config):
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])
        self.set_parameters(self.model, parameters)
        train(
            self.get_optimization_config(trainloader, config)
        )
        self.save_model_to_disk()
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def save_model_to_disk(self):
        private_model_dict, _ = self.split_model(self.model)
        torch.save(private_model_dict, self.model_save_file)
