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

    def __init__(self, n_public_layers, stateful_client=True, **kwargs):
        self.n_public_layers = n_public_layers
        super().__init__(**kwargs, stateful_client=stateful_client, strict_load=False)

    def split_model_layer_names(self, model_names):
        """Return a tuple with private and public part of the model as OrderedDict"""
        raise NotImplementedError("The split model must be implemented")

    def extract_ordereddict(self, model, layer_names):
        return OrderedDict({k: v for k, v in model.state_dict().items()
                            if ".".join(k.split(".")[:-1]) in layer_names})

    def split_model(self, model: torch.nn.Module):
        model_layer_names = [".".join(l.split(".")[:-1]) for l, _ in model.named_parameters()]
        unique_layer_names, seen_layers = [], set()
        for mln in model_layer_names:
            if mln not in seen_layers:
                unique_layer_names.append(mln)
                seen_layers.add(mln)
        private_layer_names, public_layer_names = self.split_model_layer_names(unique_layer_names)
        print(private_layer_names, public_layer_names)
        private_model_dict = self.extract_ordereddict(model, private_layer_names)
        public_model_dict = self.extract_ordereddict(model, public_layer_names)
        return private_model_dict, public_model_dict

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
