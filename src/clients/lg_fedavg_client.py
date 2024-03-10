import json

import torch
import torch.nn as nn

from src.clients.base_client import BaseClient
from src.models.helper import init_model_from_string
from src.models.training_procedures import train
from src.models.evaluation_procedures import test_accuracy
from src.helper.parameters import get_parameters, set_parameters
from src.helper.commons import sync_rng_state
from src.data.dataset_partition import DatasetPartition


class LgFedAvgClient(BaseClient):

    def __init__(self, stateful_client=True, **kwargs):
        super().__init__(**kwargs, stateful_client=stateful_client)

    def _init_model(self):
        with open("config/models/plft_model_config.json", "r") as fp:
            model_config = json.load(fp)[self.dataset_name]["lg_fedavg"]
        common_head = init_model_from_string(model_config["common_head"], None, None, self.device)
        encoder = init_model_from_string(model_config["encoders"][str(self.client_capacity)], None, None, self.device)
        print(f"Initialized model with encoder {encoder.__class__.__name__}")
        self.model = nn.Sequential(encoder, common_head)

    def get_parameters(self, config):
        return get_parameters(self.model[1])

    def set_parameters(self, model, parameters):
        set_parameters(model[1], parameters)
        try:
            load_dict = torch.load(self.model_save_file)
            self.model[0].load_state_dict(load_dict, strict=True)
        except:
            print("No model found. Saving it to disk...")
            self.save_model_to_disk()  # actually, we might not need this

    def save_model_to_disk(self):
        torch.save(self.model[0].state_dict(), self.model_save_file)

    @sync_rng_state
    def fit(self, parameters, config):
        trainloader = self._init_dataloader(
            dataset_partition=DatasetPartition.TRAIN,
            batch_size=config["batch_size"]
        )
        self.set_parameters(self.model, parameters)
        train(
            self.get_optimization_config(trainloader, config)
        )
        self.save_model_to_disk()
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    @sync_rng_state
    def evaluate(self, parameters, config):
        self.set_parameters(self.model, parameters)
        return self._evaluate()


def client_fn(cid, **kwargs) -> LgFedAvgClient:
    return LgFedAvgClient(cid=int(cid), **kwargs)
