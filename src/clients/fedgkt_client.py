import os

import numpy as np
import torch

from src.models.training_procedures import train_fedgkt_client
from src.models.resnet import Resnet8, Resnet55, ResnetCombined
from src.models.evaluation_procedures import test_accuracy
from src.clients.base_client import BaseClient


class FedGKTClient(BaseClient):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.model_save_file = f"{self.client_working_folder}/model.pth"
        if not os.path.isfile(self.model_save_file):
            torch.save(self.model.state_dict(), self.model_save_file)
        self.temperature = temperature

    def _init_model(self):
        model = Resnet8(10)
        model.to(self.device)
        return model

    def set_parameters(self, model, parameters):
        if parameters is None:
            model.load_state_dict(torch.load(self.model_save_file))
        else:
            super().set_parameters(model, parameters)

    # pylint: disable=C0103
    def fit(self, parameters, config):
        # here, parameters is a two-dimensional array that states the logits for every data point
        if len(parameters) == 0:
            parameters = None
        else:
            parameters = np.vstack(parameters)

        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"], metadata=parameters)

        self.set_parameters(self.model, None)

        H_k, Z_k, Y_k = train_fedgkt_client(
            optimization_config=self.get_optimization_config(trainloader, config),
            temperature=self.temperature
        )
        torch.save(self.model.state_dict(), self.model_save_file)
        return [H_k, Z_k, Y_k], len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        if len(parameters) == 0:
            # load client's model from disk
            return super().evaluate(parameters, config)
        else:
            server_model = Resnet55(10)
            self.set_parameters(server_model, parameters)
            testloader = self._init_dataloader(train=False, batch_size=32)
            resnet_combined = ResnetCombined(self.model, server_model)
            accuracy = test_accuracy(resnet_combined, testloader, self.device)
            print(f"Client accuracy: {accuracy}")
            return accuracy, len(testloader.dataset), {"accuracy": accuracy, "client_id": self.cid}

def client_fn(cid, temperature, **kwargs) -> FedGKTClient:
    return FedGKTClient(int(cid), temperature=temperature, **kwargs)

