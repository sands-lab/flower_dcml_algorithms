import os

import numpy as np
import flwr as fl
import torch

from src.models.training_procedures import train_fedgkt_client
from src.models.resnet import Resnet8
from src.clients.base_client import BaseClient


class FedGKTClient(BaseClient):

    def __init__(self, idx, images_folder, partition_folder, seed, experiment_folder, model_name):
        super().__init__(idx, images_folder, partition_folder, seed, experiment_folder, model_name)
        self.model_save_file = f"{self.client_working_folder}/model.pth"
        if not os.path.isfile(self.model_save_file):
            torch.save(self.model.state_dict(), self.model_save_file)

    def _init_model(self):
        model = Resnet8(10)
        model.to(self.device)
        return model

    def set_parameters(self, model, parameters):
        self.model.load_state_dict(torch.load(self.model_save_file))

    def fit(self, parameters, config):
        print(f"\n\nINSIDE FIT FUNCTION {parameters}\n\n")
        # here, parameters is a two-dimensional array that states the logits for every data point
        if len(parameters) == 0:
            parameters = None
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"], metadata=parameters)
        print("initialized dataloader")
        self.set_parameters(self.model, None)

        print(self.model)

        H_k, Z_k, Y_k = train_fedgkt_client(
            model=self.model,
            dataloader=trainloader,
            optimizer_name=config["optimizer"],
            epochs=config["local_epochs"],
            lr=config["lr"],
            device=self.device
        )
        torch.save(self.model.state_dict(), self.model_save_file)
        print(type(H_k), H_k.shape)
        print(type(Z_k), Z_k.shape)
        print(type(Y_k), Y_k.shape)
        return [H_k, Z_k, Y_k], len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        if parameters.size == 0:
            # load client's model from disk
            return super().evaluate(parameters, config)
        else:
            raise NotImplementedError("You are probably trying to pass the server-side model parameters to the client for evaluation. However, this feature is not supported yet")

def client_fn(cid, images_folder, partition_folder, seed, experiment_folder) -> FedGKTClient:
    return FedGKTClient(int(cid), images_folder, partition_folder, seed, experiment_folder, model_name=None)