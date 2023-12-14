import os

import flwr as fl
import torch

from src.models.training_procedures import train_fd
from src.clients.base_client import BaseClient


class FDClient(BaseClient):

    def __init__(self, idx, images_folder, partition_folder, seed, experiment_folder):
        super().__init__(idx, images_folder, partition_folder, seed, experiment_folder)
        self.model_save_file = f"{self.client_working_folder}/model.pth"
        if not os.path.isfile(self.model_save_file):
            torch.save(self.model.state_dict(), self.model_save_file)

    def set_parameters(self, model, parameters):
        self.model.load_state_dict(torch.load(self.model_save_file))

    def fit(self, parameters, config):
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])
        self.set_parameters(self.model, None)

        logits_matrix = train_fd(
            model=self.model,
            dataloader=trainloader,
            optimizer_name=config["optimizer"],
            epochs=config["local_epochs"],
            lr=config["lr"],
            device=self.device
        )
        torch.save(self.model.state_dict(), self.model_save_file)
        return fl.common.ndarrays_to_parameters(logits_matrix), len(trainloader.dataset), {}

def client_fn(cid, images_folder, partition_folder) -> FDClient:
    return FDClient(int(cid), images_folder, partition_folder)