import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.training_procedures import train
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state


class FedMDClient(BaseClient):

    def __init__(self, pretrain_epochs, kd_epochs, **kwargs):
        super().__init__(**kwargs)
        self.model_save_file = f"{self.client_working_folder}/model.pth"
        if not os.path.isfile(self.model_save_file):
            torch.save(self.model.state_dict(), self.model_save_file)
        self.public_dataset_file = f"{self.client_working_folder}/public_dataset.pth"
        self.pretrain_epochs = pretrain_epochs
        self.kd_epochs = kd_epochs

    def set_parameters(self, model, parameters):
        self.model.load_state_dict(torch.load(self.model_save_file))

    def _init_public_dataloader(self, public_logits=None, **kwargs):
        dataset = torch.load(self.public_dataset_file)
        if public_logits is not None:
            print(type(dataset.tensors))
            print(len(dataset.tensors))
            dataset = TensorDataset(dataset.tensors)
        dataloader = DataLoader(dataset, **kwargs)
        return dataloader

    def _compute_public_logits(self):
        dataloader = self._init_public_dataloader(shuffle=False, batch_size=32)
        public_logits = []
        for batch, _ in dataloader:
            batch.to(self.device)
            with torch.no_grad():
                logits = self.model(batch).numpy()
            public_logits.append(logits)
        public_logits = np.vstack(public_logits)
        return public_logits

    def fit(self, parameters, config):
        if not os.path.exists(self.public_dataset_file):
            assert len(parameters) == 2
            public_dataset = TensorDataset(
                torch.from_numpy(parameters[0]),
                torch.from_numpy(parameters[1]).long()
            )

            # save the public dataset
            torch.save(public_dataset, self.public_dataset_file)

            # transfer learning
            trainloader = DataLoader(public_dataset, shuffle=True, batch_size=32)
            train(
                model=self.model,
                dataloader=trainloader,
                optimizer_name=config["optimizer"],
                epochs=self.pretrain_epochs,
                lr=config["lr"],
                device=self.device
            )

        else:
            print(type(parameters))
            print(len(parameters))

            trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])
            self.set_parameters(self.model, None)
            train(
                model=self.model,
                dataloader=trainloader,
                optimizer_name=config["optimizer"],
                epochs=config["local_epochs"],
                lr=config["lr"],
                device=self.device
            )
        torch.save(self.model.state_dict(), self.model_save_file)
        public_logits = self._compute_public_logits()

            # compute logits on public dataset
        return [public_logits], len(trainloader.dataset), {}


def client_fn(cid, **kwargs) -> FedMDClient:
    return FedMDClient(cid=int(cid), **kwargs)