import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.training_procedures import train, train_kd_ds_fl
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state


# pylint: disable=C0103
class DS_FLClient(BaseClient):

    def __init__(self, kd_temperature, **kwargs):
        super().__init__(**kwargs, stateful_client=True)
        self.public_dataset_file = f"{self.client_working_folder}/public_dataset.pth"
        self.kd_temperature = kd_temperature

    def _get_public_dataset_logits(self):
        self.model.eval()
        dataset = torch.load(self.public_dataset_file)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        logits = []
        for batch in dataloader:
            batch = batch[0].to(self.device)
            with torch.no_grad():
                preds = self.model(batch)
            assert preds.ndim == 2
            # l = torch.nn.functional.softmax(preds, dim=1).clone().cpu().numpy()
            logits.append(preds.clone().cpu().numpy())
        logits = np.vstack(logits)
        assert logits.shape == (len(dataset), 10)
        return logits

    def _get_kd_trainloader(self, public_logits, batch_size):
        # construct a dataloader using the logits received from the server and the dataset on FS
        images = torch.load(self.public_dataset_file).tensors[0]
        assert images.shape[0] == public_logits.shape[0]
        dataset = TensorDataset(
            images,
            torch.from_numpy(public_logits)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    @sync_rng_state
    def fit(self, parameters, config):
        # In Algorithm 1 (see paper), distillation happens as a last step, 6. Instead, we here
        # perform it as a first step by conditioning it on the parameters the clients receives
        assert len(parameters) == 1, f"Params {len(parameters)}"
        if not os.path.exists(self.public_dataset_file):
            # it's the first time the client participates...
            # save public dataset to disk... only train on fully supervised task
            public_dataset = TensorDataset(
                torch.from_numpy(parameters[0])
            )
            print(f"Client {self.cid}: saving public dataset")
            torch.save(public_dataset, self.public_dataset_file)
            del public_dataset  # do not keep in memory the dataset while training
        else:
            # perform knowledge distillation
            public_trainloader = self._get_kd_trainloader(parameters[0], config["batch_size"])
            train_kd_ds_fl(
                self.get_optimization_config(public_trainloader, config), self.kd_temperature
            )
            del public_trainloader

        # train on private dataset
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])
        train(
            self.get_optimization_config(trainloader, config)
        )

        self.save_model_to_disk()
        private_dataset_size = len(trainloader.dataset)
        del trainloader

        # compute logits on public datases
        public_dataset_logits = self._get_public_dataset_logits()

        return [public_dataset_logits], private_dataset_size, {}


def client_fn(cid, **kwargs) -> DS_FLClient:
    return DS_FLClient(cid=int(cid), **kwargs)
