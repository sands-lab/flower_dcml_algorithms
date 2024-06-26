import numpy as np
import torch

from src.models.training_procedures import train_fd
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state
from src.data.dataset_partition import DatasetPartition


class FDClient(BaseClient):

    def __init__(self, kd_weight, temperature, **kwargs):
        super().__init__(**kwargs, stateful_client=True)
        self.kd_weight = kd_weight
        self.temperature = temperature

    @sync_rng_state
    def fit(self, parameters, config):
        trainloader = self._init_dataloader(
            dataset_partition=DatasetPartition.TRAIN,
            batch_size=config["batch_size"]
        )

        assert len(parameters) == 1 and isinstance(parameters[0], np.ndarray)
        assert parameters[0].shape == (self.n_classes, self.n_classes) or parameters[0].size == 0
        parameters = torch.from_numpy(parameters[0])
        logits_matrix = train_fd(
            optimization_config=self.get_optimization_config(trainloader, config),
            kd_weight=self.kd_weight,
            logit_matrix=parameters,
            num_classes=self.n_classes,
            temperature=self.temperature
        )
        self.save_model_to_disk()

        return logits_matrix, len(trainloader.dataset), {}


def client_fn(cid, **kwargs) -> FDClient:
    return FDClient(cid=int(cid), **kwargs)
