import numpy as np

from src.models.training_procedures import train
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state
from src.data.dataset_partition import DatasetPartition


class FedAvgClient(BaseClient):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, stateful_client=False)

    @sync_rng_state
    def fit(self, parameters, config):
        assert all(np.isfinite(param).all() for param in parameters)

        trainloader = self._init_dataloader(
            dataset_partition=DatasetPartition.TRAIN,
            batch_size=config["batch_size"]
        )
        self.set_parameters(self.model, parameters)
        train(
            self.get_optimization_config(trainloader, config)
        )
        assert all(np.isfinite(param).all() for param in self.get_parameters(config={}))
        return self.get_parameters(config={}), len(trainloader.dataset), {}


def client_fn(cid, **kwargs) -> FedAvgClient:
    return FedAvgClient(cid=int(cid), **kwargs)
