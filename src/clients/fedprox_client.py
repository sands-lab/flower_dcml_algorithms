import copy

from src.models.training_procedures import train_fpx
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state
from src.data.dataset_partition import DatasetPartition


class FedProxClient(BaseClient):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, stateful_client=False)

    @sync_rng_state
    def fit(self, parameters, config):
        trainloader = self._init_dataloader(
            dataset_partition=DatasetPartition.TRAIN,
            batch_size=config["batch_size"]
        )
        self.set_parameters(self.model, parameters)
        global_model = copy.deepcopy(self.model)

        train_fpx(
            optimization_config=self.get_optimization_config(trainloader, config),
            global_model=global_model,
            mu=config["proximal_mu"]
        )
        return self.get_parameters(config={}), len(trainloader.dataset), {}


def client_fn(cid, **kwargs) -> FedProxClient:
    return FedProxClient(cid=int(cid), **kwargs)
