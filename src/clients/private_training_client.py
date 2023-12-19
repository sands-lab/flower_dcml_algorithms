
import numpy as np

from src.models.training_procedures import train
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state


class PrivateTrainingClient(BaseClient):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, stateful_client=True)

    @sync_rng_state
    def fit(self, parameters, config):
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])
        train(
            self.get_optimization_config(trainloader, config)
        )
        self.save_model_to_disk()
        return [np.empty(0,)], len(trainloader.dataset), {}


def client_fn(cid, **kwargs) -> PrivateTrainingClient:
    return PrivateTrainingClient(cid=int(cid), **kwargs)
