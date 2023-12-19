from src.models.training_procedures import train
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state


class FedAvgClient(BaseClient):

    def __init__(self, **kwargs):
        super().__init__(**kwargs, stateful_client=False)

    @sync_rng_state
    def fit(self, parameters, config):
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])
        self.set_parameters(self.model, parameters)
        train(
            self.get_optimization_config(trainloader, config)
        )
        return self.get_parameters(config={}), len(trainloader.dataset), {}


def client_fn(cid, **kwargs) -> FedAvgClient:
    return FedAvgClient(cid=int(cid), **kwargs)
