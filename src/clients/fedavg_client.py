from src.models.training_procedures import train
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state

from fltb.decorators import MonitorFlwrClient


@MonitorFlwrClient
class FedAvgClient(BaseClient):

    @sync_rng_state
    def fit(self, parameters, config):
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])
        self.set_parameters(self.model, parameters)
        train(
            model=self.model,
            dataloader=trainloader,
            optimizer_name=config["optimizer"],
            epochs=config["local_epochs"],
            lr=config["lr"],
            device=self.device
        )
        return self.get_parameters(config={}), len(trainloader.dataset), {}


def client_fn(cid, images_folder, partition_folder, seed, experiment_folder) -> FedAvgClient:
    return FedAvgClient(int(cid), images_folder, partition_folder, seed, experiment_folder, model_name="convnet")