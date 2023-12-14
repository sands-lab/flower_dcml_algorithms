from src.models.training_procedures import train_fpx
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state


class FedProxClient(BaseClient):

    @sync_rng_state
    def fit(self, parameters, config):
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])
        self.set_parameters(self.model, parameters)
        global_model = self._init_model()
        self.set_parameters(global_model, parameters)
        train_fpx(
            model=self.model,
            dataloader=trainloader,
            optimizer_name=config["optimizer"],
            epochs=config["local_epochs"],
            device=self.device,
            lr=config["lr"],
            global_model=global_model,
            mu=config["proximal_mu"]
        )
        return self.get_parameters(config={}), len(trainloader.dataset), {}


def client_fn(cid, images_folder, partition_folder, seed, experiment_folder) -> FedProxClient:
    return FedProxClient(int(cid), images_folder, partition_folder, seed, experiment_folder, model_name="convnet")