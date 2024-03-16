import numpy as np

from src.helper.parameters import get_parameters, set_parameters
from src.helper.commons import sync_rng_state, set_seed, read_json
from src.helper.filepaths import FilePaths as FP
from src.models.training_procedures import train_fedkd
from src.models.helper import init_model_from_string
from src.clients.base_client import BaseClient
from src.data.dataset_partition import DatasetPartition


class FedKDClient(BaseClient):

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs, stateful_client=True)
        model_string = read_json(FP.FEDKD_MODEL_CONFIG, [self.dataset_name])
        set_seed(self.seed)
        # init the shared model on the CPU because if we are going to call the evaluate method
        # we don't actually need the model, so we can delete it. by initializing it on the cpu,
        # we only transfer it to cuda if it is necessary
        self.shared_model = init_model_from_string(model_string, self.n_classes, 1.0, "cpu")
        self.temperature = temperature

    def get_parameters(self, config):
        return get_parameters(self.shared_model)

    @sync_rng_state
    def fit(self, parameters, config):
        assert all(np.isfinite(param).all() for param in parameters)
        set_parameters(self.shared_model, parameters)
        trainloader = self._init_dataloader(
            dataset_partition=DatasetPartition.TRAIN,
            batch_size=config["batch_size"]
        )
        train_fedkd(
            self.get_optimization_config(trainloader, config), self.shared_model, self.temperature
        )
        assert all(np.isfinite(param).all() for param in self.get_parameters(config={}))
        self.save_model_to_disk()
        return self.get_parameters(config={}), len(trainloader.dataset), {}



def client_fn(cid, **kwargs) -> FedKDClient:
    return FedKDClient(cid=int(cid), **kwargs)
