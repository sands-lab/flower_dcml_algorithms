import numpy as np
import torch

from src.models.training_procedures import train_fd
from src.clients.base_client import BaseClient


class FDClient(BaseClient):

    def __init__(self, kd_weight, eval_after_train, **kwargs):
        super().__init__(**kwargs, stateful_client=True)
        self.kd_weight = kd_weight
        self.eval_after_train = eval_after_train

    def fit(self, parameters, config):
        trainloader = self._init_dataloader(train=True, batch_size=config["batch_size"])

        assert len(parameters) == 1 and isinstance(parameters[0], np.ndarray)
        parameters = torch.from_numpy(parameters[0])
        logits_matrix = train_fd(
            optimization_config=self.get_optimization_config(trainloader, config),
            kd_weight=self.kd_weight,
            logit_matrix=parameters,
            num_classes=self.n_classes
        )
        self.save_model_to_disk()

        return [logits_matrix], len(trainloader.dataset), {}


def client_fn(cid, **kwargs) -> FDClient:
    return FDClient(cid=int(cid), **kwargs)
