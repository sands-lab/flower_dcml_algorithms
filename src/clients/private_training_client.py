import pickle

import numpy as np

from src.models.training_procedures import train
from src.clients.base_client import BaseClient
from src.helper.commons import sync_rng_state
from src.data.dataset_partition import DatasetPartition


def handle_early_stopping(accuracy, file):
    train = True
    if accuracy is not None:
        with open(file, "rb") as fp:
            accuracies = pickle.load(fp)
            accuracies.append(accuracy)
            if (
                len(accuracies) > 10 and
                np.mean(accuracies[-10:-5]) * 0.9 > np.mean(accuracies[-5:])
            ):
                train = False
    else:
        accuracies = []
    with open(file, "wb") as fp:
        pickle.dump(accuracies, fp)
    return train


class PrivateTrainingClient(BaseClient):

    def __init__(self, weight_decay, **kwargs):
        super().__init__(**kwargs, stateful_client=True)
        self.weight_decay = weight_decay
        self.accuracy_file = f"{self.client_working_folder}/accuracies.pkl"
        self.train = handle_early_stopping(None, self.accuracy_file)

    @sync_rng_state
    def fit(self, parameters, config):
        _ = (parameters,)
        if self.train:
            trainloader = self._init_dataloader(
                dataset_partition=DatasetPartition.TRAIN,
                batch_size=config["batch_size"]
            )
            train(
                self.get_optimization_config(trainloader, config, weight_decay=self.weight_decay)
            )
            self.save_model_to_disk()
        return [np.empty(0,)], len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        res = super().evaluate(parameters, config)
        self.train = handle_early_stopping(res[0], self.accuracy_file)
        return res


def client_fn(cid, **kwargs) -> PrivateTrainingClient:
    return PrivateTrainingClient(cid=int(cid), **kwargs)
