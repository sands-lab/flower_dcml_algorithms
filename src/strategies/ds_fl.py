import os
from PIL import Image

import pandas as pd
import numpy as np
import torchvision.transforms.functional as F

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitIns
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg

from src.strategies.commons import (
    configure_evaluate_no_params,
    aggregate_fit_wrapper,
    sample_clients,
    get_config
)
from src.helper.commons import np_softmax


# pylint: disable=C0103
class DS_FL(FedAvg):

    def __init__(self, n_classes, temperature, aggregation_method, public_dataset_name,
                 public_dataset_size, public_dataset_csv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        assert aggregation_method in {"era", "sa"}
        assert (temperature == -1) == (aggregation_method == "sa")
        self.aggregation_method = aggregation_method
        self.temperature = temperature
        self.public_dataset_params = ndarrays_to_parameters(
            self._load_public_dataset(public_dataset_name, public_dataset_size, public_dataset_csv)
        )
        self.initialized_clients = set()  # track which clients have received public dataset

    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters([np.empty((0,))])

    def _load_public_dataset(self, dataset_name, dataset_size, public_dataset_csv):
        public_dataset_home_folder = \
            os.path.join(os.environ.get("FLTB_DATA_HOME_FOLDER"), dataset_name)
        if public_dataset_csv is None:
            sampled_data = pd.read_csv(f"{public_dataset_home_folder}/metadata.csv")\
                .sample(n=dataset_size, replace=False, random_state=10, axis=0)
        else:
            sampled_data = pd.read_csv(public_dataset_csv)

        images = []
        for _, row in sampled_data.iterrows():
            image = Image.open(os.path.join(public_dataset_home_folder, row["filename"]))
            image_tensor = F.to_tensor(image)
            image_np = image_tensor.numpy()[None,:,:,:]
            images.append(image_np)
        images = np.vstack(images)
        return [images]

    def _aggregate_era(self, client_logits):
        client_logits = [
            np_softmax(cl / self.temperature, axis=1) for cl in client_logits  # equations 14 & 15
        ]
        return self._aggregate_sa(client_logits)

    def _aggregate_sa(self, client_logits):
        return np.mean(client_logits, axis=0)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        config = get_config(self, server_round)
        clients = sample_clients(self, client_manager)

        fit_ins = []
        for client in clients:
            if client.cid in self.initialized_clients:
                # send the current parameters
                assert parameters_to_ndarrays(parameters)[0].size > 0
                fit_in = FitIns(parameters, config)
            else:
                # send the global dataset
                fit_in = FitIns(self.public_dataset_params, config)
                self.initialized_clients.add(client.cid)

            fit_ins.append(fit_in)

        # Return client/config pairs
        return list(zip(clients, fit_ins))

    @aggregate_fit_wrapper
    def aggregate_fit(self, server_round, results, failures):
        # Convert results
        client_logits = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]
        assert all(len(cl) == 1 for cl in client_logits)
        client_logits = [cl[0] for cl in client_logits]
        if self.aggregation_method == "sa":
            aggregated_logits = self._aggregate_sa(client_logits)
        elif self.aggregation_method == "era":
            aggregated_logits = self._aggregate_era(client_logits)
        aggregated_parameters = ndarrays_to_parameters([aggregated_logits])

        # Aggregate custom metrics if aggregation fn was provided
        return aggregated_parameters

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        clients = sample_clients(self, client_manager)
        return configure_evaluate_no_params(
            strategy=self,
            server_round=server_round,
            sampled_clients=clients,
        )
