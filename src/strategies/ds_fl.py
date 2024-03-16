import os
from PIL import Image

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays, FitIns
from flwr.server.client_manager import ClientManager

from src.strategies.commons import (
    configure_evaluate_no_params,
    aggregate_fit_wrapper,
    sample_clients,
    get_config
)
from src.strategies.fedavg import FedAvg
from src.helper.commons import read_json
from src.helper.environment_variables import EnvironmentVariables as EV


# pylint: disable=C0103
class DS_FL(FedAvg):

    def __init__(self, temperature, aggregation_method, public_dataset_name,
                 public_dataset_size, public_dataset_csv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert aggregation_method in {"era", "sa"}
        assert (temperature == -1) == (aggregation_method == "sa")
        self.aggregation_method = aggregation_method
        self.temperature = temperature
        self.public_dataset_name = public_dataset_name
        self.public_dataset_size = public_dataset_size
        self.public_dataset_params = ndarrays_to_parameters(
            self._load_public_dataset(public_dataset_name, public_dataset_size, public_dataset_csv)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized_clients = set()  # track which clients have received public dataset

    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters([np.empty((0,))])

    def _load_public_dataset(self, dataset_name, dataset_size, public_dataset_csv):
        public_dataset_home_folder = \
            os.path.join(os.environ.get(EV.DATA_HOME_FOLDER), dataset_name)
        if public_dataset_csv is None:
            sampled_data = pd.read_csv(f"{public_dataset_home_folder}/metadata.csv")\
                .sample(n=dataset_size, replace=False, random_state=10, axis=0)
        else:
            sampled_data = pd.read_csv(public_dataset_csv)

        images = []
        norm_params = read_json(
            "config/data/data_configuration.json",
            [self.public_dataset_name, "normalization_parameters"]
        )

        image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=norm_params["mean"], std=norm_params["std"])
        ])
        for _, row in sampled_data.iterrows():
            image = Image.open(os.path.join(public_dataset_home_folder, row["filename"]))
            image_np = image_transforms(image).unsqueeze(0).numpy()
            images.append(image_np)
        images = np.vstack(images)
        print(f"Public dataset shape: {images.shape}")
        assert images.ndim == 4 and images.shape[0] == self.public_dataset_size
        return [images]

    def _aggregate_sa(self, client_logits):
        return np.mean(client_logits, axis=0)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        if self.converged:
            return []
        config = get_config(self, server_round)
        clients = sample_clients(self, client_manager)

        fit_ins = []
        for client in clients:
            if client.cid in self.initialized_clients:
                # send the current parameters
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

        client_logits = [cl[0] for cl in client_logits]
        aggregated_logits = self._aggregate_sa(client_logits)

        aggregated_parameters = ndarrays_to_parameters([aggregated_logits])

        # Aggregate custom metrics if aggregation fn was provided
        return aggregated_parameters

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        if not self.evaluate_round(server_round):
            return []

        clients = sample_clients(self, client_manager)
        return configure_evaluate_no_params(
            strategy=self,
            server_round=server_round,
            sampled_clients=clients,
        )
