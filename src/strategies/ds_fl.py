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
from src.helper.commons import np_softmax
from src.strategies.fedavg import FedAvg
from src.models.helper import init_model


# pylint: disable=C0103
class DS_FL(FedAvg):

    def __init__(self, temperature, aggregation_method, public_dataset_name,
                 public_dataset_size, public_dataset_csv, train_server_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert aggregation_method in {"era", "sa"}
        assert (temperature == -1) == (aggregation_method == "sa")
        self.aggregation_method = aggregation_method
        self.temperature = temperature
        self.train_server_model = train_server_model
        self.public_dataset_name = public_dataset_name
        self.public_dataset_size = public_dataset_size
        self.public_dataset_params = ndarrays_to_parameters(
            self._load_public_dataset(public_dataset_name, public_dataset_size, public_dataset_csv)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized_clients = set()  # track which clients have received public dataset

    def initialize_parameters(self, client_manager: ClientManager):
        if self.train_server_model:
            # train on the server using (11) a model with maximum capacity
            self.model = init_model("0", 10, self.device, self.dataset_name)
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
        if self.public_dataset_name == "cifar100":
            norm = T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
        else:
            raise Exception()  # TODO - handle other datasets
        image_transforms = T.Compose([
            T.ToTensor(),
            norm
        ])
        for _, row in sampled_data.iterrows():
            image = Image.open(os.path.join(public_dataset_home_folder, row["filename"]))
            image_np = image_transforms(image).unsqueeze(0).numpy()
            images.append(image_np)
        images = np.vstack(images)
        print(f"Public dataset shape: {images.shape}")
        assert images.ndim == 4 and images.shape[0] == self.public_dataset_size
        return [images]

    def _aggregate_era(self, client_logits, client_capacities_weights):
        client_logits = [
            np_softmax(cl / self.temperature, axis=1) for cl in client_logits  # equations 14 & 15
        ]
        return self._aggregate_sa(client_logits, client_capacities_weights)

    def _aggregate_sa(self, client_logits, client_capacities_weights):
        return np.average(client_logits, axis=0, weights=client_capacities_weights)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        config = get_config(self, server_round)
        clients = sample_clients(self, client_manager)

        fit_ins = []
        for client in clients:
            if client.cid in self.initialized_clients:
                # send the current parameters
                assert parameters_to_ndarrays(parameters)[0].shape == (self.public_dataset_size, 10)
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
        client_capacities = [
            self.client_to_capacity_mapping[client.cid] for client, _ in results
        ]

        client_capacities_weights = [
            {
                0: 0.5,
                1: 0.35,
                2: 0.15
            }[capacity] for capacity in client_capacities
        ]
        assert all(len(cl) == 1 for cl in client_logits)
        client_logits = [cl[0] for cl in client_logits]
        assert all(cl.shape == (self.public_dataset_size, 10) for cl in client_logits)
        if self.aggregation_method == "sa":
            aggregated_logits = self._aggregate_sa(client_logits, client_capacities_weights)
        elif self.aggregation_method == "era":
            aggregated_logits = self._aggregate_era(client_logits, client_capacities_weights)
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
