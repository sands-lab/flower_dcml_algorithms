
import os
import json
import typing
from PIL import Image

import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as F
from flwr.server.client_proxy import ClientProxy
import numpy as np
import pandas as pd
from flwr.common import EvaluateIns, FitIns, FitRes, Parameters, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg

from src.models.resnet import Resnet55
from src.models.training_procedures import train_fedgkt_server
from src.helper.parameters import get_parameters

# from flwr.server.server import Server
class FedMD(FedAvg):

    def __init__(self, n_classes, public_dataset, size_public_dataset_sample, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.public_dataset = public_dataset
        self.size_public_dataset_sample = size_public_dataset_sample

    def _load_public_dataset(self):
        public_dataset_home_folder = os.path.join(os.environ.get("FLTB_DATA_HOME_FOLDER"), self.public_dataset)
        sampled_data = pd.read_csv(f"{public_dataset_home_folder}/metadata.csv")\
            .sample(n=self.size_public_dataset_sample, replace=False, random_state=10, axis=0)
        targets = sampled_data["label"].to_numpy(dtype=np.int32)
        images = []
        for _, row in sampled_data.iterrows():
            image = Image.open(os.path.join(public_dataset_home_folder, row["filename"]))
            image_tensor = F.to_tensor(image)
            image_np = image_tensor.numpy()[None,:,:,:]
            images.append(image_np)
        images = np.vstack(images)
        return [images, targets]

    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters([np.empty((0,))])

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        # flower does not provide a straightforward way to send data to all the clients - in this case we need to send
        # the public dataset used for pre-training and for TL. Therefore, we do the following: in the first epoch,
        # we send the public dataset and let the clients pre-train their model. In the subsequent epochs, the
        # logits are distributed to the clients. It's a bit hacky, but it should work!
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        if server_round == 1:
            # serve the public dataset to all the clients
            public_dataset_numpy = self._load_public_dataset()
            public_dataset_parameters = ndarrays_to_parameters(public_dataset_numpy)
            fit_ins = FitIns(public_dataset_parameters, config)
            clients = [client for client in client_manager.all().values()]
        else:
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
            fit_ins = FitIns(parameters, config)

        return [(client, fit_ins) for client in clients]


    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            parameters_to_ndarrays(fit_res.parameters)[0] for _, fit_res in results
        ]
        averaged_logits = np.mean(weights_results, axis=0)
        parameters_aggregated = ndarrays_to_parameters([averaged_logits])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(ndarrays_to_parameters(np.empty(0,)), config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]