
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg

from src.models.resnet import Resnet55
from src.models.training_procedures import train_fedgkt_server
from src.helper.parameters import get_parameters
from src.strategies.commons import aggregate_fit_wrapper, get_config, sample_clients


class FedGKT(FedAvg):

    def __init__(self, n_classes, server_batch_size, temperature, server_epochs,
                 server_lr, server_optimizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.global_model = Resnet55(10)
        self.server_batch_size = server_batch_size
        self.temperature = temperature
        self.server_epochs = server_epochs
        self.server_lr = server_lr
        self.server_optimizer = server_optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert kwargs["fraction_fit"] == 1.0, "FedGKT only supports full client participation!!"

    def initialize_parameters(self, client_manager: ClientManager):
        return np.empty((0,))

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ):
        config = get_config(self, server_round)
        clients = sample_clients(self, client_manager, True)

        if isinstance(parameters, np.ndarray) and parameters.size == 0:
            parameters = [parameters] * len(clients)

        return [(client, FitIns(ndarrays_to_parameters(params), config))
                for client, params in zip(clients, parameters)]

    def _get_new_client_logits(self, datasets):
        output_client_logits = []
        for dataset in datasets:
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            output_client_logits.append([])
            for embeddings, _, _ in dataloader:
                with torch.no_grad():
                    embeddings = embeddings.to(self.device)
                    output_client_logits[-1].append(
                        self.global_model(embeddings).reshape(-1, self.n_classes).cpu().numpy()
                    )
            output_client_logits[-1] = np.vstack(output_client_logits[-1])
            print(output_client_logits[-1].shape)
        return output_client_logits

    @aggregate_fit_wrapper
    def aggregate_fit(self, server_round: int, results, failures):

        # Convert results - NOTE: results are not guaranteed to be sorted, need to sort them in code
        weights_results = {
            int(client_proxy.cid): parameters_to_ndarrays(fit_res.parameters)
            for client_proxy, fit_res in results
        }

        datasets = [TensorDataset(
            *[torch.from_numpy(c) for c in weights_results[idx]]
        ) for idx in range(len(weights_results))]

        # why do we not concatenate all the parameters into a single dataset, so that
        # we can shuffle data points of different clients?
        dataloaders = [
            DataLoader(ds, batch_size=self.server_batch_size, shuffle=True) for ds in datasets
        ]
        train_fedgkt_server(
            model=self.global_model,
            dataloaders=dataloaders,
            optimizer_name=self.server_optimizer,
            epochs=self.server_epochs,
            lr=self.server_lr,
            device=self.device,
            temperature=self.temperature
        )

        # do this out of the training loop so we don't need to add to the TensorDataset index-data
        # it should not change the results too much (intuitively, it should improve the results)
        output_client_logits = self._get_new_client_logits(datasets)

        return output_client_logits

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        """Configure the next round of evaluation."""

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)

        server_model_params = ndarrays_to_parameters(get_parameters(self.global_model))
        evaluate_ins = EvaluateIns(server_model_params, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
