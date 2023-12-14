from typing import Dict, List, Optional, Tuple, Union

import torch
from flwr.server.client_proxy import ClientProxy
import numpy as np
import flwr as fl
from flwr.common import EvaluateIns, FitIns, FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg

from src.models.resnet import Resnet55
from src.models.training_procedures import train_fedgkt_server

# from flwr.server.server import Server
class FedGKT(FedAvg):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.global_model = Resnet55(10)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert kwargs["fraction_fit"] == 1.0, "FedGKT only supports full client participation!!"

    def initialize_parameters(self, client_manager: ClientManager):
        return np.empty((0,))

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}

        all_clients = client_manager.all()
        all_clients = [all_clients[str(i)] for i in range(len(all_clients))]
        print(all_clients)

        if isinstance(parameters, np.ndarray) and parameters.size == 0:
            parameters = [parameters] * len(all_clients)
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        # Return client/config pairs

        return [(client, FitIns(Parameters(params, tensor_type="numpy.ndarray"), config)) for client, params in zip(all_clients, parameters)][:1]


    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        print("AGGREGATING THE FUCKING FIT")
        embeddings = np.vstack([w[0] for w in weights_results])
        logits = np.vstack([w[0] for w in weights_results])
        targets = np.vstack([w[0] for w in weights_results])
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(embeddings),
            torch.from_numpy(logits),
            torch.from_numpy(targets)
        )
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        train_fedgkt_server(self.global_model, trainloader, "adam", 1, 0.1, self.device)

        # we compute the new logits outside of the training loop (for ease of implementation)
        new_logits = []
        for embeddings, _, _ in weights_results:
            client_logits = []
            for e in embeddings:
                e = torch.from_numpy(e).to(self.device).unsqueeze(0)
                with torch.no_grad():
                    logit = self.global_model(e)
                client_logits.append(logit.cpu().numpy())
            client_logits = np.vstack(client_logits)
            new_logits.append(client_logits)
        new_logits = np.vstack(new_logits)
        print(new_logits.shape)
        print("AGGREGATED THE FUCKING FIT!!")
        import sys
        sys.exit(0)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return None, metrics_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        print("evaluating....")
        import sys
        sys.exit(0)