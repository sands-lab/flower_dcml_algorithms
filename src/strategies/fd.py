import numpy as np
import flwr as fl
from flwr.common import Parameters, EvaluateIns
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg


class FD(FedAvg):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes

    def initialize_parameters(self, client_manager: ClientManager):
        return np.empty((self.n_classes, self.n_classes))

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)
        parameters = fl.common.ndarrays_to_parameters(np.empty(0,))  # override parameters... pass empty list
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

