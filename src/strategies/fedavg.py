from typing import Optional

from flwr.server.strategy import FedAvg as FlFedAvg
from flwr.common import parameters_to_ndarrays, Parameters
from flwr.server.client_manager import ClientManager

from src.strategies.commons import set_client_capacity_mapping


class FedAvg(FlFedAvg):

    def __init__(self, n_classes, evaluation_freq, filter_capacity=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.evaluation_freq = evaluation_freq
        self.filter_capacity = filter_capacity
        self.converged = False
        self.client_to_capacity_mapping = None
        self.available_model_capacities = None
        self.dataset_name = None

    def _unregister_clients(self, client_manager: ClientManager):

        all_clients = client_manager.all()
        delete_clients = []
        for cid, client_proxy in all_clients.items():
            if self.client_to_capacity_mapping[cid] != self.filter_capacity:
                delete_clients.append(client_proxy)
        for client_proxy in delete_clients:
            client_manager.unregister(client_proxy)
        print(f"Reduced number of clients to {len(client_manager)}")

    def set_client_capacity_mapping(self, filepath):
        print("setting capacity")
        self.client_to_capacity_mapping = set_client_capacity_mapping(filepath)
        if self.client_to_capacity_mapping is not None:
            self.available_model_capacities = list(set(self.client_to_capacity_mapping.values()))
            print(f"Available client capacities: {self.available_model_capacities}")

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        if self.filter_capacity is not None:
            # Want to train only clients with a given capacity... unregister all clients
            # with a different capacity
            self._unregister_clients(client_manager)

        return super().initialize_parameters(client_manager)

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        received_data_shape = parameters_to_ndarrays(results[0][1].parameters)
        n_params = sum(d.size for d in received_data_shape)
        print(f"Received number of parameters: {n_params}")
        return super().aggregate_fit(server_round, results, failures)

    def evaluate_round(self, server_round):
        return server_round % self.evaluation_freq == 0 and not self.converged

    def configure_fit(self, *args, **kwargs):
        if self.converged:
            return []
        return super().configure_fit(*args, **kwargs)

    def configure_evaluate(
        self, server_round: int, parameters, client_manager
    ):
        eval_ins = []
        if self.evaluate_round(server_round):
            eval_ins = super().configure_evaluate(server_round, parameters, client_manager)
        return eval_ins

    def set_converged(self):
        self.converged = True
