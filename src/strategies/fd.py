import numpy as np
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager

from src.strategies.commons import (
    configure_evaluate_no_params,
    aggregate_fit_wrapper,
    sample_clients,
)
from src.strategies.fedavg import FedAvg


class FD(FedAvg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters([np.empty((0,))])

    @aggregate_fit_wrapper
    def aggregate_fit(self, server_round, results, failures):
        summed_logits = np.zeros((self.n_classes, self.n_classes))
        received_client_data = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        client_logits, client_labels = [], np.zeros(shape=(self.n_classes,))
        for client_data in received_client_data:
            logits, labels = client_data[0], client_data[1]
            expanded_logits = np.zeros((self.n_classes, self.n_classes), dtype=np.float32)
            expanded_logits[labels] = logits

            client_logits.append(expanded_logits)
            client_labels[labels] += 1
        summed_logits = np.sum(client_logits, axis=0)
        average_logits = summed_logits / client_labels.reshape(-1, 1)
        return ndarrays_to_parameters([average_logits])

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
