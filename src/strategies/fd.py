import numpy as np
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg

from src.strategies.commons import (
    configure_evaluate_no_params,
    aggregate_fit_wrapper,
    sample_clients,
    get_config
)


class FD(FedAvg):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        assert kwargs["fraction_fit"] == 1.0, "FD only supports full client participation!!"

    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters([np.empty((0,))])

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        config = get_config(self, server_round)
        clients = sample_clients(self, client_manager, sort_clients_by_cid=True)

        params_np = parameters_to_ndarrays(parameters)
        assert (len(params_np) == 1 and params_np[0].size == 0) or \
               (len(params_np) == len(clients) and params_np[0].size == self.n_classes ** 2)
        if len(params_np) == 1:
            fit_ins = [FitIns(parameters, config)] * len(clients)
        else:
            fit_ins = [
                FitIns(ndarrays_to_parameters([params_np[i]]), config)
                for i in range(len(clients))
            ]

        # Return client/config pairs
        return list(zip(clients, fit_ins))

    @aggregate_fit_wrapper
    def aggregate_fit(self, server_round, results, failures):
        summed_logits = np.zeros((self.n_classes, self.n_classes))
        received_client_data = {
            client_proxy.cid: parameters_to_ndarrays(fit_res.parameters)
            for client_proxy, fit_res in results
        }

        client_logits, client_labels = [], []
        for i in range(len(received_client_data)):
            client_data = received_client_data[str(i)]
            logits, labels = client_data[0], client_data[1]
            expanded_logits = np.zeros((self.n_classes, self.n_classes), dtype=np.float32)
            expanded_logits[labels] = logits

            client_logits.append(expanded_logits)
            client_labels.append(labels.astype(np.int32).reshape(1, -1))
        client_labels = np.vstack(client_labels)
        summed_logits = np.sum(client_logits, axis=0)

        new_client_logits = []
        for idx, logits in enumerate(client_logits):
            n_clients_with_labels = \
                np.delete(client_labels, idx, axis=0).sum(axis=0).reshape(-1, 1)
            tmp = (summed_logits - logits) / n_clients_with_labels
            new_client_logits.append(tmp)
        logits_parameters = ndarrays_to_parameters(new_client_logits)

        return logits_parameters

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        clients = sample_clients(self, client_manager)
        return configure_evaluate_no_params(
            strategy=self,
            server_round=server_round,
            sampled_clients=clients,
        )
