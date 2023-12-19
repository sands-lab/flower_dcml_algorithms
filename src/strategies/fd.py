import numpy as np
from flwr.common import (
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
    bytes_to_ndarray
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
                FitIns(ndarrays_to_parameters(bytes_to_ndarray(parameters.tensors[i])), config)
                for i in range(len(clients))
            ]

        # Return client/config pairs
        return list(zip(clients, fit_ins))

    @aggregate_fit_wrapper
    def aggregate_fit(self, server_round, results, failures):
        summed_logits = np.zeros((self.n_classes, self.n_classes))
        client_logits = {
            client_proxy.cid: parameters_to_ndarrays(fit_res.parameters)
            for client_proxy, fit_res in results
        }
        sorted_logits = [client_logits[str(i)] for i in range(len(client_logits))]
        for logits in sorted_logits:
            assert len(logits) == 1
            summed_logits += logits[0]

        new_client_logits = []
        for logits in sorted_logits:
            tmp = (summed_logits - logits) / (len(sorted_logits) - 1)
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
