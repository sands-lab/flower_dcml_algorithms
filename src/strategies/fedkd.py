from src.strategies.fedavg import FedAvg
from src.strategies.commons import (
    configure_evaluate_no_params,
    sample_clients,
)

class FedKD(FedAvg):

    def configure_evaluate(
        self, server_round: int, parameters, client_manager
    ):
        if not self.evaluate_round(server_round):
            return []

        clients = sample_clients(self, client_manager)
        return configure_evaluate_no_params(
            strategy=self,
            server_round=server_round,
            sampled_clients=clients,
        )
