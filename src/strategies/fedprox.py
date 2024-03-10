from flwr.common import FitIns

from src.strategies.fedavg import FedAvg


class FedProx(FedAvg):

    def __init__(self, proximal_mu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu

    def configure_fit(
        self, server_round, parameters, client_manager
    ):
        if self.converged:
            return []
        client_config_pairs = super().configure_fit(
            server_round, parameters, client_manager
        )

        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "proximal_mu": self.proximal_mu},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]
