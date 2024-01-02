from flwr.server.strategy import FedAvg as FlFedAvg
from flwr.common import parameters_to_ndarrays


class FedAvg(FlFedAvg):

    def __init__(self, n_classes, evaluation_freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.evaluation_freq = evaluation_freq

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
        return server_round % self.evaluation_freq == 0

    def configure_evaluate(
        self, server_round: int, parameters, client_manager
    ):
        if self.evaluate_round(server_round):
            eval_res = super().configure_evaluate(server_round, parameters, client_manager)
        else:
            eval_res = []
        return eval_res
