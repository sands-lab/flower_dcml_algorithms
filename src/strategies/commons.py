import json
import inspect

import numpy as np
from flwr.common import EvaluateIns, ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg


def configure_evaluate_no_params(
        strategy: FedAvg,
        server_round: int,
        sampled_clients
):
    """Evaluates client by passing an empty array of parameters. Use this function for
    stateful clients.
    """
    if strategy.fraction_evaluate == 0.0:
        return []

    # Parameters and config
    config = {}
    if strategy.on_evaluate_config_fn is not None:
        config = strategy.on_evaluate_config_fn(server_round)

    eval_params = ndarrays_to_parameters(np.empty(0,))
    evaluate_ins = EvaluateIns(eval_params, config)

    # Return client/config pairs
    return [(client, evaluate_ins) for client in sampled_clients]


def aggregate_fit_wrapper(func):
    def wrapper(_self, server_round, results, failures):
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not _self.accept_failures and failures:
            return None, {}

        # trigger computation
        result = func(_self, server_round, results, failures)

        metrics_aggregated = {}
        if _self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = _self.fit_metrics_aggregation_fn(fit_metrics)

        return result, metrics_aggregated
    return wrapper


def sample_clients(
        strategy: FedAvg,
        client_manager: ClientManager,
        sort_clients_by_cid: bool=False
):
    step = inspect.stack()[1][3].split("_")[-1]
    assert step in {"fit", "evaluate"}

    # get number of clients to sample
    if step == "evaluate":
        sample_size, min_num_clients = strategy.num_evaluation_clients(
            client_manager.num_available()
        )
    else:
        sample_size, min_num_clients = strategy.num_fit_clients(
            client_manager.num_available()
        )
    # sample clients
    clients = client_manager.sample(
        num_clients=sample_size, min_num_clients=min_num_clients
    )

    # optionally sort clients
    if sort_clients_by_cid:
        clients.sort(key=lambda x: int(x.cid), reverse=False)
    return clients


def get_config(strategy: FedAvg, server_round: int):
    step = inspect.stack()[1][3].split("_")[-1]
    assert step in {"fit", "evaluate"}
    config = {}
    if step == "fit" and strategy.on_fit_config_fn is not None:
        config = strategy.on_fit_config_fn(server_round)
    elif step == "evaluate" and strategy.on_evaluate_config_fn is not None:
        config = strategy.on_evaluate_config_fn(server_round)
    return config
