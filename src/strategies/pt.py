import json

import torch
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitIns,
    EvaluateIns
)

from src.models.helper import init_pt_model
from src.models.pt_commons import aggregate_submodels
from src.models.abstract_model import AbstratModel
from src.strategies.fedavg import FedAvg
from src.strategies.commons import (
    set_client_capacity_mapping,
    get_config,
    sample_clients,
    aggregate_fit_wrapper
)
from src.helper.parameters import set_parameters, get_parameters


class PT(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.dataset_name = None
        self.round_submodel_configs = None
        self.client_to_capacity_mapping = None
        self.set_capacity_to_rate_mapping()

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def initialize_parameters(self, client_manager):
        self.model: AbstratModel = \
            init_pt_model(0, self.n_classes, torch.device("cpu"), self.dataset_name)
        return ndarrays_to_parameters(get_parameters(self.model))

    def setup_round(self, server_round, n_round_clients):
        # this method is called at the beginning of the configure_fit method, so it gets called
        # before all the calls to the configure_client_submodel. Use it to setup any additional
        # meta-configuration of the round (e.g. when using IST). Most algorithms do not need
        # to override this method
        pass

    def configure_client_submodel(self, whole_model_config, reduced_model_config):
        raise NotImplementedError("This method should be overwritten by the specific algorithm")

    def set_client_capacity_mapping(self, filepath):
        self.client_to_capacity_mapping = set_client_capacity_mapping(filepath)

    def set_capacity_to_rate_mapping(self):
        with open("pt_model_config.json", "r") as fp:
            config = json.load(fp)
        self.capacity_to_rate_mapping = config["capacity_mapping"]

    def _get_client_rate(self, cid):
        client_capacity = self.client_to_capacity_mapping[cid]
        client_rate = self.capacity_to_rate_mapping[str(client_capacity)]
        return client_rate

    def _extract_submodel_params(self, clients):
        client_submodels = []
        round_submodel_configs = {}
        for client in clients:
            cid = client.cid
            client_rate = self._get_client_rate(cid)
            capacity_config = self.model.get_reduced_model_config(client_rate)
            client_submodel_config_idx = self.configure_client_submodel(
                self.model.whole_model_config,
                capacity_config
            )
            expanded_config_idx = \
                self.model.expand_configuration_to_model(client_submodel_config_idx)
            round_submodel_configs[cid] = expanded_config_idx
            client_submodel_parameters = \
                self.model.extract_submodel_parameters(expanded_config_idx)
            client_submodels.append(client_submodel_parameters)
            print(cid, sum(p.size for p in client_submodel_parameters))
        return client_submodels, round_submodel_configs

    def configure_fit(
            self, server_round: int, parameters, client_manager
    ):
        # overthinking, will remove
        assert all(bool((a == b).all())
                   for a, b in zip(parameters_to_ndarrays(parameters), get_parameters(self.model)))
        config = get_config(self, server_round)
        clients = sample_clients(self, client_manager)
        self.setup_round(server_round, len(clients))

        client_submodels, self.round_submodel_configs = self._extract_submodel_params(clients)

        client_fit_ins = [
            FitIns(ndarrays_to_parameters(model), config) for model in client_submodels
        ]
        return list(zip(clients, client_fit_ins))

    @aggregate_fit_wrapper
    def aggregate_fit(self, server_round: int, results, failures):
        # initialize all models on the server
        submodels_list, submodel_configs = [], []
        for client, client_res in results:
            submodel_params = parameters_to_ndarrays(client_res.parameters)

            submodel = init_pt_model(
                client_capacity=self.client_to_capacity_mapping[client.cid],
                n_classes=self.n_classes,
                device=torch.device("cpu"),
                dataset=self.dataset_name
            )
            set_parameters(submodel, submodel_params)

            submodels_list.append(submodel)
            submodel_configs.append(self.round_submodel_configs[client.cid])

        updated_model_params = aggregate_submodels(self.model, submodels_list, submodel_configs)
        set_parameters(self.model, updated_model_params)

        return ndarrays_to_parameters(updated_model_params)

    def configure_evaluate(
        self, server_round: int, parameters, client_manager
    ):
        if not self.evaluate_round(server_round):
            return []
        # overthinkinh... can remove it
        assert all(bool((a == b).all())
                   for a, b in zip(parameters_to_ndarrays(parameters), get_parameters(self.model)))

        clients = sample_clients(self, client_manager)
        config = get_config(self, server_round)

        client_submodels, _ = self._extract_submodel_params(clients)

        client_eval_ins = [EvaluateIns(ndarrays_to_parameters(params), config)
                           for params in client_submodels]
        return list(zip(clients, client_eval_ins))
