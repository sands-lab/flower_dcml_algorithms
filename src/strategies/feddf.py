import json

import torch
from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate

from src.helper.parameters import get_parameters, set_parameters
from src.helper.optimization_config import OptimizationConfig
from src.models.training_procedures import train_feddf
from src.models.helper import init_model
from src.strategies.commons import (
    aggregate_fit_wrapper,
    get_config,
    sample_clients
)
from src.data.cv_dataset import UnlabeledDataset
from src.strategies.fedprox import FedProx


class FedDF(FedProx):

    def __init__(self, public_dataset_name, kd_temperature, kd_optimizer,
                 kd_lr, kd_epochs, public_dataset_size, *args,
                 client_to_capacity_mapping_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_model_capacities = [0, 1]
        self.model_lens = None
        self.set_client_capacity_mapping(client_to_capacity_mapping_file)
        self.public_dataset_name = public_dataset_name
        self.public_dataset_size = public_dataset_size
        self.kd_temperature = kd_temperature
        self.kd_optimizer = kd_optimizer
        self.kd_lr = kd_lr
        self.kd_epochs = kd_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_name = None
        # in a given epoch, a given model capacity might not be represented.
        # therefore, we need to store the models as attributes so that we can retrieve
        # the latest version if this happens. We initialize the model_arrays to None,
        # its value will be set in self.initialize_parameters(...)
        self.model_arrays = None

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def _init_models(self):
        return [
            init_model(capacity, self.n_classes, torch.device("cpu"), self.dataset_name)
            for capacity in self.available_model_capacities
        ]

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        models = self._init_models()
        model_arrays = [
            get_parameters(model) for model in models
        ]
        self.model_lens = [len(ma) for ma in model_arrays]
        self.model_arrays = model_arrays
        return ndarrays_to_parameters(
            self._flatten_models(model_arrays)
        )

    def _flatten_models(self, model_arrays):
        return [
            model_layer_array
            for single_model_arrays in model_arrays
            for model_layer_array in single_model_arrays
        ]

    def _expand_models(self, flattened_models):
        # does the reverse operation than _flatten_models
        models, idx = [], 0
        for ln in self.model_lens:
            models.append(flattened_models[idx:idx+ln])
            idx += ln
        return models

    def set_client_capacity_mapping(self, filepath):
        if filepath is None:
            client_to_capacity_mapping = None
        else:
            with open(filepath, "r") as fp:
                client_to_capacity_mapping = json.load(fp)

            # sanity check
            for k, v in client_to_capacity_mapping.items():
                assert isinstance(k, str) and isinstance(v, int), f"{type(k)} {type(v)}"

        self.client_to_capacity_mapping = client_to_capacity_mapping

    def _map_client_to_capacities(self, clients):
        return [
            self.client_to_capacity_mapping[client.cid] for client in clients
        ]

    def _get_client_models_parameters(self, clients, parameters):
        models = self._expand_models(parameters_to_ndarrays(parameters))
        client_models = [
            models[client_capacity] for client_capacity in self._map_client_to_capacities(clients)
        ]
        return [ndarrays_to_parameters(cm) for cm in client_models]

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        config = get_config(self, server_round)
        config["proximal_mu"] = self.proximal_mu
        clients = sample_clients(self, client_manager)
        client_model_parameters = self._get_client_models_parameters(clients, parameters)

        client_fit_ins = [
            FitIns(model, config) for model in client_model_parameters
        ]
        return list(zip(clients, client_fit_ins))

    def _init_dataloader(self):
        dataset = UnlabeledDataset(self.public_dataset_name, self.public_dataset_size)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                                 shuffle=True, drop_last=True)
        return dataloader

    @aggregate_fit_wrapper
    def aggregate_fit(self, server_round: int, results, failures):
        client_capacities = self._map_client_to_capacities([client for client, _ in results])
        grouped_updated_models = {k: [] for k in self.available_model_capacities}
        teacher_models = []
        for capacity, (_, fit_res) in zip(client_capacities, results):
            grouped_updated_models[capacity].append(
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            )
            tm = init_model(capacity, self.n_classes, self.device, self.dataset_name)
            set_parameters(tm, parameters_to_ndarrays(fit_res.parameters))
            teacher_models.append(tm)

        aggregated_models_dict = {}
        for capacity in self.available_model_capacities:
            if len(grouped_updated_models[capacity]) == 0:
                print(f"Using old value as {capacity} has not been trained")
                aggregated_models_dict[capacity] = self.model_arrays[capacity]
            else:
                aggregated_models_dict[capacity] = aggregate(grouped_updated_models[capacity])

        # perform KD
        updated_models_list = []
        for capacity in self.available_model_capacities:
            print(f"KD model with capacity {capacity}")
            student_model = init_model(capacity, self.n_classes, self.device, self.dataset_name)
            set_parameters(student_model, aggregated_models_dict[capacity])
            optimization_config = OptimizationConfig(
                model=student_model,
                dataloader=self._init_dataloader(),
                optimizer_name=self.kd_optimizer,
                epochs=self.kd_epochs,
                lr=self.kd_lr,
                device=self.device,
            )
            train_feddf(
                optimization_config=optimization_config,
                temperature=self.kd_temperature,
                teacher_models=teacher_models
            )
            updated_models_list.append(
                get_parameters(student_model)
            )

        # save the latest model versions
        self.model_arrays = updated_models_list
        updated_models_list = self._flatten_models(updated_models_list)
        return ndarrays_to_parameters(updated_models_list)


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        if not self.evaluate_round(server_round):
            return []

        clients = sample_clients(self, client_manager)
        config = get_config(self, server_round)

        client_model_parameters = self._get_client_models_parameters(clients, parameters)
        client_eval_ins = [EvaluateIns(params, config) for params in client_model_parameters]
        return list(zip(clients, client_eval_ins))
