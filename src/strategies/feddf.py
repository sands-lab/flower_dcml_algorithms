import torch
from flwr.common import (
    EvaluateIns,
    FitIns,
    Parameters,
    GetParametersIns,
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
                 kd_lr, kd_epochs, public_dataset_size, weight_predictions, *args,
                 client_to_capacity_mapping_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.available_model_capacities = None
        self.model_lens = None
        self.set_client_capacity_mapping(client_to_capacity_mapping_file)
        self.public_dataset_name = public_dataset_name
        self.public_dataset_size = public_dataset_size
        self.kd_temperature = kd_temperature
        self.kd_optimizer = kd_optimizer
        self.kd_lr = kd_lr
        self.kd_epochs = kd_epochs
        self.weight_predictions = weight_predictions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert torch.cuda.is_available()
        self.dataset_name = None
        # in a given epoch, a given model capacity might not be represented.
        # therefore, we need to store the models as attributes so that we can retrieve
        # the latest version if this happens. We initialize the model_arrays to None,
        # its value will be set in self.initialize_parameters(...)
        self.model_arrays = None

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        noins = GetParametersIns({})
        clients = client_manager.all()
        model_arrays = []
        for capacity in self.available_model_capacities:
            for _, client in clients.items():
                if self.client_to_capacity_mapping[client.cid] == capacity:
                    model_arrays.append(
                        parameters_to_ndarrays(client.get_parameters(noins, None).parameters)
                    )
                    break

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

    def _map_client_to_capacities(self, clients):
        return [
            self.client_to_capacity_mapping[client.cid] for client in clients
        ]

    def _get_client_models_parameters(self, clients, parameters):
        models = {k: model for k, model in zip(self.available_model_capacities, self.model_arrays)}
        client_models = [
            models[client_capacity] for client_capacity in self._map_client_to_capacities(clients)
        ]
        return [ndarrays_to_parameters(cm) for cm in client_models]

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ):
        if self.converged:
            return []
        config = get_config(self, server_round)
        config["proximal_mu"] = self.proximal_mu
        clients = sample_clients(self, client_manager)
        client_model_parameters = self._get_client_models_parameters(clients, parameters)

        client_fit_ins = [
            FitIns(model, config) for model in client_model_parameters
        ]
        return list(zip(clients, client_fit_ins))

    def _init_dataloader(self, teacher_models, teacher_capacities):
        dataset = UnlabeledDataset(
            self.public_dataset_name,
            self.public_dataset_size
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, pin_memory=True,
                                                 shuffle=False, drop_last=True, num_workers=4)
        if self.weight_predictions:
            weights = torch.Tensor([
                {
                    0: 0.6,
                    1: 0.35,
                    2: 0.05
                }[capacity] for capacity in teacher_capacities
            ])
        else:
            weights = torch.ones(len(teacher_capacities))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = weights.reshape(-1, 1, 1).to(device).float()
        weights_sum = weights.sum().to(device)
        for idx in range(len(teacher_models)):
            teacher_models[idx] = teacher_models[idx].eval().to(device)

        average_predictions, all_images = [], []
        with torch.no_grad():
            for images in dataloader:
                all_images.append(images)
                images = images.to(device)
                batch_predictions = torch.vstack([
                    teacher_model(images).unsqueeze(0) for teacher_model in teacher_models
                ])

                batch_predictions = (batch_predictions * weights).sum(dim=0) / weights_sum
                average_predictions.append(batch_predictions.detach().clone().cpu())
        average_predictions = torch.vstack(average_predictions).requires_grad_(False)
        all_images = torch.vstack(all_images).requires_grad_(False)

        dataset = torch.utils.data.TensorDataset(all_images, average_predictions)
        trainsize = int(len(dataset) * 0.8)
        valsize = len(dataset) - trainsize
        trainset, valset = torch.utils.data.random_split(dataset, [trainsize, valsize])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, pin_memory=True,
                                                  shuffle=True, drop_last=True, num_workers=4)
        valloader = torch.utils.data.DataLoader(valset, batch_size=128, pin_memory=True,
                                                shuffle=False, drop_last=True, num_workers=4)
        return trainloader, valloader

    @aggregate_fit_wrapper
    def aggregate_fit(self, server_round: int, results, failures):
        client_capacities = self._map_client_to_capacities([client for client, _ in results])
        grouped_updated_models = {k: [] for k in self.available_model_capacities}
        teacher_models, teacher_capacities = [], []
        for capacity, (_, fit_res) in zip(client_capacities, results):
            model_array = parameters_to_ndarrays(fit_res.parameters)
            grouped_updated_models[capacity].append(
                (model_array, fit_res.num_examples)
            )
            tm = init_model(capacity, self.n_classes, self.device, self.dataset_name)
            set_parameters(tm, model_array)
            teacher_models.append(tm)
            teacher_capacities.append(capacity)

        aggregated_models_dict = {}
        for capacity in self.available_model_capacities:
            if len(grouped_updated_models[capacity]) == 0:
                print(f"Using old value as {capacity} has not been trained")
                aggregated_models_dict[capacity] = self.model_arrays[capacity]
            else:
                aggregated_models_dict[capacity] = aggregate(grouped_updated_models[capacity])

        # perform KD
        updated_models_list = []
        teacher_dataloader, teacher_valloader = self._init_dataloader(teacher_models, teacher_capacities) \
            if self.kd_epochs > 0 else (None, None)
        del teacher_models  # this way they don't consume any more memory on GPU
        for capacity in self.available_model_capacities:
            print(f"KD model with capacity {capacity}")
            student_model = init_model(capacity, self.n_classes, "cpu", self.dataset_name)
            set_parameters(student_model, aggregated_models_dict[capacity])
            optimization_config = OptimizationConfig(
                model=student_model,
                dataloader=teacher_dataloader,
                optimizer_name=self.kd_optimizer,
                epochs=self.kd_epochs,
                lr=self.kd_lr,
                device=self.device,
            )
            train_feddf(
                optimization_config=optimization_config,
                temperature=self.kd_temperature,
                valloader=teacher_valloader
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
