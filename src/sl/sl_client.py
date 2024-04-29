import os

import torch
from flwr.common import (
    FitRes,
    EvaluateRes,
    NDArrays,
)

from slower.client.numpy_client import NumPyClient

from src.helper.commons import read_json, set_seed, sync_rng_state, save_rng_state_if_not_exists
from src.helper.optimization_config import OptimizationConfig
from src.helper.parameters import get_parameters, set_parameters
from src.helper.filepaths import FilePaths as FP
from src.data.helper import init_dataset
from src.models.helper import simple_init_model_from_string
from src.data.dataset_partition import DatasetPartition
from src.sl.training_procedures import train_model, train_u_model


class SlClient(NumPyClient):

    def __init__(
        self,
        cid,
        images_folder,
        partition_folder,
        seed,
        experiment_folder,
        client_capacity,
        separate_val_test_sets,
        sl_configuration # whether to use the plain SL or the U-shaped architecture. ether `plain` or `u`
    ):
        assert sl_configuration in {"u", "plain"}
        self.cid = int(cid)
        self.images_folder = images_folder
        self.dataset_name = os.path.split(images_folder)[-1]
        self.partition_folder = partition_folder
        self.experiment_folder = experiment_folder
        self.seed = seed  # need to handle this later
        self.n_classes = read_json(FP.DATA_CONFIG, [self.dataset_name, "n_classes"])
        self.client_working_folder = f"{experiment_folder}/{cid}"
        self.client_capacity = client_capacity
        self.separate_val_test_sets = separate_val_test_sets
        self.sl_configuration = sl_configuration
        if not os.path.isdir(self.client_working_folder):
            os.mkdir(self.client_working_folder)

        self.dataset_name = os.path.split(images_folder)[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if sl_configuration == "plain":
            self.model_name = read_json(
                FP.SL_MODEL_CONFIG,
                [self.dataset_name, sl_configuration, "client_model"]
            )
            self.encoder = simple_init_model_from_string(self.model_name, self.n_classes).to(self.device)
        else:
            self.model_name = read_json(FP.SL_MODEL_CONFIG, [self.dataset_name, sl_configuration])
            self.encoder = \
                simple_init_model_from_string(self.model_name["client_model"], self.n_classes).to(self.device)
            self.clf_head = \
                simple_init_model_from_string(self.model_name["client_head"], self.n_classes).to(self.device)

        set_seed(self.seed)
        save_rng_state_if_not_exists(self.client_working_folder)

    def get_parameters(self, config) -> NDArrays:
        _ = (config,)
        params = get_parameters(self.encoder)
        if self.sl_configuration == "u":
            params += get_parameters(self.clf_head)
        return params

    def _init_dataloader(self, partition, batch_size):
        dataset = init_dataset(
            cid=self.cid,
            dataset_partition=partition,
            dataset_name=self.dataset_name,
            partition_folder=self.partition_folder,
            images_folder=self.images_folder
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=partition == DatasetPartition.TRAIN,
            batch_size=batch_size,
            pin_memory=self.device.type == "cuda"
        )
        return dataloader

    def set_parameters(self, parameters):
        if self.sl_configuration == "plain":
            set_parameters(self.encoder, parameters=parameters)
        else:
            n_encoder_layers = len(self.encoder.state_dict())
            set_parameters(self.encoder, parameters=parameters[:n_encoder_layers])
            set_parameters(self.clf_head, parameters=parameters[n_encoder_layers:])

    @sync_rng_state
    def fit(self, parameters, config) -> FitRes:
        print(f"Fitting client {self.cid} {self.encoder.__class__.__name__}")
        self.set_parameters(parameters)
        dataloader = self._init_dataloader(DatasetPartition.TRAIN, config["batch_size"])

        if self.sl_configuration == "plain":
            model = self.encoder
        else:
            model = torch.nn.ModuleDict({"encoder": self.encoder, "clf_head": self.clf_head})

        optimization_config = OptimizationConfig(
            model=model,
            dataloader=dataloader,
            lr=config["lr"],
            epochs=config["local_epochs"],
            optimizer_name=config["optimizer"],
            device=self.device,
            grad_norm_clipping_param=4.0,
            weight_decay=config.get("weight_decay", 3e-4),
        )

        if self.sl_configuration == "plain":
            train_model(optimization_config, self.server_model_proxy)
        else:
            train_u_model(optimization_config, self.server_model_proxy)

        return self.get_parameters({}), len(dataloader.dataset), {}

    def _get_accuracy(self, dataloader):
        correct = 0
        for images, labels in dataloader:
            images = images.to(self.device)

            with torch.no_grad():
                embeddings = self.encoder(images)
            preds = self.server_model_proxy.numpy_serve_prediction_request(
                embeddings=embeddings.cpu().numpy()
            )
            if self.sl_configuration == "u":
                preds = torch.from_numpy(preds).to(self.device)
                with torch.no_grad():
                    preds = self.clf_head(preds)
                preds = preds.argmax(dim=1).cpu().numpy()

            correct += (preds == labels.numpy()).sum()  # compute this on cpu

        accuracy = float(correct / len(dataloader.dataset))
        return accuracy

    @sync_rng_state
    def evaluate(
        self,
        parameters,
        config
    ) -> EvaluateRes:
        _ = (config,)
        self.encoder.eval()
        self.set_parameters(parameters)

        out_dict = {"client_id": self.cid, "client_capacity": self.client_capacity}
        if self.separate_val_test_sets:
            testloader = self._init_dataloader(DatasetPartition.TEST, 32)
            accuracy = self._get_accuracy(testloader)
            out_dict["test_accuracy"] = accuracy
            del testloader

            valloader = self._init_dataloader(DatasetPartition.VAL, 32)
            accuracy = self._get_accuracy(valloader)
            out_dict["accuracy"] = accuracy
            dataset_size = len(valloader.dataset)
        else:
            testloader = self._init_dataloader(DatasetPartition.TEST, 32)
            accuracy = self._get_accuracy(testloader)
            out_dict["accuracy"] = accuracy
            dataset_size = len(testloader.dataset)
        # print("starting evaluation....", self.get_parameters({})[1][:3])

        return accuracy, dataset_size, out_dict

    def get_properties(self, config):
        return {
            "client_capacity": 0
        }


def client_fn(cid, **kwargs) -> SlClient:
    # print("client fn")
    return SlClient(cid=int(cid), **kwargs)
