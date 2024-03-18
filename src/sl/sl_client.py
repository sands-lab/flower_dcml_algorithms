import os

import torch
from flwr.common import (
    FitRes,
    EvaluateRes,
    NDArrays,
)

from slower.server.server_model_segment.proxy.server_model_segment_proxy import (
    ServerModelSegmentProxy
)
from slower.client.numpy_client import NumPyClient
from slower.common import (
    torch_to_bytes,
    bytes_to_torch
)

from src.helper.commons import read_json, set_seed
from src.helper.optimization_config import init_optimizer
from src.helper.parameters import get_parameters, set_parameters
from src.helper.filepaths import FilePaths as FP
from src.data.helper import init_dataset
from src.models.helper import simple_init_model_from_string
from src.data.dataset_partition import DatasetPartition


class SlClient(NumPyClient):

    def __init__(
        self,
        cid,
        images_folder,
        partition_folder,
        seed,
        experiment_folder,
        client_capacity,
        separate_val_test_sets
    ):
        self.cid = cid
        self.images_folder = images_folder
        self.partition_folder = partition_folder
        self.seed = seed  # need to handle this later
        set_seed(self.seed)
        self.experiment_folder = experiment_folder
        self.dataset_name = os.path.split(images_folder)[-1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = read_json(FP.SL_MODEL_CONFIG, [self.dataset_name, "client_model"])
        self.model = simple_init_model_from_string(self.model_name, None).to(self.device)
        self.client_capacity = client_capacity
        self.separate_val_test_sets = separate_val_test_sets

    def get_parameters(self, config) -> NDArrays:
        return get_parameters(self.model)

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

    def fit(
        self,
        parameters,
        server_model_segment_proxy: ServerModelSegmentProxy,
        config,
    ) -> FitRes:
        print(f"Fitting client {self.cid} {self.model.__class__.__name__}")
        set_parameters(self.model, parameters)
        self.model.train()
        # print("starting training....", self.get_parameters({})[1][:3])
        optimizer = init_optimizer(
            self.model.parameters(),
            optimizer_name=config["optimizer"],
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 3e-4)
        )
        dataloader = self._init_dataloader(DatasetPartition.TRAIN, config["batch_size"])
        set_seed(self.seed)
        for _ in range(config["local_epochs"]):
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                embeddings = self.model(images)

                error = server_model_segment_proxy.serve_gradient_update_request_wrapper(
                    embeddings=torch_to_bytes(embeddings),
                    labels=torch_to_bytes(labels),
                    timeout=None
                )
                error = bytes_to_torch(error, False)
                error = error.to(self.device)

                self.model.zero_grad()
                embeddings.backward(error)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
                optimizer.step()
        # print("finished training....", self.get_parameters({})[1][:3])
        return self.get_parameters({}), len(dataloader.dataset), {}

    def _get_accuracy(self, dataloader, server_model_segment_proxy):
        correct = 0
        for images, labels in dataloader:
            images = images.to(self.device)

            with torch.no_grad():
                embeddings = self.model(images)
            preds = server_model_segment_proxy.serve_prediction_request_wrapper(
                embeddings=torch_to_bytes(embeddings),
                timeout=None
            )
            preds = bytes_to_torch(preds, False).int()
            correct += (preds == labels).int().sum()  # compute this on cpu
        # print("finished evaluation....", self.get_parameters({})[1][:3])

        accuracy = float(correct / len(dataloader.dataset))
        return accuracy

    def evaluate(
        self,
        parameters,
        server_model_segment_proxy: ServerModelSegmentProxy,
        config
    ) -> EvaluateRes:
        self.model.eval()
        set_parameters(self.model, parameters)

        out_dict = {"client_id": self.cid, "client_capacity": self.client_capacity}
        if self.separate_val_test_sets:
            testloader = self._init_dataloader(DatasetPartition.TEST, 32)
            accuracy = self._get_accuracy(testloader, server_model_segment_proxy)
            out_dict["test_accuracy"] = accuracy
            del testloader

            valloader = self._init_dataloader(DatasetPartition.VAL, 32)
            accuracy = self._get_accuracy(valloader, server_model_segment_proxy)
            out_dict["accuracy"] = accuracy
            dataset_size = len(valloader.dataset)
        else:
            testloader = self._init_dataloader(DatasetPartition.TEST, 32)
            accuracy = self._get_accuracy(testloader, server_model_segment_proxy)
            out_dict["accuracy"] = accuracy
            dataset_size = len(testloader.dataset)
        # print("starting evaluation....", self.get_parameters({})[1][:3])

        return accuracy, dataset_size, out_dict


def client_fn(cid, **kwargs) -> SlClient:
    # print("client fn")
    return SlClient(cid=int(cid), **kwargs)
