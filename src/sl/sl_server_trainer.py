import random

import torch
from flwr.common import GetParametersRes

from slower.common import (
    torch_to_bytes,
    bytes_to_torch
)
from slower.server.server_model_segment.numpy_server_model_segment import NumPyServerModelSegment

from src.helper.commons import read_json, set_seed
from src.helper.optimization_config import init_optimizer
from src.helper.parameters import get_parameters, set_parameters
from src.helper.filepaths import FilePaths as FP
from src.models.helper import simple_init_model_from_string


class SlServerSegment(NumPyServerModelSegment):

    def __init__(self, dataset_name, seed, n_classes) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = read_json(FP.SL_MODEL_CONFIG, [self.dataset_name, "server_model"])
        set_seed(seed * 2 + 1)  # seems to get better performance if we use different seeds on client and server
        self.model = simple_init_model_from_string(self.model_name, n_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

    def serve_prediction_request(self, embeddings) -> bytes:
        embeddings = bytes_to_torch(embeddings, False).to(self.device)
        with torch.no_grad():
            preds = self.model(embeddings)
        preds = torch.argmax(preds, axis=1)
        preds = torch_to_bytes(preds)
        return preds

    def serve_gradient_update_request(self, embeddings, labels) -> bytes:
        embeddings = bytes_to_torch(embeddings, False)
        embeddings = embeddings.to(self.device)
        embeddings.requires_grad_(True)
        labels = bytes_to_torch(labels, False).to(self.device)

        preds = self.model(embeddings)
        loss = self.criterion(preds, labels)

        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
        self.optimizer.step()

        error = torch_to_bytes(embeddings.grad)
        return error

    def get_parameters(self) -> GetParametersRes:
        parameters = get_parameters(self.model)
        return parameters

    def configure_fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.optimizer = init_optimizer(
            parameters=self.model.parameters(),
            optimizer_name=config["optimizer"],
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 3e-4)
        )
        # print(self.get_parameters({})[1][:5])
        self.model.train()

    def configure_evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()


def server_trainer_fn(**kwargs):
    return SlServerSegment(**kwargs)
