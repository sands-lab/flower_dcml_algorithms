import random

from numpy import ndarray
import torch
from flwr.common import GetParametersRes

from slower.server.server_model.numpy_server_model import NumPyServerModel

from src.helper.commons import read_json
from src.helper.optimization_config import init_optimizer
from src.helper.parameters import get_parameters, set_parameters
from src.helper.filepaths import FilePaths as FP
from src.models.helper import simple_init_model_from_string


class SlServerModel(NumPyServerModel):

    def __init__(self, dataset_name, seed, n_classes, sl_configuration) -> None:
        super().__init__()
        assert sl_configuration in {"plain", "u"}
        self.sl_configuration = sl_configuration
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Server uses {self.device}")
        self.model_name = read_json(
            FP.SL_MODEL_CONFIG,
            [self.dataset_name, self.sl_configuration, "server_model"]
        )

        self.model = simple_init_model_from_string(self.model_name, n_classes).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss() if self.sl_configuration == "plain" else None

    def serve_prediction_request(self, embeddings) -> bytes:
        embeddings = torch.from_numpy(embeddings).to(self.device)
        with torch.no_grad():
            preds = self.model(embeddings)
        if self.sl_configuration == "plain":
            preds = torch.argmax(preds, axis=1)
        return preds.cpu().numpy()

    def serve_gradient_update_request(self, embeddings, labels) -> bytes:
        assert self.sl_configuration == "plain"
        embeddings = torch.from_numpy(embeddings).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)
        embeddings.requires_grad_(True)

        preds = self.model(embeddings)
        loss = self.criterion(preds, labels)

        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
        self.optimizer.step()

        return embeddings.grad.detach().cpu().numpy()

    def u_forward(self, embeddings: ndarray) -> ndarray:
        assert self.sl_configuration == "u"
        self.client_embeddings = torch.from_numpy(embeddings).to(self.device)
        self.client_embeddings.requires_grad_(True)
        self.server_embeddings = self.model(self.client_embeddings)
        return self.server_embeddings.detach().cpu().numpy()

    def u_backward(self, gradient: ndarray) -> ndarray:
        assert self.sl_configuration == "u"
        server_gradient = torch.from_numpy(gradient).to(self.device)
        self.model.zero_grad()
        self.server_embeddings.backward(server_gradient)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 4.0)
        self.optimizer.step()
        return self.client_embeddings.grad.detach().cpu().numpy()

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
        self.model.train()

    def configure_evaluate(self, parameters, config):
        _ = (config,)
        set_parameters(self.model, parameters)
        self.model.eval()


def server_trainer_fn(**kwargs):
    return SlServerModel(**kwargs)
