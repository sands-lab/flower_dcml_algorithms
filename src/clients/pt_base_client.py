import inspect

import numpy as np

from src.clients.fedavg_client import FedAvgClient
from src.models.helper import init_pt_model


class PtClient(FedAvgClient):

    def __init__(self, evaluate_on_whole_model, constant_rate, **kwargs):
        self.evaluate_on_whole_model = evaluate_on_whole_model
        self.constant_rate = constant_rate
        super().__init__(**kwargs)

    def _init_model(self):
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        is_evaluate = calframe[1][3] == "evaluate"

        capacity = "0" if self.evaluate_on_whole_model and is_evaluate else self.client_capacity
        rate = 1.0 if self.evaluate_on_whole_model and is_evaluate else self.constant_rate
        self.model = init_pt_model(
            client_capacity=capacity,
            n_classes=self.n_classes,
            device=self.device,
            dataset=self.dataset_name,
            rate=rate
        )

    def get_optimization_config(self, trainloader, config):
        kwargs = {}
        if int(self.client_capacity) > 0:
            kwargs["grad_norm_clipping_param"] = 1.0
        return super().get_optimization_config(trainloader, config, **kwargs)

    def evaluate(self, parameters, config):
        assert all(np.isfinite(p).all() for p in parameters)
        if self.evaluate_on_whole_model:
            self._init_model()

        if not self.stateful_client:
            self.set_parameters(self.model, parameters)
        return self._evaluate()
