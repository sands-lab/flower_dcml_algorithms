from flwr.server.client_manager import ClientManager
import torch
import numpy as np
from flwr.common import FitIns, ndarrays_to_parameters

from src.helper.filepaths import FilePaths as FP
from src.models.helper import simple_init_model_from_string
from src.helper.commons import read_json
from src.helper.parameters import get_parameters

from slower.server.strategy import PlainSlStrategy


class SlStrategy(PlainSlStrategy):
    def __init__(self, evaluation_freq, single_training_client, n_classes, sl_configuration, **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.evaluation_freq = evaluation_freq
        self.single_training_client = single_training_client
        self.sl_configuration = sl_configuration
        self.client_order = []
        self.converged = False
        self._init_models()

    def _init_models(self):
        # init model on the server for reproducibility
        # it's a temporary fix until we fix the seed issue
        config = read_json(FP.SL_MODEL_CONFIG, ["cifar10", self.sl_configuration])
        torch.manual_seed(10)
        client_parameteres = get_parameters(
            simple_init_model_from_string(config["client_model"], self.n_classes)
        )
        server_parameteres = get_parameters(
            simple_init_model_from_string(config["server_model"], self.n_classes)
        )

        if self.sl_configuration == "u":
            client_parameteres += get_parameters(
                simple_init_model_from_string(config["client_head"], self.n_classes)
            )
            print(client_parameteres[-1])
        else:
            print(server_parameteres[-1])
        self.client_parameters, self.server_parameters = client_parameteres, server_parameteres

    def initialize_client_parameters(self, client_manager: ClientManager):
        _ = (client_manager,)
        cp = self.client_parameters
        del self.client_parameters
        return ndarrays_to_parameters(cp)

    def initialize_server_parameters(self):
        sp = self.server_parameters
        del self.server_parameters
        return ndarrays_to_parameters(sp)

    def _initialize_client_training_order(self, client_manager):
        assert self.single_training_client
        n_clients = len(client_manager.all())
        self.client_order = np.random.permutation(n_clients).tolist()
        print(f"Generated following client order: {self.client_order}")

    def configure_client_fit(self, server_round, parameters, client_manager):
        if self.converged:
            return []
        if not self.single_training_client:
            return super().configure_client_fit(server_round, parameters, client_manager)
        # only one client is supposed to be training at a time as in the vanilla SL framework
        # create a random permutation if there is not one available. note, that here we assume
        # that clients will remain available for the duration of the experiment
        if len(self.client_order) == 0:
            self._initialize_client_training_order(client_manager)

        # use the first client for training, then remove it from list
        training_client_cid = str(self.client_order.pop(0))
        training_client = client_manager.all()[training_client_cid]

        config = {}
        if self.config_client_fit_fn:
            config = self.config_client_fit_fn(server_round)
        fit_ins = FitIns(parameters, config)
        return [(training_client, fit_ins)]

    def configure_client_evaluate(
        self, server_round, parameters, client_manager
    ):
        eval_ins = []
        if server_round % self.evaluation_freq == 0 and not self.converged:
            eval_ins = super().configure_client_evaluate(server_round, parameters, client_manager)
        return eval_ins

    def set_converged(self):
        self.converged = True

    def set_dataset_name(self, dataset_name):
        # just for compatibility
        _ = (dataset_name,)
        pass
