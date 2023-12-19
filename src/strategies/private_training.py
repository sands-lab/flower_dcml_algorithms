from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg
import numpy as np


class PrivateTrainer(FedAvg):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters([np.empty((0,))])
