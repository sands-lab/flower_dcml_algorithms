from flwr.common import ndarrays_to_parameters
import numpy as np

from src.strategies.fedavg import FedAvg


class PrivateTrainer(FedAvg):

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters([np.empty((0,))])
