import numpy as np

from src.strategies.pt import PT


class HeteroFL(PT):
    def configure_client_submodel(self, whole_model_config, reduced_model_config):
        return [np.arange(conf) for conf in reduced_model_config]
