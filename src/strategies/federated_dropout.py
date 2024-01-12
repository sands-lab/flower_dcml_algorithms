import numpy as np

from src.strategies.pt import PT


class FederatedDropout(PT):

    def configure_client_submodel(self, whole_model_config, reduced_model_config):
        return [np.random.choice(whole_conf, reduced_conf, replace=False)
                for whole_conf, reduced_conf in zip(whole_model_config, reduced_model_config)]
