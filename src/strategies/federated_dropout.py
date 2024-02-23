import numpy as np

from src.strategies.pt import PT


class FederatedDropout(PT):

    def configure_client_submodel(self, idx, whole_model_config, reduced_model_config):
        config = [np.sort(np.random.choice(whole_conf, reduced_conf, replace=False))
                  for whole_conf, reduced_conf in zip(whole_model_config, reduced_model_config)]
        return config
