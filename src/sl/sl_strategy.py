import numpy as np
from flwr.common import FitIns

from slower.server.strategy import PlainSlStrategy


class SlStrategy(PlainSlStrategy):
    def __init__(self, evaluation_freq, single_training_client, **kwargs):
        super().__init__(**kwargs)
        self.evaluation_freq = evaluation_freq
        self.single_training_client = single_training_client
        self.client_order = []

    def _initialize_client_training_order(self, client_manager):
        assert self.single_training_client
        n_clients = len(client_manager.all())
        self.client_order = np.random.permutation(10).tolist()
        print(f"Generated following client order: {self.client_order}")

    def configure_client_fit(self, server_round, parameters, client_manager):
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
        if server_round % self.evaluation_freq == 0:
            eval_ins = super().configure_client_evaluate(server_round, parameters, client_manager)
        return eval_ins
