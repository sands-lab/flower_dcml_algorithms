from collections import OrderedDict

from src.clients.plft import PLFT


class LgFedAvgClient(PLFT):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_model(self, model):
        model_lst = list(model.state_dict().items())
        private_model = model_lst[:-self.n_public_layers]
        public_model = model_lst[-self.n_public_layers:]
        return OrderedDict(private_model), OrderedDict(public_model)


def client_fn(cid, **kwargs) -> LgFedAvgClient:
    return LgFedAvgClient(cid=int(cid), **kwargs)
