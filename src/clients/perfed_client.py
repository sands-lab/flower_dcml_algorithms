from collections import OrderedDict

from src.clients.plft import PLFT


class PerFedClient(PLFT):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_model(self, model):
        model_lst = list(model.state_dict().items())
        public_model = model_lst[:self.n_public_layers]
        private_model = model_lst[self.n_public_layers:]
        return OrderedDict(private_model), OrderedDict(public_model)


def client_fn(cid, **kwargs) -> PerFedClient:
    return PerFedClient(cid=int(cid), **kwargs)
