from src.clients.plft import PLFT


class PerFedClient(PLFT):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_model_layer_names(self, model_names):
        return model_names[:self.n_public_layers], model_names[self.n_public_layers:]



def client_fn(cid, **kwargs) -> PerFedClient:
    return PerFedClient(cid=int(cid), **kwargs)
