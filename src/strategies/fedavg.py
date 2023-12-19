from flwr.server.strategy import FedAvg as FlFedAvg


class FedAvg(FlFedAvg):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
