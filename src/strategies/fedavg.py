from flwr.server.strategy import FedAvg

from fltb.decorators import MonitorFlwrStrategy


@MonitorFlwrStrategy
class FedAvg(FedAvg):

    def __init__(self, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
