import torch.nn as nn

from src.models.fully_conv_net import WHOLE_MODEL_CONFIG
from src.models.abstract_model import get_reduced_model_config


class CommonHead(nn.Sequential):

    def __init__(self, n_classes, rate) -> None:
        _ = (n_classes, rate)
        model_config = get_reduced_model_config(WHOLE_MODEL_CONFIG, 0.2)
        super().__init__(
            nn.Conv2d(model_config[5], model_config[6], 3),
            nn.ReLU(),
            nn.Conv2d(model_config[6], model_config[7], 1),
            nn.ReLU(),
            nn.AvgPool2d(6),
            nn.Flatten(1)
        )


class Encoder(nn.Sequential):

    def __init__(self, rate):
        model_config = get_reduced_model_config(WHOLE_MODEL_CONFIG, rate)
        super().__init__(
            nn.Conv2d(model_config[0], model_config[1], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(model_config[1], model_config[2], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(model_config[2], model_config[3], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(model_config[3], model_config[4], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(model_config[4], 39, 3, padding=1),
            nn.ReLU()
        )


class SmallEncoder(Encoder):
    def __init__(self, n_classes, rate):
        super().__init__(0.2)


class MediumEncoder(Encoder):
    def __init__(self, n_classes, rate):
        _ = (n_classes, rate)
        super().__init__(0.5)


class LargeEncoder(Encoder):
    def __init__(self, n_classes, rate):
        _ = (n_classes, rate)
        super().__init__(1.0)
