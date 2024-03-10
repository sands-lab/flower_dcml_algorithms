import torch
import torch.nn as nn

from src.models.abstract_model import AbstratModel, get_reduced_model_config


WHOLE_MODEL_CONFIG = [1, 64, 128, 256, 256, 10]

class MnistConvNet(AbstratModel):
    def __init__(self, model_config, rate) -> None:
        super().__init__(model_config, rate)
        self.model = nn.Sequential(
            nn.Conv2d(self.model_config[0], self.model_config[1], kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.model_config[1], self.model_config[2], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.model_config[2], self.model_config[3], kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten(),
            nn.Linear(self.model_config[3], self.model_config[4]),
            nn.ReLU(),
            nn.Linear(self.model_config[4], self.model_config[5])
        )
        self.layer_names = self.get_ordered_layer_names()

    def forward(self, x):
        return self.model(x)

    def expand_configuration_to_model(self, idx_config):
        idx_config = [torch.tensor(sorted(c)) for c in idx_config]
        expanded_config = {
            "model.0": (idx_config[1], idx_config[0]),
            "model.3": (idx_config[2], idx_config[1]),
            "model.6": (idx_config[3], idx_config[2]),
            "model.10": (idx_config[4], idx_config[3]),
            "model.12": (idx_config[5], idx_config[4]),
        }
        return expanded_config


class SmallMnistConvNet(MnistConvNet):
    def __init__(self, n_classes, rate) -> None:
        model_config = get_reduced_model_config(WHOLE_MODEL_CONFIG, 0.1)
        super().__init__(model_config, rate)


class MediumMnistConvNet(MnistConvNet):
    def __init__(self, n_classes, rate) -> None:
        model_config = get_reduced_model_config(WHOLE_MODEL_CONFIG, 0.4)
        super().__init__(model_config, rate)


class LargeMnistConvNet(MnistConvNet):
    def __init__(self, n_classes, rate) -> None:
        model_config = get_reduced_model_config(WHOLE_MODEL_CONFIG, 1.0)
        super().__init__(model_config, rate)
