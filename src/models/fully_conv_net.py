import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.abstract_model import AbstratModel, get_reduced_model_config



WHOLE_MODEL_CONFIG = [3, 96, 96, 192, 192, 192, 192, 10]


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, inp):
        """Forward of Scalar nn.Module."""
        output = inp / self.rate if self.train else inp
        return output


class AllConvNet(AbstratModel):
    def __init__(self, model_config, rate, **kwargs):

        super().__init__(model_config, rate)
        self.whole_model_config = copy.copy(model_config)
        self.conv1_1 = nn.Conv2d(self.model_config[0], self.model_config[1], 3, padding=1)
        self.conv1_2 = nn.Conv2d(self.model_config[1], self.model_config[2], 3, padding=1)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(self.model_config[2], self.model_config[3], 3, padding=1)
        self.conv2_2 = nn.Conv2d(self.model_config[3], self.model_config[4], 3, padding=1)
        self.conv3 = nn.Conv2d(self.model_config[4], self.model_config[5], 3, padding=1)
        self.conv4 = nn.Conv2d(self.model_config[5], self.model_config[6], 3)
        self.conv5 = nn.Conv2d(self.model_config[6], self.model_config[7], 1)
        self.scaler = Scaler(rate)
        self.relu = nn.ReLU(inplace=True)
        self.global_pooling = nn.AvgPool2d(6)
        self.layer_names = self.get_ordered_layer_names()
        self.flatten = nn.Flatten(start_dim=1)


    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.scaler(x)
        x = self.relu(self.conv1_2(x))
        x = self.scaler(x)
        x = self.max_pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.scaler(x)
        x = self.relu(self.conv2_2(x))
        x = self.scaler(x)
        x = self.max_pool(x)

        x = self.relu(self.conv3(x))
        x = self.scaler(x)

        x = self.relu(self.conv4(x))
        x = self.scaler(x)

        x = self.relu(self.conv5(x))
        x = self.scaler(x)
        x = self.global_pooling(x)
        x = self.flatten(x)
        return x

    def expand_configuration_to_model(self, idx_config):
        idx_config = [torch.tensor(sorted(c)) for c in idx_config]
        expanded_config = {
            "conv1_1": (idx_config[1], idx_config[0]),
            "conv1_2": (idx_config[2], idx_config[1]),
            "conv2_1": (idx_config[3], idx_config[2]),
            "conv2_2": (idx_config[4], idx_config[3]),
            "conv3": (idx_config[5], idx_config[4]),
            "conv4": (idx_config[6], idx_config[5]),
            "conv5": (idx_config[7], idx_config[6]),
        }
        return expanded_config



class MicroAllConvNet(AllConvNet):
    def __init__(self, n_classes, rate):
        model_config = copy.deepcopy(WHOLE_MODEL_CONFIG)[:-1] + [n_classes]
        model_config = get_reduced_model_config(model_config, 0.1)
        super().__init__(model_config, rate)


class SmallAllConvNet(AllConvNet):
    def __init__(self, n_classes, rate):
        model_config = copy.deepcopy(WHOLE_MODEL_CONFIG)[:-1] + [n_classes]
        model_config = get_reduced_model_config(model_config, 0.2)
        super().__init__(model_config, rate)


class MediumAllConvNet(AllConvNet):
    def __init__(self, n_classes, rate):
        model_config = copy.deepcopy(WHOLE_MODEL_CONFIG)[:-1] + [n_classes]
        model_config = get_reduced_model_config(model_config, 0.5)
        super().__init__(model_config, rate)


class LargeAllConvNet(AllConvNet):
    def __init__(self, n_classes, rate):
        model_config = copy.deepcopy(WHOLE_MODEL_CONFIG)[:-1] + [n_classes]
        model_config = get_reduced_model_config(model_config, 1.0)
        super().__init__(model_config, rate)
