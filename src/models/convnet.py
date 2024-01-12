from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.abstract_model import AbstratModel



class ConvNet(AbstratModel):
    def __init__(self, n_classes, rate, whole_model_config=None) -> None:
        if whole_model_config is None:
            whole_model_config = [3, 6, 16, 120, 84, n_classes]
        super().__init__(whole_model_config, rate)

        self.conv1 = nn.Conv2d(self.model_config[0], self.model_config[1], 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.model_config[1], self.model_config[2], 5)

        # when we flatten the tensor, each channel gets mapped into 25 values
        self.flatten_expansion = 5 * 5
        self.fc1 = nn.Linear(self.model_config[2] * self.flatten_expansion, self.model_config[3])
        self.fc2 = nn.Linear(self.model_config[3], self.model_config[4])
        self.fc3 = nn.Linear(self.model_config[4], self.model_config[5])
        self.layer_names = self.get_ordered_layer_names()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.model_config[2] * self.flatten_expansion)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def expand_configuration_to_model(self, idx_config: List[np.ndarray]) -> Dict[str, List[int]]:
        idx_config = [torch.tensor(sorted(c)) for c in idx_config]
        flattened_layers = [a for a in range(self.flatten_expansion * self.whole_model_config[2])
                            if a // self.flatten_expansion  in idx_config[2]]
        expanded_config = {
            "conv1": (idx_config[1], idx_config[0]),
            "conv2": (idx_config[2], idx_config[1]),
            "fc1": (idx_config[3], torch.tensor(flattened_layers)),
            "fc2": (idx_config[4], idx_config[3]),
            "fc3": (idx_config[5], idx_config[4]),
        }
        return expanded_config


class LargeConvNet(ConvNet):
    def __init__(self, n_classes, rate) -> None:
        whole_model_config = [3, 32, 48, 168, 84, n_classes]
        super().__init__(n_classes, rate, whole_model_config)


################################################################################################
################################################################################################
######### THE FOLLOWING MODELS SHOULD ONLY BE USED FOR REPRODUCING RESULTS FROM PAPERS #########
################################################################################################
################################################################################################


class MnistFDNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
        self.fc1 = nn.Linear(9216, 128, bias=False)
        self.fc2 = nn.Linear(128, n_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class DS_FL_ConvNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        return self.model(x)


class LeNet5(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 100),
            nn.Tanh(),
            nn.Linear(100, n_classes)
        )

    def forward(self, x):
        return self.model(x)
