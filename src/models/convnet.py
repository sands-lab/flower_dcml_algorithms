from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.models.abstract_model import AbstratModel



class ConvNet(AbstratModel):
    def __init__(self, rate, whole_model_config) -> None:
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


class SmallConvNet(ConvNet):
    def __init__(self, n_classes, rate) -> None:
        self.whole_model_config = [3, 32, 64, 256, 128, n_classes]
        self.whole_model_config = self.get_reduced_model_config(0.2)
        super().__init__(rate, self.whole_model_config)

class MediumConvNet(ConvNet):
    def __init__(self, n_classes, rate) -> None:
        self.whole_model_config = [3, 32, 64, 256, 128, n_classes]
        self.whole_model_config = self.get_reduced_model_config(0.5)
        super().__init__(rate, self.whole_model_config)

class LargeConvNet(ConvNet):
    def __init__(self, n_classes, rate) -> None:
        whole_model_config = [3, 32, 64, 256, 128, n_classes]
        super().__init__(rate, whole_model_config)


################################################################################################
################################################################################################
######### THE FOLLOWING MODELS SHOULD ONLY BE USED FOR REPRODUCING RESULTS FROM PAPERS #########
################################################################################################
################################################################################################


class MnistFDNet(nn.Module):
    def __init__(self, n_classes, ratio):
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


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
        print(self.rate)

    def forward(self, inp):
        """Forward of Scalar nn.Module."""
        output = inp / self.rate  # if self.training else inp
        return output


class FedDropoutNet(AbstratModel):
    """Implementation of the model defined in STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
       and used in the Federated Dropout algorithm."""
    # https://github.com/saxenamudit/DL_RP5/blob/master/pytorchModelsPythonFiles/ModelC.py
    def __init__(self, n_classes, rate) -> None:
        whole_model_config = [3, 96, 96, 192, 192, 192, 192, n_classes]
        super().__init__(whole_model_config, rate=rate)

        self.conv1_1 = nn.Conv2d(self.model_config[0], self.model_config[1], 3, padding=1)
        self.conv1_2 = nn.Conv2d(self.model_config[1], self.model_config[2], 3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(self.model_config[2], self.model_config[3], 3, padding=1)
        self.conv2_2 = nn.Conv2d(self.model_config[3], self.model_config[4], 3, padding=1)
        self.conv3 = nn.Conv2d(self.model_config[4], self.model_config[5], 3, padding=1)
        self.conv4 = nn.Conv2d(self.model_config[5], self.model_config[6], 3, padding=1)
        self.conv5 = nn.Conv2d(self.model_config[6], self.model_config[7], 1)

        self.relu = nn.ReLU()
        self.global_pooling = nn.AvgPool2d(8)
        self.flatten = nn.Flatten(start_dim=1)
        self.layer_names = self.get_ordered_layer_names()
        self.scaler = Scaler(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scaler(self.relu(self.conv1_1(x)))
        x = self.scaler(self.relu(self.conv1_2(x)))
        x = self.max_pool(x)
        x = self.scaler(self.relu(self.conv2_1(x)))
        x = self.scaler(self.relu(self.conv2_2(x)))
        x = self.max_pool(x)
        x = self.scaler(self.relu(self.conv3(x)))
        x = self.scaler(self.relu(self.conv4(x)))
        x = self.scaler(self.relu(self.conv5(x)))
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




class FedAvgMnistNet(AbstratModel):
    # MNIST in federated dropout
    def __init__(self, n_classes, rate) -> None:
        whole_model_config = [1, 32, 64, 512, n_classes]
        super().__init__(whole_model_config, rate)
        self.flatten_expansion = 7 * 7
        self.conv1 = nn.Conv2d(self.model_config[0], self.model_config[1], 5, padding=1)
        self.conv2 = nn.Conv2d(self.model_config[1], self.model_config[2], 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(self.model_config[2] * self.flatten_expansion, self.model_config[3])
        self.fc2 = nn.Linear(self.model_config[3], self.model_config[4])
        self.scaler = nn.Identity(rate)
        self.layer_names = self.get_ordered_layer_names()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def expand_configuration_to_model(self, idx_config):
        idx_config = [torch.tensor(sorted(c)) for c in idx_config]
        flattened_layers = [a for a in range(self.flatten_expansion * self.whole_model_config[2])
                            if a // self.flatten_expansion  in idx_config[2]]

        expanded_config = {
            "conv1": (idx_config[1], idx_config[0]),
            "conv2": (idx_config[2], idx_config[1]),
            "fc1": (idx_config[3], torch.tensor(flattened_layers)),
            "fc2": (idx_config[4], idx_config[3]),
        }
        return expanded_config



class VGG9(nn.Module):
    def __init__(self, n_classes, rate):
        super().__init__()
        _ = (rate,)
        conv_conf = [
            3, 32, 64, "M", 128, 128, "M", "D5", 256, 256, "M", "D10"
        ]
        layers = []
        channels = (None, conv_conf[0])
        for i in range(1, len(conv_conf) - 1):
            current_val = conv_conf[i]
            if isinstance(current_val, int):
                channels = (channels[1], current_val)
                layers.append(
                    nn.Conv2d(*channels, kernel_size=3, stride=1, padding=1)
                )
                layers.append(
                    nn.ReLU()
                )
            else:
                if current_val == "M":
                    layers.append(
                        nn.MaxPool2d(kernel_size=2, stride=2)
                    )
                elif current_val.startswith("D"):
                    prob = int(current_val[1:]) / 10
                    layers.append(
                        nn.Dropout(p=prob)
                    )
        head = [
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        ]
        head = nn.Sequential(*head)
        conv = nn.Sequential(*layers)
        model = nn.Sequential(conv, nn.Flatten(start_dim=1), head)
        self.model = model

    def forward(self, x):
        return self.model(x)
