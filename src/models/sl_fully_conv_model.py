import torch.nn as nn


WHOLE_MODEL_CONFIG = [3, 64, 96, 192, 192, 192, 192, 10]


class SlEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(WHOLE_MODEL_CONFIG[0], WHOLE_MODEL_CONFIG[1], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(WHOLE_MODEL_CONFIG[1], WHOLE_MODEL_CONFIG[2], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )


class SlClfHead(nn.Sequential):
    def __init__(self, n_classes):
        super().__init__(
            nn.Conv2d(WHOLE_MODEL_CONFIG[2], WHOLE_MODEL_CONFIG[3], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(WHOLE_MODEL_CONFIG[3], WHOLE_MODEL_CONFIG[4], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(WHOLE_MODEL_CONFIG[4], WHOLE_MODEL_CONFIG[5], 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(WHOLE_MODEL_CONFIG[5], WHOLE_MODEL_CONFIG[6], 3),
            nn.ReLU(),
            nn.Conv2d(WHOLE_MODEL_CONFIG[6], n_classes, 1),
            nn.ReLU(),
            nn.AvgPool2d(6),
            nn.Flatten(start_dim=1)
        )


class WholeModel(nn.Sequential):
    def __init__(self, n_classes, rate):
        _ = (rate,)
        super().__init__(
            SlEncoder(),
            SlClfHead(n_classes)
        )
