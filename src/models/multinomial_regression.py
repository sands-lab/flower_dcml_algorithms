import torch
import torch.nn as nn


class MultinomialRegression(nn.Module):

    def __init__(self, n_classes, input_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.linear(torch.flatten(x))
