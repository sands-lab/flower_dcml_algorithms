import copy
import math
import typing

import torch
import torch.nn as nn

from src.helper.commons import get_numpy


def get_reduced_model_config(model_config, rate):
    model_config = copy.deepcopy(model_config)
    for i in range(1, len(model_config) - 1):
        model_config[i] = int(math.ceil(model_config[i] * rate))
    return model_config


class AbstratModel(nn.Module):
    def __init__(self, whole_model_config, rate) -> None:
        super().__init__()
        self.whole_model_config = whole_model_config
        self.model_config = \
            self.whole_model_config if rate == 1.0 else self.get_reduced_model_config(rate)
        self.rate = rate
        self.layer_names = None

    def get_reduced_model_config(self, rate):
        return get_reduced_model_config(self.whole_model_config, rate)

    def expand_configuration_to_model(self, config) -> typing.Dict[str, typing.List[int]]:
        raise NotImplementedError("Method should be implemented for every model separately")

    def _extract_data_from_layer(self, layer, config):
        if isinstance(layer, nn.Linear):
            wd = get_numpy(layer.weight[torch.meshgrid(config, indexing="ij")])
            bd = get_numpy(layer.bias[config[0]])

        elif isinstance(layer, nn.Conv2d):
            wd = get_numpy(layer.weight[torch.meshgrid(config, indexing="ij")])
            bd = get_numpy(layer.bias[config[0]])

        else:
            raise ValueError(f"Layer {layer} not supported")
        return wd, bd

    def extract_submodel_parameters(self, config):
        modules = dict(self.named_modules())

        out = []
        for layer_name in self.layer_names:
            layer = modules[layer_name]
            weight_data, bias_data = self._extract_data_from_layer(layer, config[layer_name])
            out.append(weight_data)
            out.append(bias_data)
        return out

    def get_ordered_layer_names(self):
        parameter_names = list(self.state_dict().keys())

        layer_names = []
        for i in range(0, len(parameter_names)-1, 2):
            assert parameter_names[i].endswith("weight")
            assert parameter_names[i+1].endswith("bias")
            layer_names.append(parameter_names[i][:-len(".weight")])
        return layer_names
