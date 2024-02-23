import typing

import numpy as np
import torch
import torch.nn as nn

from src.helper.parameters import get_parameters
from src.models.abstract_model import AbstratModel


def fill_missing_model_parts(previous_model, out_params):
    cleaned_parameters = []

    for previous_value, updated_value in zip(get_parameters(previous_model), out_params):
        nv = np.where(updated_value.isfinite(), updated_value, previous_value)
        cleaned_parameters.append(nv)
    return cleaned_parameters

def aggregate_submodels(
        previous_model: AbstratModel,
        submodels_list: typing.List[AbstratModel],
        submodels_configs: typing.List[typing.List[np.ndarray]]
):
    layers = dict(previous_model.named_modules())
    out_params = []
    for layer_name in previous_model.layer_names:
        layer = layers[layer_name]
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            # weight/bias sum/count
            ws, bs = torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
            wc, bc = torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)

            for submodel, submodel_conf in zip(submodels_list, submodels_configs):
                submodel_weight = submodel.state_dict()[f"{layer_name}.weight"]
                submodel_bias = submodel.state_dict()[f"{layer_name}.bias"]

                submodel_layer_conf = submodel_conf[layer_name]
                idxs = torch.meshgrid(submodel_layer_conf, indexing="ij")
                ws[idxs] += submodel_weight
                wc[idxs] += 1

                bs[submodel_layer_conf[0]] += submodel_bias
                bc[submodel_layer_conf[0]] += 1
            out_params.append(ws / wc)
            out_params.append(bs / bc)
    return fill_missing_model_parts(previous_model, out_params)
