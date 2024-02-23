import copy
from typing import Dict, List
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.helper.parameters import set_parameters
from src.models.abstract_model import AbstratModel
from src.models.pt_commons import aggregate_submodels
from src.models.convnet import SmallConvNet, FedAvgMnistNet, FedDropoutNet


class MLP(AbstratModel):

    def __init__(self, in_features, hidden_features, out_features, ratio):
        whole_model_config = [in_features, hidden_features, out_features]
        super().__init__(whole_model_config, ratio)
        self.a = nn.Linear(self.model_config[0], self.model_config[1])
        self.b = nn.Linear(self.model_config[1], self.model_config[2])
        self.relu = nn.ReLU()
        self.layer_names = self.get_ordered_layer_names()

    def forward(self, x):
        x = self.a(x)
        x = self.relu(x)
        x = self.b(x)
        return x

    def expand_configuration_to_model(self, config) -> Dict[str, List[int]]:
        return {
            "a": (config[1], config[0]),
            "b": (config[2], config[1])
        }


class Conv(AbstratModel):

    def __init__(self, in_channels, hidden_channels, out_channels, ratio, use_sequential):
        whole_model_config = [in_channels, hidden_channels, out_channels]
        super().__init__(whole_model_config, ratio)
        a = nn.Conv2d(self.model_config[0], self.model_config[1], kernel_size=5, stride=1)
        b = nn.Conv2d(self.model_config[1], self.model_config[2], kernel_size=5, stride=1)
        relu = nn.ReLU()
        if use_sequential:
            self.model = nn.Sequential(a, relu, b)
        else:
            self.a = a
            self.b = b
            self.relu = relu
        self.layer_names = self.get_ordered_layer_names()
        self.use_sequential = use_sequential

    def forward(self, x):
        if self.use_sequential:
            return self.model(x)
        x = self.a(x)
        x = self.relu(x)
        x = self.b(x)
        return x

    def expand_configuration_to_model(self, config) -> Dict[str, List[int]]:
        return {
            self.layer_names[0]: (config[1], config[0]),
            self.layer_names[1]: (config[2], config[1])
        }


class ModelExtractionTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ratio = 0.3
        self.batch_size = 10

        # Conv layers
        self.input_resolution = 10
        self.input_channels = 3
        self.hidden_channels = 20
        self.out_channels = 4

        # MLP
        self.in_features = 10
        self.hidden_features = 20
        self.out_features = 4

    def test_extract_linear(self):
        model = MLP(self.in_features, self.hidden_features, self.out_features, 1.0)
        submodel = MLP(self.in_features, self.hidden_features, self.out_features, self.ratio)
        reduced_config = model.get_reduced_model_config(self.ratio)
        heterofl_reduced_config = [torch.arange(a) for a in reduced_config]

        expanded_config = model.expand_configuration_to_model(heterofl_reduced_config)
        submodel_params = model.extract_submodel_parameters(expanded_config)

        set_parameters(submodel, submodel_params)

        x = torch.randn(self.batch_size, self.in_features)
        sumbodel_pred = submodel(x)

        tmp = model.a(x)
        missing = [i for i in range(self.hidden_features) if i not in expanded_config["a"][0]]
        tmp[:, missing] = 0
        tmp = model.relu(tmp)
        self.assertTrue(
            bool(
                model.b(tmp).allclose(sumbodel_pred)
            )
        )

    def test_extract_conv(self):
        use_sequential = False
        model = Conv(self.input_channels, self.hidden_channels, self.out_channels,
                     ratio=1.0, use_sequential=use_sequential)

        submodel = Conv(self.input_channels, self.hidden_channels, self.out_channels,
                        self.ratio, use_sequential=use_sequential)
        reduced_config = model.get_reduced_model_config(self.ratio)
        heterofl_reduced_config = [torch.arange(a) for a in reduced_config]
        expanded_config = model.expand_configuration_to_model(heterofl_reduced_config)

        submodel_params = model.extract_submodel_parameters(expanded_config)
        set_parameters(submodel, submodel_params)
        x = torch.randn(self.batch_size, self.input_channels,
                        self.input_resolution, self.input_resolution)
        sumbodel_pred = submodel(x)

        missing = [i for i in range(self.hidden_channels)
                   if i not in expanded_config[model.layer_names[0]][0]]
        if use_sequential:
            tmp = model.model[0](x)
            tmp[:, missing] = 0
            p = model.model[1:](tmp)
        else:
            tmp = model.a(x)
            tmp[:, missing] = 0
            tmp = model.relu(tmp)
            p = model.b(tmp)
        self.assertTrue(p.allclose(sumbodel_pred))

    def test_aggregate_disjoint(self):
        model = MLP(self.in_features, self.hidden_features, self.out_features, 1.0)

        model.a.requires_grad_(False)
        model.a.weight.fill_(0.)
        model.a.bias.fill_(0.)

        submodel1 = MLP(self.in_features, self.hidden_features, self.out_features, 0.5)
        submodel2 = MLP(self.in_features, self.hidden_features, self.out_features, 0.25)
        config1 = {
            "a": (torch.arange(submodel1.model_config[1]), torch.arange(self.in_features)),
            "b": (torch.arange(self.out_features), torch.arange(submodel1.model_config[1]))
        }
        mn = submodel1.model_config[1]+1
        mx = submodel1.model_config[1]+1 + submodel2.model_config[1]
        tmp = torch.arange(mn, mx)
        assert tmp.shape[0] == submodel2.model_config[1]
        config2 = {
            "a": (tmp, torch.arange(self.in_features)),
            "b": (torch.arange(self.out_features), tmp)
        }
        updated_model_weights = \
            aggregate_submodels(model, [submodel1, submodel2], [config1, config2])
        set_parameters(model, updated_model_weights)

        assert submodel1.a.weight.allclose(model.a.weight[config1["a"][0]])
        assert submodel2.a.weight.allclose(model.a.weight[config2["a"][0]])

        missing = [k for k in range(self.hidden_features)
                   if k > submodel1.model_config[1] + submodel2.model_config[1]]
        assert (model.a.weight[missing] == 0.).all()

    def test_agg_overlapping(self):
        model = MLP(self.in_features, self.hidden_features, self.out_features, 1.0)

        model.a.requires_grad_(False)
        model.a.weight.fill_(0.)
        model.a.bias.fill_(0.)

        submodel1 = MLP(self.in_features, self.hidden_features, self.out_features, 1e-5)
        submodel2 = MLP(self.in_features, self.hidden_features, self.out_features, 1e-5)
        config = {
            "a": (torch.tensor([0]), torch.arange(self.in_features)),
            "b": (torch.arange(self.out_features), torch.tensor([0]))
        }
        updated_model_weights = aggregate_submodels(model, [submodel1, submodel2], [config, config])
        set_parameters(model, updated_model_weights)
        self.assertTrue(
            model.a.weight[0].allclose((submodel1.a.weight + submodel2.a.weight) / 2)
        )

    def _get_config(self, model, rate):
        capacity_config = model.get_reduced_model_config(rate)
        client_submodel_config_idx = [torch.arange(c) for c in capacity_config]
        expanded_config_idx = \
            model.expand_configuration_to_model(client_submodel_config_idx)
        config = expanded_config_idx
        return config

    def test_aggregate_modelc(self):
        model = FedDropoutNet(10, 1.0)
        model_copy = copy.deepcopy(model)

        rate1, rate2 = 0.25, 0.5
        submodel1 = FedDropoutNet(10, rate1)
        submodel2 = FedDropoutNet(10, rate2)
        config1, config2 = self._get_config(model, rate1), self._get_config(model, rate2)
        updated_model_weights = \
        aggregate_submodels(model, [submodel1, submodel2], [config1, config2])
        set_parameters(model, updated_model_weights)

        for smp1, smp2, p, op in zip(submodel1.parameters(), submodel2.parameters(), model.parameters(), model_copy.parameters()):
            min_size1 = smp1.shape[0]

            slice1_1 = slice(0, min_size1)
            slice2_1 = slice(min_size1, smp2.shape[0])
            slice3_1 = slice(smp2.shape[0], p.shape[0])
            if p.ndim > 1:
                min_size2 = smp1.shape[1]

                slice1_2 = slice(0, min_size2)
                slice1 = (slice1_1, slice1_2)

                slice2_2 = slice(min_size2, smp2.shape[1])
                slice2 = (slice2_1, slice2_2)

                slice3_2 = slice(smp2.shape[1], p.shape[1])
                slice3 = (slice3_1, slice3_2)

                avg = (smp1[slice1] + smp2[slice1]) / 2
                assert (avg == p[slice1]).all()
                assert (smp2[slice2] == p[slice2]).all()
                assert (p[slice3] == op[slice3]).all()

            else:
                avg = (smp1[slice1_1] + smp2[slice1_1]) / 2
                assert (avg == p[slice1_1]).all()
                assert (smp2[slice2_1] == p[slice2_1]).all()
                assert (p[slice3_1] == op[slice3_1]).all()


    def test_extracting_convnet(self):
        model = SmallConvNet(10, 1.0)
        submodel = SmallConvNet(10, self.ratio)

        submodel_config = model.get_reduced_model_config(self.ratio)
        submodel_config = [np.arange(conf) for conf in submodel_config]
        submodel_config_expanded = model.expand_configuration_to_model(submodel_config)
        submodel_parameters = model.extract_submodel_parameters(submodel_config_expanded)
        set_parameters(submodel, submodel_parameters)

        x = torch.randn(self.batch_size, 3, 32, 32)

        submodel_pred = submodel(x)
        wmc = model.whole_model_config
        missings = []
        for idx in range(1, len(submodel_config) - 1):
            mx = wmc[idx]
            missing = [i for i in range(mx) if i not in submodel_config[idx]]
            missings.append(missing)

        # compute output layer by layer by setting missing values to 0
        t = model.conv1(x)
        t[:,missings[0]] = 0.
        t = model.pool(F.relu(t))
        t = model.conv2(t)
        t[:,missings[1]] = 0.
        t = model.pool(F.relu(t))
        t = t.view(-1, model.model_config[2] * model.flatten_expansion)
        t = F.relu(model.fc1(t))
        t[:,missings[2]] = 0.
        t = F.relu(model.fc2(t))
        t[:,missings[3]] = 0.
        t = model.fc3(t)

        self.assertTrue(
            bool(t.allclose(submodel_pred))
        )

    def test_extracting_fedavgmnistnet(self):
        model = FedAvgMnistNet(10, 1.0)
        submodel = FedAvgMnistNet(10, self.ratio)
        model.scaler = nn.Identity()
        submodel.scaler = nn.Identity()

        # model.scaler = nn.Identity()
        submodel_config = model.get_reduced_model_config(self.ratio)
        submodel_config = [np.arange(conf) for conf in submodel_config]
        submodel_config_expanded = model.expand_configuration_to_model(submodel_config)
        submodel_parameters = model.extract_submodel_parameters(submodel_config_expanded)
        set_parameters(submodel, submodel_parameters)

        x = torch.randn(self.batch_size, 1, 28, 28)

        submodel_pred = submodel(x)
        wmc = model.whole_model_config
        missings = []
        for idx in range(1, len(submodel_config) - 1):
            mx = wmc[idx]
            missing = [i for i in range(mx) if i not in submodel_config[idx]]
            missings.append(missing)

        # compute output layer by layer by setting missing values to 0
        t = model.conv1(x)
        t[:,missings[0]] = 0.
        t = model.pool(F.relu(t))
        t = model.conv2(t)
        t[:,missings[1]] = 0.
        t = model.pool(F.relu(t))
        t = t.view(self.batch_size, model.model_config[2] * model.flatten_expansion)
        t = F.relu(model.fc1(t))
        t[:,missings[2]] = 0.
        t = model.fc2(t)

        self.assertTrue(
            bool(t.allclose(submodel_pred))
        )
