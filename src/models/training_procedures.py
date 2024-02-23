import copy

import numpy as np
import torch

from src.models.helper import construct_matrix
from src.models.utils import KLLoss
from src.helper.optimization_config import OptimizationConfig, init_optimizer


def _iterate_dataloader(optimization_config: OptimizationConfig):
    for _ in range(optimization_config.epochs):
        for data in optimization_config.dataloader:
            data = [d.to(optimization_config.device) for d in data]
            yield data


def _clip_gradient(model, config):
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clipping_param)


def compute_proximal_term(model, global_model, mu):
    proximal_term = 0.0
    for local_weights, global_weights in zip(model.parameters(), global_model.parameters()):
        proximal_term += torch.square((local_weights - global_weights).norm(2))
    return (mu / 2) * proximal_term



def train(optimization_config: OptimizationConfig):
    model = optimization_config.model
    optimizer = optimization_config.optimizer

    criterion = torch.nn.CrossEntropyLoss()
    for data, targets in _iterate_dataloader(optimization_config):
        preds = model(data)
        loss = criterion(preds, targets)
        assert torch.isfinite(loss).all()

        model.zero_grad()
        loss.backward()
        _clip_gradient(model, optimization_config)
        optimizer.step()


def train_fpx(optimization_config: OptimizationConfig, global_model, mu):

    model = optimization_config.model
    optimizer = optimization_config.optimizer

    criterion = torch.nn.CrossEntropyLoss()

    for data, targets in _iterate_dataloader(optimization_config):
        preds = model(data)
        loss = criterion(preds, targets)
        loss += compute_proximal_term(model, global_model, mu)

        model.zero_grad()
        loss.backward()
        _clip_gradient(model, optimization_config)
        optimizer.step()


def train_fd(
        optimization_config: OptimizationConfig,
        kd_weight,
        logit_matrix,
        num_classes,
        temperature
):

    model = optimization_config.model
    optimizer = optimization_config.optimizer
    criterion = torch.nn.CrossEntropyLoss()

    # sanity checks
    assert isinstance(logit_matrix, torch.Tensor)
    assert logit_matrix.ndim == 2 or logit_matrix.numel() == 0

    empty_logit_matrix = logit_matrix.ndim == 1
    logit_matrix = logit_matrix.to(optimization_config.device)
    cnts = torch.zeros((num_classes,)).to(optimization_config.device)
    running_sums = torch.zeros((num_classes, num_classes)).to(optimization_config.device)
    if not empty_logit_matrix:
        valid_logits = (logit_matrix >= 0).all(axis=1).nonzero(as_tuple=True)[0]
        if len(valid_logits) < num_classes:
            missing_labels = set(range(num_classes)) - set(valid_logits.tolist())
            print(f"WARNING: the following logits are missing: {missing_labels}")

    for data, targets in _iterate_dataloader(optimization_config):
        preds = model(data)
        loss = criterion(preds, targets)
        if not empty_logit_matrix:
            # CrossEntropyLoss performs softmax internally, so no need to call the softmax here
            assert not torch.isnan(preds).any()
            assert not torch.isnan(criterion(preds / temperature, logit_matrix[targets])), f"{preds}, \n\n {logit_matrix[targets]}\n\n\n"
            mask = torch.isin(targets, valid_logits)
            targets_ = targets[mask]
            preds_ = preds[mask,:]
            loss += kd_weight * criterion(preds_ / temperature, logit_matrix[targets_])

        model.zero_grad()
        loss.backward()
        _clip_gradient(model, optimization_config)
        optimizer.step()

        cnts += torch.bincount(targets, minlength=num_classes)
        preds = torch.nn.functional.softmax(preds / temperature, dim=1)
        running_sums += construct_matrix(preds, targets, num_classes)
    running_sums = running_sums[cnts > 0]
    client_classes = (cnts > 0).cpu().numpy()
    cnts = cnts[cnts > 0].reshape(-1, 1)
    running_sums = (running_sums / cnts).cpu().numpy()
    return [running_sums, client_classes]


# pylint: disable=C0103
def train_fedgkt_client(optimization_config: OptimizationConfig, temperature):
    model = optimization_config.model
    optimizer = optimization_config.optimizer

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_kl = KLLoss(temperature)

    # train model
    for data in _iterate_dataloader(optimization_config):
        if len(data) == 2:
            data, targets = data
            server_logits = None
        elif len(data) == 3:
            data, targets, server_logits = data

        preds = model(data)
        loss = criterion_ce(preds, targets)
        if server_logits is not None:
            loss += criterion_kl(preds, server_logits)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    # extract features and logits
    ordered_dataloader = torch.utils.data.DataLoader(optimization_config.dataloader.dataset,
                                                     shuffle=False, batch_size=32, drop_last=False)
    H_k, Z_k, Y_k = [], [], []
    for data in ordered_dataloader:
        data = [d.to(optimization_config.device) for d in data]
        if len(data) == 2:
            data, targets = data
        else:
            data, targets, _ = data
        with torch.no_grad():
            hk = model.get_embedding(data)
            zk = model.get_predictions(hk)

        H_k.append(hk.cpu().numpy())
        Z_k.append(zk.cpu().numpy())
        Y_k.append(targets.detach().cpu().numpy())
    H_k = np.vstack(H_k)
    Z_k = np.vstack(Z_k)
    Y_k = np.hstack(Y_k)
    print(H_k.shape)
    return H_k, Z_k, Y_k


def train_fedgkt_server(model, dataloaders, optimizer_name, epochs, lr, device, temperature):
    model.train()
    model.to(device)
    print(f"Server training on {device}")
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_kl = KLLoss(temperature)

    optimizer = init_optimizer(model, optimizer_name, lr)
    for i in range(epochs):
        print(f"Epoch {i}")
        train_order = np.random.permutation(len(dataloaders))
        for cid in train_order:
            print(f"Client {cid}")
            dataloader = dataloaders[cid]

            for _ in range(epochs):
                for data in dataloader:
                    data = [d.to(device) for d in data]
                    embeddings, logits, targets = data

                    pred_logits = model(embeddings)

                    ce_loss = criterion_ce(pred_logits, targets)
                    kl_loss = criterion_kl(pred_logits, logits)
                    loss = ce_loss + kl_loss

                    model.zero_grad()
                    loss.backward()
                    optimizer.step()


def train_kd_ds_fl(optimization_config: OptimizationConfig, kd_temperature):
    model = optimization_config.model
    optimizer = optimization_config.optimizer

    criterion = KLLoss(kd_temperature)  # which loss function should be used?

    for images, target_logits in _iterate_dataloader(optimization_config):
        pred_logits = model(images)

        loss = criterion(pred_logits, target_logits)

        model.zero_grad()
        loss.backward()
        _clip_gradient(model, optimization_config)
        optimizer.step()


def train_feddf(optimization_config: OptimizationConfig, temperature):
    # model is the student (server) model
    # train with AVGLOGITS (page 3 in the paper)

    model = optimization_config.model
    optimizer = optimization_config.optimizer
    global_model = copy.deepcopy(model)
    criterion = KLLoss(temperature)

    for images, teacher_consensus in _iterate_dataloader(optimization_config):
        student_predictions = model(images)

        loss = criterion(student_predictions, teacher_consensus)  # in the paper it seems they do the opposite but it is strange...
        loss += compute_proximal_term(model, global_model, 0.0001)
        assert torch.isfinite(loss).all()

        model.zero_grad()
        loss.backward()
        _clip_gradient(model, optimization_config)
        optimizer.step()


def train_model_layers(optimization_config: OptimizationConfig, train_layers, gd_steps: int):
    model = optimization_config.model
    optimizer = optimization_config.optimizer
    for k, v in model.state_dict().items():
        v.requires_grad = k in train_layers
    criterion = torch.nn.CrossEntropyLoss()

    for i, (images, target_logits) in enumerate(_iterate_dataloader(optimization_config)):
        pred_logits = model(images)

        loss = criterion(pred_logits, target_logits)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        if gd_steps is not None and i >= gd_steps:
            break

    for k, v in model.state_dict().items():
        v.requires_grad = True
