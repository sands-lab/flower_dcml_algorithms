
import numpy as np
import torch

from src.models.helper import construct_matrix


def _iterate_dataloader(dataloader, epochs, device):
    for _ in range(epochs):
        for data in dataloader:
            data = [d.to(device) for d in data]
            yield data


def init_optimizer(model, optimizer_name, lr):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise Exception(f"Optimizer {optimizer_name} is not allowed")
    return optimizer

def train(model, dataloader, optimizer_name, epochs, lr, device):

    model.train()
    optimizer = init_optimizer(model, optimizer_name, lr)
    criterion = torch.nn.CrossEntropyLoss()

    for data, targets in _iterate_dataloader(dataloader, epochs, device):

        preds = model(data)
        loss = criterion(preds, targets)

        model.zero_grad()
        loss.backward()
        optimizer.step()


def train_fpx(model, dataloader, optimizer_name, epochs, lr, device, global_model, mu):
    model.train()
    optimizer = init_optimizer(model, optimizer_name, lr)
    criterion = torch.nn.CrossEntropyLoss()

    for data, targets in _iterate_dataloader(dataloader, epochs, device):
        preds = model(data)
        loss = criterion(preds, targets)

        proximal_term = 0.0
        for local_weights, global_weights in zip(model.parameters(), global_model.parameters()):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss += (mu / 2) * proximal_term

        model.zero_grad()
        loss.backward()
        optimizer.step()


def train_fd(model, dataloader, optimizer_name, epochs, lr, device, kd_weight, logit_matrix):
    assert isinstance(logit_matrix, torch.Tensor)
    num_classes = logit_matrix.shape[0]
    assert logit_matrix.ndim() == 2
    assert num_classes == logit_matrix.shape[1] == model.modules[-1].out_features

    model.train()
    logit_matrix = logit_matrix.to(device)
    optimizer = init_optimizer(model, optimizer_name, lr)
    criterion = torch.nn.CrossEntropyLoss()

    cnts = torch.zeros((num_classes,))
    running_sums = torch.zeros((num_classes, num_classes))

    for data, targets in _iterate_dataloader(dataloader, epochs, device):
        preds = model(data)
        ce_loss = criterion(preds, targets)
        kd_loss = kd_weight * criterion(preds, logit_matrix[targets])
        loss = ce_loss + kd_loss

        model.zero_grad()
        loss.backward()
        optimizer.step()

        cnts += torch.bincount(input, minlength=num_classes)
        running_sums += construct_matrix(preds, targets, num_classes)
    running_sums /= cnts
    return running_sums


def train_fedgkt_client(model, dataloader, optimizer_name, epochs, lr, device):
    model.train()
    optimizer = init_optimizer(model, optimizer_name, lr)
    criterion = torch.nn.CrossEntropyLoss()

    # train model
    for data in _iterate_dataloader(dataloader, epochs, device):
        if len(data) == 2:
            data, targets = data
        elif len(data) == 3:
            data, targets, logits = data

        preds = model(data)
        ce_loss = criterion(preds, targets)

        loss = ce_loss
        model.zero_grad()
        loss.backward()
        optimizer.step()

    # extract features and logits
    H_k, Z_k, Y_k = [], [], []
    for idx in range(len(dataloader.dataset)):
        data = dataloader.dataset[idx]
        if len(data) == 2:
            data, y = data
        else:
            data, y, _ = data
        data = data.to(device).unsqueeze(0)
        with torch.no_grad():
            hk = model.get_embedding(data)
            zk = model.get_predictions(hk)

        H_k.append(hk.cpu().numpy())
        Z_k.append(zk.cpu().numpy())
        Y_k.append(y)
    H_k = np.vstack(H_k)
    Z_k = np.vstack(Z_k)
    Y_k = np.vstack(Y_k)
    print(H_k.shape)
    return H_k, Z_k, Y_k


def train_fedgkt_server(model, dataloader, optimizer_name, epochs, lr, device):
    model.train()
    model.to(device)

    optimizer = init_optimizer(model, optimizer_name, lr)
    for embeddings, logits, targets in _iterate_dataloader(dataloader, epochs, device):
        pred_logits = model(embeddings)

        model.zero_grad()
        loss.backward()
        optimizer.step()

