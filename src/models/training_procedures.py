import copy

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


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



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


def compute_validation_loss_feddf(model, valloader, criterion, device):
    loss = 0
    for images, teacher_consensus in valloader:
        images, teacher_consensus = images.to(device), teacher_consensus.to(device)
        loss += criterion(model(images), teacher_consensus)
    return loss.item() / len(valloader)


def train_feddf(optimization_config: OptimizationConfig, temperature, valloader):
    # model is the student (server) model
    # train with AVGLOGITS (page 3 in the paper)

    model = optimization_config.model
    optimizer = optimization_config.optimizer
    global_model = copy.deepcopy(model)
    criterion = KLLoss(temperature)
    early_stopper = EarlyStopper(patience=200, min_delta=0.1)

    for idx, (images, teacher_consensus) in enumerate(_iterate_dataloader(optimization_config)):
        student_predictions = model(images)

        # in the paper it seems they do the opposite but it is strange...
        loss = criterion(student_predictions, teacher_consensus)
        loss += compute_proximal_term(model, global_model, 0.0001)
        assert torch.isfinite(loss).all()

        model.zero_grad()
        loss.backward()
        _clip_gradient(model, optimization_config)
        optimizer.step()

        if idx % 5 == 0:
            val_loss = compute_validation_loss_feddf(model, valloader, criterion,
                                                     optimization_config.device)
            print(f"{idx} {val_loss:.5f}")
            if early_stopper.early_stop(val_loss):
                print(f"Breaking after {idx} iterations...")
                break


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


def train_fedkd(
    optimization_config: OptimizationConfig,
    shared_model: torch.nn.Module,
    temperature: float
):
    private_model = optimization_config.model
    private_optimizer = optimization_config.optimizer

    shared_model.train().to(optimization_config.device)
    shared_optimizer = init_optimizer(
        parameters=shared_model.parameters(),
        optimizer_name=optimization_config.optimizer_name,
        lr=optimization_config.lr,
        weight_decay=optimization_config.weight_decay
    )

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_kl = KLLoss(temperature)

    for data, targets in _iterate_dataloader(optimization_config):
        private_preds = private_model(data)
        shared_preds = shared_model(data)

        # eq 1 & 2 in the paper
        private_supervised_loss = criterion_ce(private_preds, targets)
        shared_supervised_loss = criterion_ce(shared_preds, targets)
        sup_loss_sum = (private_supervised_loss + shared_supervised_loss).detach()

        # eq 3 & 4
        private_kl_loss = criterion_kl(private_preds, shared_preds.detach())  / sup_loss_sum
        shared_kl_loss = criterion_kl(shared_preds, private_preds.detach()) / sup_loss_sum

        private_loss = private_kl_loss + private_supervised_loss
        shared_loss = shared_kl_loss + shared_supervised_loss

        # backward and optimize the two losses
        private_model.zero_grad()
        private_loss.backward()
        _clip_gradient(private_model, optimization_config)
        private_optimizer.step()

        shared_model.zero_grad()
        shared_loss.backward()
        _clip_gradient(shared_model, optimization_config)
        shared_optimizer.step()
