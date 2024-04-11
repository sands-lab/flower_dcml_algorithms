import torch
import torch.nn as nn

from src.helper.optimization_config import OptimizationConfig
from slower.server.server_model.proxy.server_model_proxy import ServerModelProxy


def train_model(optimization_config: OptimizationConfig, server_model_proxy: ServerModelProxy):
    model = optimization_config.model
    optimizer = optimization_config.optimizer

    for _ in range(optimization_config.epochs):
        for images, labels in optimization_config.dataloader:
            images = images.to(optimization_config.device)
            embeddings = model(images)

            error = server_model_proxy.numpy_serve_gradient_update_request(
                embeddings=embeddings.detach().cpu().numpy(),
                labels=labels.numpy()
            )
            error = torch.from_numpy(error).to(optimization_config.device)

            model.zero_grad()
            embeddings.backward(error)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()


def train_u_model(optimization_config: OptimizationConfig, server_model_proxy: ServerModelProxy):
    model = optimization_config.model
    encoder, clf_head = model["encoder"], model["clf_head"]

    criterion = nn.CrossEntropyLoss()
    optimizer = optimization_config.optimizer

    for _ in range(optimization_config.epochs):
        for images, labels in optimization_config.dataloader:
            images = images.to(optimization_config.device)
            labels = labels.to(optimization_config.device)

            # forward pass
            client_encoder_embeddings = encoder(images)  # client encoder
            server_embeddings = server_model_proxy.numpy_u_forward(  # server encoder
                embeddings=client_encoder_embeddings.detach().cpu().numpy()
            )
            server_embeddings = torch.from_numpy(server_embeddings).to(optimization_config.device)
            server_embeddings.requires_grad_(True)
            final_predictions = clf_head(server_embeddings)  # client head

            # prepare everyting for the packward pass
            encoder.zero_grad()
            clf_head.zero_grad()

            # backward pass
            loss = criterion(final_predictions, labels)

            loss.backward()
            client_encoder_gradient = \
                server_model_proxy.numpy_u_backward(server_embeddings.grad.detach().cpu().numpy())
            client_encoder_gradient = \
                torch.from_numpy(client_encoder_gradient).to(optimization_config.device)
            client_encoder_embeddings.backward(client_encoder_gradient)

            # make optimization step
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), optimization_config.grad_norm_clipping_param)
            torch.nn.utils.clip_grad_norm_(clf_head.parameters(), optimization_config.grad_norm_clipping_param)
            optimizer.step()
