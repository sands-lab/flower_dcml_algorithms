import torch


@torch.no_grad
def test_accuracy(model, dataloader, device=torch.device("cuda:0")):
    model.eval()
    correct_predictions = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        preds = torch.argmax(preds, axis=1)
        correct_predictions += (preds == targets).sum().cpu().item()
    return float(correct_predictions / len(dataloader.dataset))
