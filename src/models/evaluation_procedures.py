import torch


def test_accuracy(model, dataloader, device):
    model.eval()
    model = model.to(device)
    correct_predictions = 0
    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        with torch.no_grad():
            preds = model(images)
        preds = torch.argmax(preds, axis=1)
        correct_predictions += (preds == targets).sum().cpu().item()
    return float(correct_predictions / len(dataloader.dataset))
