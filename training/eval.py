import torch

def evaluate_acc(model, device, dataloader):
    model.eval()
    total_acc = 0.0
    for batch_X, batch_y in dataloader:
        # Move data to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions==batch_y).sum()

    return total_acc.cpu() / len(dataloader.dataset)