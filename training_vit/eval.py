import torch

def evaluate_acc(model, device, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total

def evaluate_error(model, device, dataloader):
    model.eval()
    total_error = 0.0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            error = torch.nn.functional.mse_loss(outputs, batch_y)
            total_error += error.item()

    return total_error / len(dataloader)
