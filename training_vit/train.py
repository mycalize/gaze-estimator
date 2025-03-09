import torch

def train(model, device, mode, criterion, optimizer, train_loader, val_loader, epochs=100):
    model.train()
    loss_history = []
    train_perf_history = []
    val_perf_history = []
    print("hi")
    for epoch in range(epochs):
        epoch_loss = 0.0
        print("yayyy")
        for batch_X, batch_y in train_loader:
            print("batch_X")
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        loss_history.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_history[-1]}")

    return loss_history, train_perf_history, val_perf_history
