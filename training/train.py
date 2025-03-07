from training.eval import *

def train(model, criterion, optimizer, device, train_loader, val_loader, epochs=10):
    train_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(1, epochs+1):
        model.train()
        for batch_X, batch_y in train_loader:
            # Move data to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_history.append(loss.item())

        train_acc = evaluate_acc(model, device, train_loader)
        valid_acc = evaluate_acc(model, device, val_loader)
        train_acc_history.append(train_acc)
        val_acc_history.append(valid_acc)

        print(f"| epoch {epoch:2d} | train acc {train_acc:.6f} | valid acc {valid_acc:.6f} |")

    return train_loss_history, train_acc_history, val_acc_history