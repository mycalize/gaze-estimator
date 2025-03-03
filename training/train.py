from training.eval import *

def train(model, criterion, optimizer, train_loader, epochs=10):
    iter_train_loss_list = []
    train_loss_list = []
    valid_loss_list = []
    train_acc_list = []
    valid_acc_list = []
    for epoch in range(1, epochs+1):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            iter_train_loss_list.append(loss.item())

        # train_loss = evaluate_loss(model, criterion, train_loader)
        # valid_loss = evaluate_loss(model, criterion, valid_loader)
        train_acc = evaluate_acc(model, train_loader)
        # valid_acc = evaluate_acc(model, valid_loader)
        # train_loss_list.append(train_loss)
        # valid_loss_list.append(valid_loss)
        train_acc_list.append(train_acc)
        # valid_acc_list.append(valid_acc)

        # print(f"| epoch {epoch:2d} | train loss {train_loss:.6f} | train acc {train_acc:.6f} | valid loss {valid_loss:.6f} | valid acc {valid_acc:.6f} |")
        print(f"| epoch {epoch:2d} | train acc {train_acc:.6f} |")

    # return train_loss_list, valid_loss_list, train_acc_list, valid_acc_list
    return iter_train_loss_list, train_acc_list