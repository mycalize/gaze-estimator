from training.eval import *

def train(model, device, mode, criterion, optimizer, train_loader, val_loader, epochs=10, verbose=False):
  train_loss_history = []
  train_perf_history = []
  val_perf_history = []

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

    if mode == 'reg':
      train_perf = evaluate_error(model, device, train_loader)
      valid_perf = evaluate_error(model, device, val_loader)
      print(f"| Epoch {epoch:2d} | Train err {train_perf:.6f} | Val err {valid_perf:.6f} |")
    else:
      train_perf = evaluate_acc(model, device, train_loader)
      valid_perf = evaluate_acc(model, device, val_loader)
      print(f"| Epoch {epoch:2d} | Train acc {train_perf:.6f} | Val acc {valid_perf:.6f} |")

    train_perf_history.append(train_perf)
    val_perf_history.append(valid_perf)

    if verbose:
      print(f'Current allocated memory (B): {torch.mps.current_allocated_memory()}')

  return train_loss_history, train_perf_history, val_perf_history