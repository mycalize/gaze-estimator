import torch
# import numpy as np
# from data.convert_labels import *

def evaluate_acc(model, device, dataloader):
  model.eval()
  total_acc = 0.0
  for batch_X, batch_y in dataloader:
    # Move data to device
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    outputs = model(batch_X)

    predictions = torch.argmax(outputs.detach(), dim=1)
    total_acc += (predictions == batch_y).sum()

  return total_acc.cpu() / len(dataloader.dataset)

def evaluate_error(model, device, dataloader):
  model.eval()
  total_angles = 0.0
  for batch_X, batch_y in dataloader:
    # Move data to device
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    predictions = model(batch_X)
    gaze_x, gaze_y = batch_y[:, 0], batch_y[:, 1]
    predictions_detached = predictions.detach()
    gaze_x_pred, gaze_y_pred = predictions_detached[:, 0], predictions_detached[:, 1]

    total_angles += __angle_diff(gaze_x, gaze_y, gaze_x_pred, gaze_y_pred).sum()

  return total_angles.cpu() / len(dataloader.dataset)

def __angle_diff(gaze_x1, gaze_y1, gaze_x2, gaze_y2):
  cos_sim = torch.cos(gaze_y1) * torch.cos(gaze_y2)
  cos_sim *= torch.cos(gaze_x1 - gaze_x2)
  cos_sim += torch.sin(gaze_y1) * torch.sin(gaze_y2)
  return torch.acos(cos_sim)
