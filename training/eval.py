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

        # Classify labels after regression
        # target_transform = lambda target: convert_labels(target)
        # batch_y = np.array([target_transform(y) for y in batch_y], dtype='float32')
        # predictions = np.array([target_transform(output) for output in outputs], dtype='float32')

        predictions = torch.argmax(outputs, dim=1)
        total_acc += (predictions==batch_y).sum()

    return total_acc.cpu() / len(dataloader.dataset)

def evaluate_error(model, device, dataloader):
    model.eval()
    total_angles = 0.0

    def angle_between(gaze_x1, gaze_y1, gaze_x2, gaze_y2):
        cos_sim = torch.cos(gaze_y1) * torch.cos(gaze_y2)
        cos_sim *= torch.cos(gaze_x1 - gaze_x2)
        cos_sim += torch.sin(gaze_y1) * torch.sin(gaze_y2)
        return torch.acos(cos_sim)

    for batch_X, batch_y in dataloader:
        # Move data to device
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_X)
        gaze_x, gaze_y = batch_y[:, 0], batch_y[:, 1]
        gaze_x_pred, gaze_y_pred = predictions[:, 0], predictions[:, 1]

        print(torch.nn.MSELoss()(predictions, batch_y).item())
        total_angles += angle_between(gaze_x, gaze_y, gaze_x_pred, gaze_y_pred).sum()

    return total_angles.cpu() / len(dataloader.dataset)