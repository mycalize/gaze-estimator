import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class CNN(torch.nn.Module):
    # for now: (HW 5) [conv - relu - pool] - [conv - relu] - [aff] - softmax
    def __init__(self):
        super(CNN, self).__init__()
        self.m1 = torch.nn.Conv2d(1, 128, 3, padding=1)
        self.m2 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.m3 = torch.nn.Linear(256, 16)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)

    def forward(self, x):
        h1 = self.pool(self.relu(self.m1(x)))
        h2 = self.relu(self.m2(h1))
        outputs = self.m3(h2)
        return outputs