import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # for now: (HW 5) [conv - relu - pool] - [conv - relu] - [aff] - softmax
    def __init__(self):
        super(CNN, self).__init__()
        self.m1 = nn.Conv2d(1, 128, 3, padding=1)
        self.m2 = nn.Conv2d(128, 256, 3, padding=1)
        self.m3 = nn.Linear(256, 16)

    def forward(self, x):
        h1 = F.max_pool2d(F.relu(self.m1(x)))
        h2 = F.relu(self.m2(h1))
        outputs = self.m3(h2)
        return outputs