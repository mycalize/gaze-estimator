import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    # for now: (HW 5) [conv - relu - pool] - [conv - relu] - [aff] - softmax
    def __init__(self, input_dims):
        super(ConvNet, self).__init__()
        self.h, self.w = input_dims
        self.m1 = nn.Conv2d(1, 8, 3, padding=1)
        self.m2 = nn.Conv2d(8, 4, 3, padding=1)
        self.lin_in_features = 4 * (self.h // 2) * (self.w // 2)
        self.m3 = nn.Linear(self.lin_in_features, 16)

    def forward(self, x):
        z1 = F.relu(self.m1(x))
        h1 = F.max_pool2d(z1, kernel_size=2)
        h2 = F.relu(self.m2(h1))
        h2 = h2.view(-1, self.lin_in_features)
        outputs = self.m3(h2)
        return outputs