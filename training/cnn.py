import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    # for now: (HW 5) [conv - relu - pool] - [conv - relu] - [aff] - softmax
    def __init__(self):
        super(ConvNet, self).__init__()
        self.m1 = nn.Conv2d(1, 128, 3, padding=1)
        self.m2 = nn.Conv2d(128, 256, 3, padding=1)
        self.m3 = nn.Linear(256*80*60, 16)

    def forward(self, x):
        z1 = F.relu(self.m1(x))
        h1 = F.max_pool2d(z1, kernel_size=2)
        # 80 * 60
        h2 = F.relu(self.m2(h1))
        # 80 * 60
        h2 = h2.view(-1, 256*80*60)
        outputs = self.m3(h2)
        return outputs