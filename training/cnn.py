import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # for now: (HW 5) [conv - bn - relu - pool] - [conv - bn - relu] - [aff] - softmax
    def __init__(self, input_dims):
        super(CNN, self).__init__()
        self.h, self.w = input_dims
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 4, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(4)
        self.fc_in_features = 4 * (self.h // 2) * (self.w // 2)
        self.fc = nn.Linear(self.fc_in_features, 16)

    def forward(self, x):
        z1 = F.relu(self.conv1_bn(self.conv1(x)))
        h1 = F.max_pool2d(z1, kernel_size=2)
        h2 = F.relu(self.conv2_bn(self.conv2(h1)))
        h2 = h2.view(-1, self.fc_in_features)
        outputs = self.fc(h2)
        return outputs