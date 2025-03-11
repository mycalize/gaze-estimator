import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    """
    Simple CNN with batchnorm.
    Layers: [conv - bn - relu - pool] - [conv - bn - relu] - [aff]
    """
    def __init__(self, input_dims):
        super(ConvNet, self).__init__()
        self.h, self.w = input_dims
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 4, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(4)
        self.fc_in_features = 4 * (self.h // 2) * (self.w // 2)
        self.fc = nn.Linear(self.fc_in_features, 16)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = x.view(-1, self.fc_in_features)
        outputs = self.fc(x)
        return outputs