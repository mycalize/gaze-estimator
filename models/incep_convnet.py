import torch
import torch.nn as nn
import torch.nn.functional as F

class IncepConvNet(nn.Module):
  """ CNN with inception layers. """
  def __init__(self, input_dims, out_num_features=2):
    super().__init__()
    c, h, w = input_dims

    self.conv1 = BasicConv2d(c, 64, kernel_size=3, stride=2)
    self.conv2a = BasicConv2d(64, 96, kernel_size=3, stride=2)
    self.conv2b = BasicConv2d(96, 192, kernel_size=3, stride=2)

    self.incep3a = Inception(192)
    self.incep3b = Inception(192)

    self.fc_in_features = self.__size_fc(c, h, w)
    self.fc = nn.Linear(self.fc_in_features, out_num_features)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2a(x)
    x = self.conv2b(x)
    x = self.incep3a(x)
    x = self.incep3b(x)
    x = torch.flatten(x, start_dim=1)
    output = self.fc(x)
    return output
  
  def __size_fc(self, c, h, w):
    x = torch.zeros(1, c, h, w)
    x = self.conv1(x)
    x = self.conv2a(x)
    x = self.conv2b(x)
    x = self.incep3a(x)
    x = self.incep3b(x)
    return torch.numel(x)

class Inception(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

    self.branch3x3_1 = BasicConv2d(in_channels, 48, kernel_size=1)
    self.branch3x3_2 = BasicConv2d(48, 64, kernel_size=3, padding=1)
    
    self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
    self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch3x3 = self.branch3x3_1(x)
    branch3x3 = self.branch3x3_2(branch3x3)

    branch5x5 = self.branch5x5_1(x)
    branch5x5 = self.branch5x5_2(branch5x5)

    outputs = [branch1x1, branch3x3, branch5x5]
    return torch.cat(outputs, dim=1)
  
class BasicConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, **kwargs):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
    self.inorm = nn.InstanceNorm2d(out_channels, affine=True)

  def forward(self, x):
    x = self.conv(x)
    x = self.inorm(x)
    output = F.relu(x, inplace=True)
    return output