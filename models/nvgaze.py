import torch
import torch.nn as nn
import torch.nn.functional as F

class NVGaze(nn.Module):
    """ CNN based on NVGaze paper """
    # Note: paper used inputs with resolution 127 x 127
    # [conv - relu - dropout] * 6 - [aff]
    def __init__(self, input_dims, out_num_features=2, dropout_param=0.1):
        super(NVGaze, self).__init__()
        c, h, w = input_dims
        self.p = dropout_param
        self.conv1 = nn.Conv2d(c, 16, 3, stride=2)
        self.conv1_in = nn.InstanceNorm2d(16, affine=True)
        self.conv2 = nn.Conv2d(16, 24, 3, stride=2)
        self.conv2_in = nn.InstanceNorm2d(24, affine=True)
        self.conv3 = nn.Conv2d(24, 36, 3, stride=2)
        self.conv3_in = nn.InstanceNorm2d(36, affine=True)
        self.conv4 = nn.Conv2d(36, 54, 3, stride=2)
        self.conv4_in = nn.InstanceNorm2d(54, affine=True)
        self.conv5 = nn.Conv2d(54, 81, 3, stride=2)
        # self.conv6 = nn.Conv2d(81, 122, 3, stride=2)
        self.fc_in_features = self.size_fc(c, h, w)
        self.fc = nn.Linear(self.fc_in_features, out_num_features)

    def forward(self, x):
        x = F.dropout2d(self.conv1_in(F.relu(self.conv1(x))), p=self.p)
        x = F.dropout2d(self.conv2_in(F.relu(self.conv2(x))), p=self.p)
        x = F.dropout2d(self.conv3_in(F.relu(self.conv3(x))), p=self.p)
        x = F.dropout2d(self.conv4_in(F.relu(self.conv4(x))), p=self.p)
        x = F.dropout2d(F.relu(self.conv5(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv6(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv1(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv2(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv3(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv4(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv5(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv6(x)), p=self.p)
        x = x.view(-1, self.fc_in_features)
        outputs = self.fc(x)
        return outputs
    
    def size_fc(self, c, h, w):
        x = torch.zeros(1, c, h, w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)

        return torch.numel(x)