import torch
import torch.nn as nn
import torch.nn.functional as F

class NVGaze(nn.Module):
    """ CNN based on NVGaze paper """
    # Note: paper used inputs with resolution 127 x 127
    # [conv - relu - dropout] * 6 - [aff]
    def __init__(self, input_dims, dropout_param=0.1, out_num_features=2):
        super(NVGaze, self).__init__()
        self.h, self.w = input_dims
        self.p = dropout_param
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 12, 3, stride=2)
        self.conv3 = nn.Conv2d(12, 18, 3, stride=2)
        self.conv4 = nn.Conv2d(18, 27, 3, stride=2)
        self.conv5 = nn.Conv2d(27, 40, 3, stride=2)
        # self.conv6 = nn.Conv2d(40, 60, 3, stride=2) # do 1 fewer layer for now b/c of resolution differences
        self.fc_in_features = self.size_fc()
        self.fc = nn.Linear(self.fc_in_features, out_num_features)

    def forward(self, x):
        x = F.dropout2d(F.relu(self.conv1(x)), p=self.p)
        x = F.dropout2d(F.relu(self.conv2(x)), p=self.p)
        x = F.dropout2d(F.relu(self.conv3(x)), p=self.p)
        x = F.dropout2d(F.relu(self.conv4(x)), p=self.p)
        x = F.dropout2d(F.relu(self.conv5(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv6(x)), p=self.p)
        x = x.view(-1, self.fc_in_features)
        outputs = self.fc(x)
        return outputs
    
    def size_fc(self):
        x = torch.zeros(1, 1, self.h, self.w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)

        m = 1
        for i in x.size():
            m *= i
        return m