import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNetLSTM(nn.Module):
    """ CNN connected to LSTM. """
    def __init__(self, input_dims, dropout_param=0.1, out_num_features=2):
        super().__init__()
        self.h, self.w = input_dims
        self.p = dropout_param
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2)
        self.conv1_in = nn.InstanceNorm2d(8, affine=True)
        self.conv2 = nn.Conv2d(8, 12, 3, stride=2)
        self.conv2_in = nn.InstanceNorm2d(12, affine=True)
        self.conv3 = nn.Conv2d(12, 18, 3, stride=2)
        self.conv3_in = nn.InstanceNorm2d(18, affine=True)
        self.conv4 = nn.Conv2d(18, 27, 3, stride=2)
        self.conv4_in = nn.InstanceNorm2d(27, affine=True)
        self.conv5 = nn.Conv2d(27, 40, 3, stride=2)
        self.lstm_in_features = self.size_lstm()
        self.lstm = LSTM(self.lstm_in_features, out_num_features=out_num_features)

    def forward(self, x):
        x = F.dropout2d(self.conv1_in(F.relu(self.conv1(x))), p=self.p)
        x = F.dropout2d(self.conv2_in(F.relu(self.conv2(x))), p=self.p)
        x = F.dropout2d(self.conv3_in(F.relu(self.conv3(x))), p=self.p)
        x = F.dropout2d(self.conv4_in(F.relu(self.conv4(x))), p=self.p)
        x = F.dropout2d(F.relu(self.conv5(x)), p=self.p)
        # x = F.dropout2d(F.relu(self.conv6(x)), p=self.p)
        x = x.view(-1, self.lstm_in_features)
        outputs = self.lstm(x)
        return outputs
    
    def size_lstm(self):
        x = torch.zeros(1, 1, self.h, self.w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.conv6(x)

        return torch.numel(x)

class LSTM(nn.Module):
    """ Simple LSTM implementation. """
    # Expected input dims: (L, H_in), where L is seq length and H_in is input size
    def __init__(self, in_size, out_num_features=2):
        super().__init__()
        lstm_hidden_size = 64
        self.lstm_layers = nn.LSTM(in_size, lstm_hidden_size, 4, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_size * 2, lstm_hidden_size * 2)
        self.lnorm = nn.LayerNorm(lstm_hidden_size * 2)
        self.fc_out = nn.Linear(lstm_hidden_size * 2, out_num_features)
    
    def forward(self, x):
        x, _ = self.lstm_layers(x)
        x = self.lnorm(self.fc(F.relu(self.fc(x))))
        x = self.fc_out(x)
        return x