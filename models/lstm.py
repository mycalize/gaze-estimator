import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    """
    Simple LSTM implementation.
    Expected input dims: (L, N, H_in), where L is seq length
    and H_in is input size
    """
    def __init__(self, in_size, out_num_features=2):
        super(LSTM, self).__init__()
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