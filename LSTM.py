import torch.nn as nn

class LSTMNet(nn.Module):
    """
    双向LSTM模型
    """
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_out = lstm_out[:, -1, :]          
        pred = self.fc(last_out)               
        return pred