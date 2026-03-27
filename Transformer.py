import torch
import torch.nn as nn

class TransformerNet(nn.Module):
    """
    Transformer编码器模型
    """
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)                     
        x = self.transformer(x)               
        x = x.mean(dim=1)                     
        pred = self.fc(x)                     
        return pred