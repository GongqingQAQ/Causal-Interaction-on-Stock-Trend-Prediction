import torch
import torch.nn as nn
import json as js

class Tradition(nn.Module):
    """
    消融实验模型
    """
    def __init__(self, input_dim, train_config_file="config/train_config.json"):
        super().__init__()
        with open(train_config_file, 'r', encoding='utf-8') as f:
            train_config = js.load(f)

        hidden_dim = int(train_config['hidden_dim'])
        
        # 特征提取器
        self.feature_extractor = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x_data):
        z_data, _ = self.feature_extractor(x_data)
        z_data = torch.mean(z_data, dim=1)
        
        predictions = self.prediction_head(z_data)
        return predictions