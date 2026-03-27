import torch
import torch.nn as nn
import json as js

class CausalTSF(nn.Module):
    """
    Causal-TSF模型
    """
    def __init__(self, input_dim, train_config_file="config/train_config.json"):
        super().__init__()
        with open(train_config_file, 'r', encoding='utf-8') as f:
            train_config = js.load(f)

        num_samples = int(train_config['num_samples'])
        confounder_dim = min(int(train_config['confounder_dim']), input_dim)
        hidden_dim = int(train_config['hidden_dim'])
        
        self.num_samples = num_samples
        self.confounder_dim = confounder_dim

        # 特征提取器
        self.feature_extractor = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # 混淆变量估计器
        self.confounder_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, confounder_dim*2)
        )
        
        # 去偏模块
        self.debiasing_module = nn.Sequential(
            nn.Linear(hidden_dim+confounder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        
        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x_data):
        # 特征提取
        z_data, _ = self.feature_extractor(x_data)
        z_data = z_data[:, -1, :]
        
        # 代理变量计算
        proxy_values = torch.mean(z_data, dim=0, keepdim=True)
        
        # 估计混淆变量分布
        params = self.confounder_estimator(proxy_values)
        mu, log_var = torch.chunk(params, 2, dim=1)
        sigma = torch.exp(0.5 * log_var)
        
        # 采样混淆变量
        batch_size = x_data.shape[0]
        epsilon = torch.randn(self.num_samples, batch_size, self.confounder_dim, device=x_data.device)
        c_samples = mu.unsqueeze(0)+sigma.unsqueeze(0)*epsilon
        
        # 去偏预测
        z_expanded = z_data.unsqueeze(0).expand(self.num_samples, -1, -1)
        combined = torch.cat([z_expanded, c_samples.expand(-1, z_data.size(0), -1)], dim=2)
        debiased = self.debiasing_module(combined)
        predictions = self.prediction_head(debiased)
        final_pred = torch.mean(predictions, dim=0)
        
        return final_pred, mu, sigma
        
        
