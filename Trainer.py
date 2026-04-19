import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import json as js
import CausalTSF as tsf
import Tradition as tra
import Transformer as trans
import LSTM as lstm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def cal_pearsonr(x, y):
    """
    计算Pearson相关系数
    """
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    corr, _ = pearsonr(x, y)
    return corr if not np.isnan(corr) else 0.0

def cal_spearmanr(x, y):
    """
    安全计算Spearman相关系数
    """
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return 0.0
    corr, _ = spearmanr(x, y)
    return corr if not np.isnan(corr) else 0.0

def prepare_time_series_data_for_causal_tsf(earnings, seq_length, target_col, target_shift, train_ratio, random_seed):
    """ 
    生成 Causal-TSF 的时间序列数据
    """
    np.random.seed(random_seed)
    all_features = earnings.columns.tolist()
    
    # 生成窗口
    def create_windows(data):
        x, y = [], []   
        for i in range(0, len(data)-seq_length-target_shift+1, seq_length+target_shift):
            x.append(data[i: i+seq_length])  
            y.append(data[i+seq_length+target_shift-1, all_features.index(target_col)])
        return np.array(x), np.array(y)
    
    data = earnings.values
    x_all, y_all = create_windows(data)
    
    # 随机划分索引
    n_samples = len(x_all)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    x_train_raw = x_all[train_indices]
    x_test_raw = x_all[test_indices]
    y_train = y_all[train_indices]
    y_test = y_all[test_indices]
    
    # 标准化
    n_samples_train, seq_len, n_features = x_train_raw.shape
    x_train_flat = x_train_raw.reshape(-1, n_features) 
    
    scaler = StandardScaler()
    x_train_scaled_flat = scaler.fit_transform(x_train_flat)
    x_train = x_train_scaled_flat.reshape(n_samples_train, seq_len, n_features)
    
    n_samples_test, seq_len, n_features = x_test_raw.shape
    x_test_flat = x_test_raw.reshape(-1, n_features)
    x_test_scaled_flat = scaler.transform(x_test_flat)
    x_test = x_test_scaled_flat.reshape(n_samples_test, seq_len, n_features)       
    
    return x_train, y_train, x_test, y_test


def prepare_time_series_data_for_tradition(earnings, seq_length, target_col, target_shift, train_ratio, random_seed):
    """ 
    生成非因果时间序列模型数据
    """
    np.random.seed(random_seed)
    features = [target_col]
    
    # 生成窗口
    def create_windows(data):
        x, y = [], []   
        for i in range(0, len(data) - seq_length - target_shift + 1, seq_length + target_shift):
            x.append(data[i: i+seq_length])  
            y.append(data[i+seq_length+target_shift-1, 0])
        return np.array(x), np.array(y)
    
    data = earnings[features].values
    x_all, y_all = create_windows(data)
    
    # 随机划分索引
    n_samples = len(x_all)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    x_train_raw = x_all[train_indices]
    x_test_raw = x_all[test_indices]
    y_train = y_all[train_indices]
    y_test = y_all[test_indices]
    
    # 标准化
    samples_train, seq_len, n_features = x_train_raw.shape
    x_train_flat = x_train_raw.reshape(-1, n_features)
    scaler = StandardScaler()
    scaler.fit(x_train_flat)
    
    x_train_scaled_flat = scaler.transform(x_train_flat)
    x_train = x_train_scaled_flat.reshape(samples_train, seq_len, n_features)
    
    samples_test, seq_len, n_features = x_test_raw.shape
    x_test_flat = x_test_raw.reshape(-1, n_features)
    x_test_scaled_flat = scaler.transform(x_test_flat)
    x_test = x_test_scaled_flat.reshape(samples_test, seq_len, n_features)
    
    return x_train, y_train, x_test, y_test

def train_causal_tsf(earnings, stock_config_file="config/stock_config.json", train_config_file="config/train_config.json"):
    """ 
    Causal-TSF训练器
    """
    with open(stock_config_file, 'r', encoding='utf-8') as f:
        stock_config = js.load(f)
        
    with open(train_config_file, 'r', encoding='utf-8') as f:
        train_config = js.load(f)
        
    target_col = stock_config['target_stock']['name']
    seq_length = int(train_config['seq_length'])
    batch_size = int(train_config['batch_size'])
    epochs = int(train_config['epochs'])
    lr = float(train_config['lr'])
    target_shift = int(train_config['target_shift'])
    patience = int(train_config['patience'])
    min_delta = float(train_config['min_delta'])
    n_runs = int(train_config['n_runs'])
    train_ratio = float(train_config['train_ratio'])
    random_seed = int(train_config['random_seed'])
    
    metrics = {
        'r2': [], 'rmse': [], 'ic': [], 'rank_ic': []
    }
    
    x_train, y_train, x_test, y_test = prepare_time_series_data_for_causal_tsf(earnings, seq_length, target_col, target_shift, train_ratio, random_seed)
    
    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Causal-TSF Run {run+1}/{n_runs} Starting...")
        print(f"{'='*60}")
    
        # 数据加载器
        train_dataset = TimeSeriesDataset(x_train, y_train)
        test_dataset = TimeSeriesDataset(x_test, y_test)
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        # 初始化模型
        input_dim = x_train.shape[2]
        model = tsf.CausalTSF(input_dim).to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        best_train_r2 = float('-inf')
        epochs_without_improvement = 0
    
        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0.0
            train_true = []
            train_pred = []
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs, _, _ = model(x_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            
                train_true.append(y_batch.cpu().numpy())
                train_pred.append(outputs.detach().cpu().numpy().squeeze(-1))

            train_loss /= len(train_loader.dataset)
            train_true_np = np.concatenate(train_true, axis=0).flatten()
            train_pred_np = np.concatenate(train_pred, axis=0).flatten()
            train_r2 = r2_score(train_true_np, train_pred_np)
            train_rmse = np.sqrt(np.mean((train_true_np - train_pred_np)**2))
            train_ic = cal_pearsonr(train_pred_np, train_true_np)
            train_rank_ic = cal_spearmanr(train_pred_np, train_true_np)
        
            # 记录最佳模型
            if train_r2 > best_train_r2 + min_delta:
                best_train_r2 = train_r2
                torch.save(model.state_dict(), 'best_model.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # 早停检查
            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.')
                break
            
            print(f'Epoch of Causal-TSF[{epoch+1}/{epochs}], \n'
                  f'Train Results | R²: {train_r2:.4f} | RMSE: {train_rmse:.4f} | IC: {train_ic:.4f} | RankIC: {train_rank_ic:.4f}\n')
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_model.pth'))
        model.to(device)

        # 测试集评估
        model.eval()
        test_loss = 0.0
        test_true = []
        test_pred = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs, _, _ = model(x_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                test_loss += loss.item() * x_batch.size(0)
            
                test_true.append(y_batch.cpu().numpy())
                test_pred.append(outputs.detach().cpu().numpy().squeeze(-1))
            
        test_loss /= len(test_loader.dataset)
        test_true_np = np.concatenate(test_true, axis=0).flatten()
        test_pred_np = np.concatenate(test_pred, axis=0).flatten()
        test_r2 = r2_score(test_true_np, test_pred_np)
        test_rmse = np.sqrt(np.mean((test_true_np - test_pred_np)**2))
        test_ic = cal_pearsonr(test_pred_np, test_true_np)
        test_rank_ic = cal_spearmanr(test_pred_np, test_true_np)
    
        metrics['r2'].append(test_r2)
        metrics['rmse'].append(test_rmse)
        metrics['ic'].append(test_ic)
        metrics['rank_ic'].append(test_rank_ic)
        
        print(f"Run {run+1} Test Results | R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | IC: {test_ic:.4f} | RankIC: {test_rank_ic:.4f}")
    
    return model, metrics
                
def train_tradition(earnings, stock_config_file="config/stock_config.json", train_config_file="config/train_config.json"):
    """ 
    非因果时间序列预测模型训练器
    """
    with open(stock_config_file, 'r', encoding='utf-8') as f:
        stock_config = js.load(f)
        
    with open(train_config_file, 'r', encoding='utf-8') as f:
        train_config = js.load(f)
        
    target_col = stock_config['target_stock']['name']
    seq_length = int(train_config['seq_length'])
    batch_size = int(train_config['batch_size'])
    epochs = int(train_config['epochs'])
    lr = float(train_config['lr'])
    target_shift = int(train_config['target_shift'])
    patience = int(train_config['patience'])
    min_delta = float(train_config['min_delta'])
    n_runs = int(train_config['n_runs'])
    train_ratio = float(train_config['train_ratio'])
    random_seed = int(train_config['random_seed'])
    
    metrics = {
        'r2': [], 'rmse': [], 'ic': [], 'rank_ic': []
    }
    
    x_train, y_train, x_test, y_test = prepare_time_series_data_for_tradition(earnings, seq_length, target_col, target_shift, train_ratio, random_seed)
    
    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Tradition Run {run+1}/{n_runs} Starting...")
        print(f"{'='*60}")
    
        train_dataset = TimeSeriesDataset(x_train, y_train)
        test_dataset = TimeSeriesDataset(x_test, y_test)
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
        input_dim = x_train.shape[2]
        model = tra.Tradition(input_dim).to(device)
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
        best_train_r2 = float('-inf')
        epochs_without_improvement = 0
    
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_true = []
            train_pred = []
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)
            
                train_true.append(y_batch.cpu().numpy())
                train_pred.append(outputs.detach().cpu().numpy().squeeze(-1))
            
            train_loss /= len(train_loader.dataset)
            train_true_np = np.concatenate(train_true, axis=0).flatten()
            train_pred_np = np.concatenate(train_pred, axis=0).flatten()
            train_r2 = r2_score(train_true_np, train_pred_np)
            train_rmse = np.sqrt(np.mean((train_true_np - train_pred_np)**2))
            train_ic = cal_pearsonr(train_pred_np, train_true_np)
            train_rank_ic = cal_spearmanr(train_pred_np, train_true_np)
        
            if train_r2 > best_train_r2 + min_delta:
                best_train_r2 = train_r2
                torch.save(model.state_dict(), 'best_tradition_model.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping triggered at epoch {epoch+1}. No improvement for {patience} epochs.')
                break
            
            print(f'Epoch of Tradition[{epoch+1}/{epochs}], \n'
                  f'Train Results | R²: {train_r2:.4f} | RMSE: {train_rmse:.4f} | IC: {train_ic:.4f} | RankIC: {train_rank_ic:.4f}\n')
        
        model.load_state_dict(torch.load('best_tradition_model.pth'))
    
        model.to(device)
        model.eval()
        test_loss = 0.0
        test_true = []
        test_pred = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                test_loss += loss.item() * x_batch.size(0)
            
                test_true.append(y_batch.cpu().numpy())
                test_pred.append(outputs.detach().cpu().numpy().squeeze(-1))
            
        test_loss /= len(test_loader.dataset)
        test_true_np = np.concatenate(test_true, axis=0).flatten()
        test_pred_np = np.concatenate(test_pred, axis=0).flatten()
        test_r2 = r2_score(test_true_np, test_pred_np)
        test_rmse = np.sqrt(np.mean((test_true_np - test_pred_np)**2))
        test_ic = cal_pearsonr(test_pred_np, test_true_np)
        test_rank_ic = cal_spearmanr(test_pred_np, test_true_np)
    
        metrics['r2'].append(test_r2)
        metrics['rmse'].append(test_rmse)
        metrics['ic'].append(test_ic)
        metrics['rank_ic'].append(test_rank_ic)
        
        print(f"Run {run+1} Test Results | R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | IC: {test_ic:.4f} | RankIC: {test_rank_ic:.4f}")
    
    print(f"Test samples: {len(y_test)}")
    
    return model, metrics

def train_transformer(earnings, stock_config_file="config/stock_config.json", train_config_file="config/train_config.json"):
    """
    Transformer模型训练器
    """
    with open(stock_config_file, 'r', encoding='utf-8') as f:
        stock_config = js.load(f)
    with open(train_config_file, 'r', encoding='utf-8') as f:
        train_config = js.load(f)

    target_col = stock_config['target_stock']['name']
    seq_length = int(train_config['seq_length'])
    batch_size = int(train_config['batch_size'])
    epochs = int(train_config['epochs'])
    lr = float(train_config['lr'])
    target_shift = int(train_config['target_shift'])
    patience = int(train_config['patience'])
    min_delta = float(train_config['min_delta'])
    n_runs = int(train_config['n_runs'])
    train_ratio = float(train_config['train_ratio'])
    random_seed = int(train_config['random_seed'])
    d_model = int(train_config['d_model'])
    nhead = int(train_config['nhead'])
    num_layers = int(train_config['num_layers'])
    dropout = float(train_config['dropout'])

    x_train, y_train, x_test, y_test = prepare_time_series_data_for_tradition(
        earnings, seq_length, target_col, target_shift, train_ratio, random_seed
    )

    metrics = {'r2': [], 'rmse': [], 'ic': [], 'rank_ic': []}

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Transformer Run {run+1}/{n_runs} Starting...")
        print(f"{'='*60}")

        train_dataset = TimeSeriesDataset(x_train, y_train)
        test_dataset = TimeSeriesDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_dim = x_train.shape[2]
        model = trans.TransformerNet(input_dim, d_model, nhead, num_layers, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_train_r2 = float('-inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_true, train_pred = [], []
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)

                train_true.append(y_batch.cpu().numpy())
                train_pred.append(outputs.detach().cpu().numpy().squeeze(-1))

            train_loss /= len(train_loader.dataset)
            train_true_np = np.concatenate(train_true)
            train_pred_np = np.concatenate(train_pred)
            train_r2 = r2_score(train_true_np, train_pred_np)
            train_rmse = np.sqrt(np.mean((train_true_np - train_pred_np)**2))
            train_ic = cal_pearsonr(train_pred_np, train_true_np)
            train_rank_ic = cal_spearmanr(train_pred_np, train_true_np)

            if train_r2 > best_train_r2 + min_delta:
                best_train_r2 = train_r2
                torch.save(model.state_dict(), 'best_transformer_model.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

            print(f'Epoch of Transformer[{epoch+1}/{epochs}], \n'
                  f'Train Results | R²: {train_r2:.4f} | RMSE: {train_rmse:.4f} | IC: {train_ic:.4f} | RankIC: {train_rank_ic:.4f}\n')


        model.load_state_dict(torch.load('best_transformer_model.pth'))
        model.to(device)
        model.eval()
        test_true, test_pred = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                test_true.append(y_batch.cpu().numpy())
                test_pred.append(outputs.cpu().numpy().squeeze(-1))
        test_true_np = np.concatenate(test_true)
        test_pred_np = np.concatenate(test_pred)
        test_r2 = r2_score(test_true_np, test_pred_np)
        test_rmse = np.sqrt(np.mean((test_true_np - test_pred_np)**2))
        test_ic = cal_pearsonr(test_pred_np, test_true_np)
        test_rank_ic = cal_spearmanr(test_pred_np, test_true_np)

        metrics['r2'].append(test_r2)
        metrics['rmse'].append(test_rmse)
        metrics['ic'].append(test_ic)
        metrics['rank_ic'].append(test_rank_ic)

        print(f"Run {run+1} Test Results | R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | IC: {test_ic:.4f} | RankIC: {test_rank_ic:.4f}")

    return model, metrics

def train_lstm(earnings, stock_config_file="config/stock_config.json", train_config_file="config/train_config.json"):
    """
    双向LSTM模型训练器
    """
    with open(stock_config_file, 'r', encoding='utf-8') as f:
        stock_config = js.load(f)
    with open(train_config_file, 'r', encoding='utf-8') as f:
        train_config = js.load(f)

    target_col = stock_config['target_stock']['name']
    hidden_dim = int(train_config['hidden_dim'])
    seq_length = int(train_config['seq_length'])
    batch_size = int(train_config['batch_size'])
    epochs = int(train_config['epochs'])
    lr = float(train_config['lr'])
    target_shift = int(train_config['target_shift'])
    patience = int(train_config['patience'])
    min_delta = float(train_config['min_delta'])
    n_runs = int(train_config['n_runs'])
    train_ratio = float(train_config['train_ratio'])
    random_seed = int(train_config['random_seed'])
    num_layers = int(train_config['num_layers'])
    dropout = float(train_config['dropout'])

    x_train, y_train, x_test, y_test = prepare_time_series_data_for_tradition(
        earnings, seq_length, target_col, target_shift, train_ratio, random_seed
    )

    metrics = {'r2': [], 'rmse': [], 'ic': [], 'rank_ic': []}

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"LSTM Run {run+1}/{n_runs} Starting...")
        print(f"{'='*60}")

        train_dataset = TimeSeriesDataset(x_train, y_train)
        test_dataset = TimeSeriesDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_dim = x_train.shape[2]
        model = lstm.LSTMNet(input_dim, hidden_dim, num_layers, dropout).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_train_r2 = float('-inf')
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_true, train_pred = [], []
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch.view(-1, 1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x_batch.size(0)

                train_true.append(y_batch.cpu().numpy())
                train_pred.append(outputs.detach().cpu().numpy().squeeze(-1))

            train_loss /= len(train_loader.dataset)
            train_true_np = np.concatenate(train_true)
            train_pred_np = np.concatenate(train_pred)
            train_r2 = r2_score(train_true_np, train_pred_np)
            train_rmse = np.sqrt(np.mean((train_true_np - train_pred_np)**2))
            train_ic = cal_pearsonr(train_pred_np, train_true_np)
            train_rank_ic = cal_spearmanr(train_pred_np, train_true_np)

            if train_r2 > best_train_r2 + min_delta:
                best_train_r2 = train_r2
                torch.save(model.state_dict(), 'best_lstm_model.pth')
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

            print(f'Epoch of LSTM[{epoch+1}/{epochs}], \n'
                  f'Train Results | R²: {train_r2:.4f} | RMSE: {train_rmse:.4f} | IC: {train_ic:.4f} | RankIC: {train_rank_ic:.4f}\n')

        model.load_state_dict(torch.load('best_lstm_model.pth', map_location=device))
        model.to(device)
        model.eval()
        test_true, test_pred = [], []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                test_true.append(y_batch.cpu().numpy())
                test_pred.append(outputs.cpu().numpy().squeeze(-1))
        test_true_np = np.concatenate(test_true)
        test_pred_np = np.concatenate(test_pred)
        test_r2 = r2_score(test_true_np, test_pred_np)
        test_rmse = np.sqrt(np.mean((test_true_np - test_pred_np)**2))
        test_ic = cal_pearsonr(test_pred_np, test_true_np)
        test_rank_ic = cal_spearmanr(test_pred_np, test_true_np)

        metrics['r2'].append(test_r2)
        metrics['rmse'].append(test_rmse)
        metrics['ic'].append(test_ic)
        metrics['rank_ic'].append(test_rank_ic)

        print(f"Run {run+1} Test Results | R²: {test_r2:.4f} | RMSE: {test_rmse:.4f} | IC: {test_ic:.4f} | RankIC: {test_rank_ic:.4f}")

    return model, metrics