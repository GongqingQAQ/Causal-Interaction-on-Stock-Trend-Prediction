import pandas as pd
import numpy as np
import Preprocessing as pre
import Construction as con
from datetime import datetime
import Trainer as train
import logging
import sys

pd.set_option('display.max_columns', None)

# 记录日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 格式化
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

# 文件处理
log_file = f"log/{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台处理
console_handler = logging.StreamHandler(sys.__stdout__)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''
        
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())
            
    def flush(self):
        pass
    
sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

# 加载数据
target_price_data, market_price_data, price_related_data = pre.load_data()

# 合并数据
earnings = pre.data_merge(target_price_data, market_price_data, price_related_data)

# 构建因果网络
causal_indices, causal_vector, adj_matrix = con.build_granger_causal_network(earnings)

available_cols = [col for col in causal_indices if col in earnings.columns]
filtered_earnings = earnings[available_cols].copy()

# 因果网络可视化
con.plot_correlation_heatmap(filtered_earnings)

# 模型训练
model_causal_tsf, metrics_causal_tsf = train.train_causal_tsf(filtered_earnings)
model_tradition, metrics_tradition = train.train_tradition(filtered_earnings)
model_transformer, metrics_transformer = train.train_transformer(filtered_earnings)

# 输出结果
def print_results(model_name, metrics):
    results = {}
    for key in metrics:
        results[f'avg_{key}'] = np.mean(metrics[key])
        results[f'std_{key}'] = np.std(metrics[key])
    print(f"\n{'='*60}")
    print(f"{model_name} Final Results (Avg ± Std):")
    print(f"R²:       {results['avg_r2']:.4f} ± {results['std_r2']:.4f}")
    print(f"RMSE:     {results['avg_rmse']:.4f} ± {results['std_rmse']:.4f}")
    print(f"IC:       {results['avg_ic']:.4f} ± {results['std_ic']:.4f} {'★' if results['avg_ic'] > 0.03 else ''}")
    print(f"RankIC:   {results['avg_rank_ic']:.4f} ± {results['std_rank_ic']:.4f} {'★' if results['avg_rank_ic'] > 0.03 else ''}")
    print(f"{'='*60}")

print_results("Causal-TSF", metrics_causal_tsf)
print_results("Tradition", metrics_tradition)
print_results("Transformer", metrics_transformer)


