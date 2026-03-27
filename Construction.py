import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import matplotlib.pyplot as plt
import json as js
from collections import deque
import pickle
import os

def build_granger_causal_network(earnings, stock_config_file='config/stock_config.json', max_lags=8, significance_level=0.1, cache_dir='./causal_cache'):
    """ 
    构建因果网络
    """
    # 创建缓存目录
    os.makedirs(cache_dir, exist_ok=True)
    
    # 缓存文件名
    cache_file = os.path.join(cache_dir, f"granger_causal_{significance_level}.pkl")
    
    # 尝试加载已有缓存
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        adj_matrix = cached['adj_matrix']
        causal_vector = cached['causal_vector']
        print(f"已从缓存加载因果网络（显著性水平={significance_level}）")
    else:
        # 未找到缓存，执行完整计算
        with open(stock_config_file, 'r', encoding='utf-8') as f:
            stock_config = js.load(f)
        
        target_col = stock_config['target_stock']['name']
        target_idx = earnings.columns.get_loc(target_col)
        vars_num = len(earnings.columns)
        
        causal_vector = []
        variable_list = earnings.columns
        
        # 构建因果矩阵
        for reason_variable in variable_list:
            for result_variable in variable_list:
                if reason_variable != result_variable:
                    test_data = np.column_stack([earnings[reason_variable], earnings[result_variable]])
                    granger_test_result = grangercausalitytests(test_data, maxlag=max_lags, verbose=False)
                    p_value = granger_test_result[max_lags][0]['ssr_ftest'][1]
                    
                    if p_value < significance_level:
                        causal_vector.append((reason_variable, result_variable, p_value))
        
        adj_matrix = pd.DataFrame(False, index=variable_list, columns=variable_list)
        
        for reason, result, _ in causal_vector:
            adj_matrix.loc[reason, result] = True
        
        # 保存到缓存
        with open(cache_file, 'wb') as f:
            pickle.dump({'adj_matrix': adj_matrix, 'causal_vector': causal_vector}, f)
        print(f"因果网络计算完成，已保存至缓存（显著性水平={significance_level}）")
    
    # 根据当前配置的目标变量提取因果变量
    with open(stock_config_file, 'r', encoding='utf-8') as f:
        stock_config = js.load(f)
    target_col = stock_config['target_stock']['name']
    target_idx = earnings.columns.get_loc(target_col)
       
    # 提取所有与目标变量有因果关系的变量             
    reverse_graph = {}
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix.iloc[i, j]:
                reverse_graph.setdefault(j, []).append(i)
    
    visited = set([target_idx])
    queue = deque([target_idx])
    
    while queue:
        node = queue.popleft()
        if node in reverse_graph:
            for neighbor in reverse_graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    causal_indices = sorted(earnings.columns[i] for i in sorted(visited))
    
    print("因果矩阵已生成：")
    print(adj_matrix)
    print("影响目标变量的因果变量为：")
    print(causal_indices)
    
    return causal_indices, causal_vector, adj_matrix

def plot_correlation_heatmap(data, target="宁德时代", top_n=30, figsize=(12,10), cmap='RdBu_r', annot=False, 
                             title="部分相关股票热力图", save_path="./heatmap.png"):
    """
    绘制与目标股票相关性最高的前 top_n 只股票的热力图。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    corr = data.corr()
    target_corr_abs = corr[target].abs().sort_values(ascending=False)
    plot_vars = target_corr_abs.head(top_n).index.tolist()

    # 计算子集相关系数矩阵
    corr_sub = data[plot_vars].corr()

    # 绘制热力图
    plt.figure(figsize=figsize)
    sns.heatmap(corr_sub, annot=annot, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')

            
    
    
    
    

            