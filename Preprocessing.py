import pandas as pd
import json as js
import baostock as bs
from datetime import datetime
import os

def get_stock_data(symbol, start_date, end_date):
    adjust_flag = "3"
    
    rs = bs.query_history_k_data_plus(
        symbol, 
        "date,code,open,high,low,close,preclose,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag=adjust_flag
    )
    
    data = []
    while (rs.error_code == '0') & rs.next():
        data.append(rs.get_row_data())
    df = pd.DataFrame(data, columns=rs.fields)
    
    return df

def data_fetch_and_save(start_date="2019-01-01", end_date="2025-12-31", cache_dir="./date_cache", config_file="config/stock_config.json"):
    """
    获取并存储数据
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = js.load(f)
    
    bs.login()
    
    # 获取目标股票数据
    target_symbol = config['target_stock']['symbol']
    target_price_data = get_stock_data(target_symbol, start_date, end_date)
    
    # 获取大盘数据
    market_price_data = []
    market_names = []
    for stock in config['market_stock']:
        market_symbol = stock['symbol']
        name = stock['name']
        market_price_data.append(get_stock_data(market_symbol, start_date, end_date))
        market_names.append(name)
    
    # 获取相关企业股票数据
    price_related_data = []
    related_stock_names = []
    for stock in config['related_stock']:
        symbol = stock['symbol']
        name = stock['name']
        price_related_data.append(get_stock_data(symbol, start_date, end_date))
        related_stock_names.append(name)
        
    os.makedirs(cache_dir, exist_ok=True)
    
    bs.logout()
    
    # 保存目标股票数据
    target_price_data.to_csv(f"{cache_dir}/target_data.csv", index=False, encoding='utf-8-sig')
    
    # 保存大盘数据
    for i, (df, name) in enumerate(zip(market_price_data, market_names), start=1):
        df.to_csv(f"{cache_dir}/market_data_{i}_{name}.csv", index=False, encoding='utf-8-sig')
    
    # 保存相关企业数据
    for i, (df, name) in enumerate(zip(price_related_data, related_stock_names), start=1):
        df.to_csv(f"{cache_dir}/related_stock_{i}_{name}.csv", index=False, encoding='utf-8-sig')
    
    # 保存配置和元数据为TXT文件
    metadata = {
        'config': config,
        'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'related_stock': related_stock_names,
        'files': {
            'target_data': 'target_data.csv',
            'market_data': [f'market_data_{i}_{name}.csv' for i, name in enumerate(market_names, start=1)],
            'related_stock_files': [f'related_stock_{i}_{name}.csv' for i, name in enumerate(related_stock_names, start=1)]
        }
    }
    
    with open(f"{cache_dir}/metadata.txt", 'w', encoding='utf-8') as f:
        js.dump(metadata, f, ensure_ascii=False, indent=2)

def load_data(cache_dir="./date_cache"):
    """
    加载已存储的数据
    """
    # 读取元数据
    with open(f"{cache_dir}/metadata.txt", 'r', encoding='utf-8') as f:
        metadata = js.load(f)
        
    # 加载目标股票数据
    target_price_data = pd.read_csv(f"{cache_dir}/target_data.csv", encoding='utf-8-sig')
    
    # 加载大盘数据
    market_price_data = []
    for filename in metadata['files']['market_data']:
        df = pd.read_csv(f"{cache_dir}/{filename}", encoding='utf-8-sig')
        market_price_data.append((filename, df))
    
    # 加载相关企业数据
    price_related_data = []
    for filename in metadata['files']['related_stock_files']:
        df = pd.read_csv(f"{cache_dir}/{filename}", encoding='utf-8-sig')
        price_related_data.append((filename, df))
    
    return target_price_data, market_price_data, price_related_data
    
def data_merge(target_price_data, market_price_data, price_related_data):
    """
    准备用于因果分析的统一数据集
    """ 
    # 统一数据格式
    target_price_data['date'] = pd.to_datetime(target_price_data['date'])
                                                                          
    # 处理主要数据
    price_dict = {
        '宁德时代': target_price_data.set_index('date')['close'].pct_change()
    }
    
    # 处理大盘数据
    if market_price_data:
        for i, (name, df) in enumerate(market_price_data, start=1):
            df['date'] = pd.to_datetime(df['date'])
            name = name[:-4]
            price_dict[f'{name}'] = df.set_index('date')['close'].pct_change()
            
    # 处理相关企业股票数据
    if price_related_data:
        for i, (name, df) in enumerate(price_related_data, start=1):
            df['date'] = pd.to_datetime(df['date'])
            name = name[:-4]
            price_dict[f'{name}'] = df.set_index('date')['close'].pct_change()
    
    earnings = pd.concat(price_dict, axis=1)
    earnings = earnings.dropna(how='any')

    print("合并后的价格数据预览：")
    print(earnings)

    return earnings

