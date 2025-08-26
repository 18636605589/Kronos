#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据获取模块
支持从多个数据源获取股票数据并保存为Kronos格式
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StockDataFetcher:
    """股票数据获取器"""
    
    def __init__(self, data_dir="my_stock_predictor/stock_data"):
        """
        初始化数据获取器
        
        Args:
            data_dir (str): 数据保存目录
        """
        self.data_dir = data_dir
        self.ensure_data_dir()
        
    def ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建数据目录: {self.data_dir}")
    
    def fetch_from_akshare(self, symbol, start_date=None, end_date=None, period='5'):
        """
        使用akshare获取股票数据
        
        Args:
            symbol (str): 股票代码，如'000001'
            start_date (str): 开始日期，格式'YYYY-MM-DD'
            end_date (str): 结束日期，格式'YYYY-MM-DD'
            period (str): 周期，'1'=1分钟, '5'=5分钟, '15'=15分钟, '30'=30分钟, '60'=1小时, 'D'=日线
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            import akshare as ak
            
            # 获取股票名称
            stock_name = "Unknown"
            try:
                stock_info_df = ak.stock_individual_info_em(symbol=symbol)
                stock_name = stock_info_df.loc[stock_info_df['item'] == '股票简称', 'value'].values[0]
                print(f"获取到股票名称: {stock_name}")
            except Exception as e:
                print(f"警告: 获取股票名称失败: {e}")

            # 设置默认日期范围
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            print(f"正在获取 {symbol} 的 {period}分钟数据...")
            print(f"时间范围: {start_date} 到 {end_date}")
            
            # 获取股票数据
            if period == 'D':
                # 日线数据
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                      start_date=start_date, end_date=end_date, 
                                      adjust="qfq")
            else:
                # 分钟数据
                df = ak.stock_zh_a_hist_min_em(symbol=symbol, period=period, 
                                             start_date=start_date, end_date=end_date, 
                                             adjust='qfq')
            
            # 重命名列以匹配Kronos格式
            column_mapping = {
                '日期': 'timestamps',
                '时间': 'timestamps',
                '开盘': 'open',
                '最高': 'high', 
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 处理时间戳
            if 'timestamps' in df.columns:
                if '日期' in df.columns and '时间' in df.columns:
                    # 合并日期和时间
                    df['timestamps'] = df['日期'] + ' ' + df['时间']
                elif '日期' in df.columns:
                    df['timestamps'] = df['日期'] + ' 00:00:00'
                
                df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            # 确保所有必需的列都存在
            required_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'amount':
                        # 如果没有成交额，用成交量*收盘价估算
                        df['amount'] = df['volume'] * df['close']
                    else:
                        print(f"警告: 缺少列 {col}")
            
            # 选择并排序列
            df = df[required_columns].copy()
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除无效数据
            df = df.dropna()
            
            print(f"成功获取 {len(df)} 条数据")
            return df, stock_name
            
        except ImportError:
            print("错误: 请先安装akshare: pip install akshare")
            return None, None
        except Exception as e:
            print(f"获取数据时出错: {e}")
            return None, None
    
    def fetch_from_yfinance(self, symbol, start_date=None, end_date=None, interval='5m'):
        """
        使用yfinance获取股票数据（适用于美股等）
        
        Args:
            symbol (str): 股票代码，如'AAPL'
            start_date (str): 开始日期
            end_date (str): 结束日期
            interval (str): 时间间隔，'1m', '5m', '15m', '30m', '1h', '1d'
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            import yfinance as yf
            
            # 设置默认日期范围
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            print(f"正在获取 {symbol} 的数据...")
            print(f"时间范围: {start_date} 到 {end_date}")
            
            # 获取股票 Ticker
            ticker = yf.Ticker(symbol)

            # 获取股票名称
            stock_name = "Unknown"
            try:
                info = ticker.info
                stock_name = info.get('shortName', info.get('longName', 'Unknown'))
                print(f"获取到股票名称: {stock_name}")
            except Exception as e:
                print(f"警告: 获取股票名称失败: {e}")

            # 获取股票数据
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            # 重置索引，将日期变为列
            df = df.reset_index()
            
            # 重命名列
            column_mapping = {
                'Datetime': 'timestamps',
                'Date': 'timestamps',
                'Open': 'open',
                'High': 'high',
                'Low': 'low', 
                'Close': 'close',
                'Volume': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # 添加成交额列（用成交量*收盘价估算）
            df['amount'] = df['volume'] * df['close']
            
            # 确保所有必需的列都存在
            required_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
            df = df[required_columns].copy()
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 删除无效数据
            df = df.dropna()
            
            print(f"成功获取 {len(df)} 条数据")
            return df, stock_name
            
        except ImportError:
            print("错误: 请先安装yfinance: pip install yfinance")
            return None, None
        except Exception as e:
            print(f"获取数据时出错: {e}")
            return None, None
    
    def save_data(self, df, symbol, period, stock_name, latest_timestamp):
        """
        保存数据到CSV文件和元数据到JSON文件
        
        Args:
            df (pd.DataFrame): 股票数据
            symbol (str): 股票代码
            period (str): 时间周期
            stock_name (str): 股票名称
            latest_timestamp (pd.Timestamp): 最新数据的时间戳
            
        Returns:
            tuple: (保存的CSV文件路径, 元数据)
        """
        if df is None or df.empty:
            print("没有数据可保存")
            return None, None
        
        # 1. 创建以股票代码命名的文件夹
        stock_dir = os.path.join(self.data_dir, symbol)
        os.makedirs(stock_dir, exist_ok=True)
        
        # 2. 保存数据到CSV文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{period}_{timestamp}.csv"
        filepath = os.path.join(stock_dir, filename)
        
        df.to_csv(filepath, index=False)
        print(f"数据已保存到: {filepath}")
        
        # 3. 准备并保存元数据
        metadata = {
            'symbol': symbol,
            'stock_name': stock_name,
            'latest_timestamp': latest_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'data_points': len(df),
            'period': period,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_file': filename
        }
        meta_filepath = os.path.join(stock_dir, 'metadata.json')
        with open(meta_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"元数据已保存到: {meta_filepath}")
        
        return filepath, metadata
    
    def get_stock_data(self, symbol, source='akshare', start_date=None, end_date=None, period='5', save=True, force_refetch=False):
        """
        获取股票数据，优先从本地缓存加载。
        
        Args:
            symbol (str): 股票代码
            source (str): 数据源，'akshare' 或 'yfinance'
            start_date (str): 开始日期
            end_date (str): 结束日期
            period (str): 时间周期
            save (bool): 是否保存数据
            force_refetch (bool): 是否强制重新从网络获取数据，忽略缓存。
        
        Returns:
            tuple: (数据DataFrame, 文件路径, 元数据)
        """
        print(f"开始获取股票 {symbol} 的数据...")
        
        stock_dir = os.path.join(self.data_dir, symbol)
        meta_filepath = os.path.join(stock_dir, 'metadata.json')

        # --- 新增：数据缓存加载逻辑 ---
        if not force_refetch and os.path.exists(meta_filepath):
            print(f"🔍 找到股票 {symbol} 的缓存元数据，尝试加载...")
            try:
                with open(meta_filepath, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                data_filename = metadata.get('data_file')
                if data_filename:
                    filepath = os.path.join(stock_dir, data_filename)
                    if os.path.exists(filepath):
                        print(f"✅ 成功加载缓存数据: {filepath}")
                        df = pd.read_csv(filepath)
                        # 加载后，确保timestamps列是datetime对象
                        df['timestamps'] = pd.to_datetime(df['timestamps'])
                        return df, filepath, metadata
                print("⚠️ 缓存元数据不完整或数据文件丢失。")
            except Exception as e:
                print(f"⚠️ 加载缓存失败: {e}")

        print("ℹ️ 未找到有效缓存或已强制刷新，将从网络获取新数据。")
        # --- 缓存逻辑结束 ---
        
        # 根据数据源获取数据
        df, stock_name = None, "Unknown"
        if source.lower() == 'akshare':
            df, stock_name = self.fetch_from_akshare(symbol, start_date, end_date, period)
        elif source.lower() == 'yfinance':
            df, stock_name = self.fetch_from_yfinance(symbol, start_date, end_date, period)
        else:
            print(f"不支持的数据源: {source}")
            return None, None, None
        
        if df is None or df.empty:
            print("获取数据失败")
            return None, None, None
        
        # 保存数据
        filepath, metadata = None, None
        if save:
            period_str = f"{period}min" if period != 'D' else "daily"
            latest_timestamp = df['timestamps'].iloc[-1]
            filepath, metadata = self.save_data(df, symbol, period_str, stock_name, latest_timestamp)
        
        return df, filepath, metadata

def main():
    """主函数示例"""
    # 创建数据获取器
    fetcher = StockDataFetcher()
    
    print("=" * 50)
    print("获取A股数据")
    print("=" * 50)
    
    # 获取平安银行5分钟数据
    df_a, filepath_a, metadata_a = fetcher.get_stock_data(
        symbol='000001',  # 平安银行
        source='akshare',
        start_date='2024-01-01',
        end_date='2024-01-31',
        period='5',
        save=True
    )
    
    if df_a is not None:
        print(f"数据预览:")
        print(df_a.head())
        print(f"\n元数据:")
        print(json.dumps(metadata_a, indent=2, ensure_ascii=False))
    
    # 示例2: 获取美股数据
    # print("\n" + "=" * 50)
    # print("示例2: 获取美股数据")
    # print("=" * 50)
    
    # # 获取苹果公司5分钟数据
    # df_us, filepath_us, metadata_us = fetcher.get_stock_data(
    #     symbol='AAPL',  # 苹果公司
    #     source='yfinance',
    #     start_date='2024-01-01',
    #     end_date='2024-01-31',
    #     period='5m',
    #     save=True
    # )
    
    # if df_us is not None:
    #     print(f"数据预览:")
    #     print(df_us.head())
    #     print(f"\n元数据:")
    #     print(json.dumps(metadata_us, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
