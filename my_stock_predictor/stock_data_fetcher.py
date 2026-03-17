#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票数据获取模块
支持从多个数据源获取股票数据并保存为Kronos格式

使用方法:
    python stock_data_fetcher.py --symbol 000001 --source akshare --period 5
    python stock_data_fetcher.py --symbol AAPL --source yfinance --period 5m --days 365
"""

import argparse
import pandas as pd
import numpy as np
import requests
import json
import time
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from constants import (
    TRADING_MINUTES_PER_DAY,
    TRADING_DAYS_RATIO,
    CHUNK_DAYS_MAP,
    MAX_ATTEMPTS_MAP,
    REQUEST_DELAY_MAP,
    DATA_AMOUNT_CHECK_RATIO,
    MIN_DATA_FOR_CHUNK,
    MAX_CONSECUTIVE_EMPTY
)

class StockDataFetcher:
    """股票数据获取器"""

    # 类变量：baostock 登录状态
    _baostock_logged_in = False

    # 默认数据目录使用绝对路径，避免依赖调用时的工作目录
    _default_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stock_data")

    def __init__(self, data_dir=None):
        """
        初始化数据获取器

        Args:
            data_dir (str): 数据保存目录，默认为脚本同级的 stock_data/ 目录
        """
        self.data_dir = data_dir or self._default_data_dir
        self.ensure_data_dir()
    
    @classmethod
    def _ensure_baostock_login(cls):
        """确保 baostock 已登录（单例模式）"""
        if not cls._baostock_logged_in:
            try:
                import baostock as bs
                lg = bs.login()
                if lg.error_code == '0':
                    cls._baostock_logged_in = True
                    print("✅ baostock 登录成功")
                else:
                    print(f"⚠️ baostock 登录失败: {lg.error_msg}")
                    return False
            except ImportError:
                print("错误: 请先安装 baostock: pip install baostock")
                return False
            except Exception as e:
                print(f"baostock 登录异常: {e}")
                return False
        return True
        
    def ensure_data_dir(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"创建数据目录: {self.data_dir}")

    def _format_period_label(self, period):
        """统一生成用于保存文件的周期字符串"""
        if period is None:
            return 'custom'

        period_str = str(period).strip()

        if period_str.upper() == 'D':
            return 'daily'

        lower_period = period_str.lower()
        if lower_period.endswith(('m', 'h', 'd')):
            return lower_period

        if period_str.isdigit():
            return f"{period_str}min"

        return lower_period
    
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
            
            # 获取股票数据（带重试机制）
            max_retries = 3
            for attempt in range(max_retries):
                try:
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
                    break  # 成功则跳出重试循环
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"第{attempt + 1}次尝试失败，{max_retries - attempt - 1}秒后重试...")
                        time.sleep(max_retries - attempt)  # 递增延迟
                    else:
                        print(f"重试{max_retries}次后仍然失败: {e}")
                        raise
            
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

    def fetch_from_baostock(self, symbol, start_date=None, end_date=None, period='5'):
        """
        使用baostock获取股票数据（百度金融数据）

        数据特点:
        - 百度金融数据平台
        - 数据质量较高，支持分钟线
        - 需要注册获取token
        - 支持A股、指数等

        Args:
            symbol (str): 股票代码，如'sh.600000' (浦发银行)
            start_date (str): 开始日期，格式'YYYY-MM-DD'
            end_date (str): 结束日期，格式'YYYY-MM-DD'
            period (str): 周期，'5'表示5分钟

        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            import baostock as bs

            # 设置默认日期范围
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

            print(f"正在获取 {symbol} 的数据 (baostock)...")
            print(f"时间范围: {start_date} 到 {end_date}")

            # 确保已登录（使用单例模式避免重复登录）
            if not self._ensure_baostock_login():
                return None, None

            try:
                # 获取股票数据
                # 注意：baostock的股票代码格式需要转换
                if '.' not in symbol:
                    # 如果是纯数字，添加市场前缀
                    if symbol.startswith(('000', '001', '002', '003', '300', '301')):
                        symbol = f"sz.{symbol}"  # 创业板/中小板
                    elif symbol.startswith(('600', '601', '603', '605', '688')):
                        symbol = f"sh.{symbol}"  # 主板/科创板
                    else:
                        symbol = f"sh.{symbol}"  # 默认沪市

                # 根据周期设置frequency
                freq_map = {
                    '1': '1',
                    '5': '5',
                    '15': '15',
                    '30': '30',
                    '60': '60',
                    'D': 'd'
                }
                frequency = freq_map.get(period, '5')

                if frequency in ['d', 'w', 'm']:
                    columns_str = "date,open,high,low,close,volume,amount"
                else:
                    columns_str = "date,time,open,high,low,close,volume,amount"

                rs = bs.query_history_k_data_plus(
                    symbol,
                    columns_str,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency,
                    adjustflag="2"  # 前复权调整 (1=后复权, 2=前复权, 3=不复权)
                )

                if rs.error_code != '0':
                    print(f"获取数据失败: {rs.error_msg}")
                    return None, None

                # 处理数据
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())

                print(f"baostock 返回数据行数: {len(data_list)}")
                if len(data_list) > 0 and len(data_list) <= 3:
                    print(f"调试: 数据样例: {data_list[0]}")

                if not data_list:
                    print("未获取到任何数据")
                    return None, None

                # 转换为DataFrame
                if frequency in ['d', 'w', 'm']:
                    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
                else:
                    columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount']
                df = pd.DataFrame(data_list, columns=columns)

                # 处理时间戳（优化：使用向量化操作）
                def parse_timestamp_vectorized(date_series, time_series):
                    """向量化解析时间戳"""
                    timestamps = []
                    for date_val, time_val in zip(date_series, time_series):
                        try:
                            time_str = str(time_val).strip()
                            date_str = str(date_val).strip()

                            # 处理 baostock 的不同时间格式
                            if len(time_str) >= 14 and time_str.isdigit():
                                # YYYYMMDDHHMMSS 格式 (如: 20250512093500000)
                                try:
                                    year = time_str[:4]
                                    month = time_str[4:6]
                                    day = time_str[6:8]
                                    hour = time_str[8:10]
                                    minute = time_str[10:12]
                                    second = time_str[12:14]
                                    datetime_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                                    timestamps.append(pd.to_datetime(datetime_str))
                                except Exception:
                                    timestamps.append(pd.to_datetime(date_str))
                            else:
                                # 尝试其他格式
                                try:
                                    combined_str = f"{date_str} {time_str}"
                                    timestamps.append(pd.to_datetime(combined_str))
                                except Exception:
                                    timestamps.append(pd.to_datetime(date_str))
                        except Exception:
                            try:
                                timestamps.append(pd.to_datetime(date_val))
                            except:
                                timestamps.append(pd.NaT)
                    return pd.Series(timestamps)

                if 'time' in df.columns:
                    df['timestamps'] = parse_timestamp_vectorized(df['date'], df['time'])
                else:
                    df['timestamps'] = pd.to_datetime(df['date'])
                
                # 检查是否有无效时间戳
                invalid_timestamps = df['timestamps'].isna().sum()
                if invalid_timestamps > 0:
                    print(f"⚠️ 警告: 有 {invalid_timestamps} 个无效时间戳，将使用日期字段")
                    # 用日期字段填充无效时间戳
                    invalid_mask = df['timestamps'].isna()
                    df.loc[invalid_mask, 'timestamps'] = pd.to_datetime(df.loc[invalid_mask, 'date'])

                # 选择并重命名列
                df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()

                # 数据类型转换
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

                # 删除无效数据
                df = df.dropna()

                print(f"成功获取 {len(df)} 条数据")
                return df, symbol

            except Exception as inner_e:
                print(f"获取数据时出错: {inner_e}")
                return None, None

        except ImportError:
            print("错误: 请先安装baostock: pip install baostock")
            print("并注册账号获取token: https://www.baostock.com/")
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
    
    def get_stock_data(self, symbol, source='baostock', start_date=None, end_date=None,
                       period='5', save=True, force_refetch=False,
                       min_fresh_days=None, fallback_days=365):
        """
        获取股票数据，优先从本地缓存加载。
        
        Args:
            symbol (str): 股票代码
            source (str): 数据源，'baostock'(默认), 'akshare', 'yfinance', 'tushare', 'jqdatasdk'
            start_date (str): 开始日期
            end_date (str): 结束日期
            period (str): 时间周期
            save (bool): 是否保存数据
            force_refetch (bool): 是否强制重新从网络获取数据，忽略缓存。
            min_fresh_days (int|None): 允许的最大数据滞后天数，超过后自动刷新
            fallback_days (int|None): 当数据过旧时重新拉取的时间跨度（单位：天）
        
        Returns:
            tuple: (数据DataFrame, 文件路径, 元数据)
        """
        print(f"开始获取股票 {symbol} 的数据...")
        
        stock_dir = os.path.join(self.data_dir, symbol)
        meta_filepath = os.path.join(stock_dir, 'metadata.json')

        refresh_needed = force_refetch
        stale_refresh_triggered = False
        cached_df = None
        cached_metadata = None
        query_start_date = start_date
        query_end_date = end_date

        # --- 新增：数据缓存加载逻辑 ---
        if not refresh_needed and os.path.exists(meta_filepath):
            print(f"🔍 找到股票 {symbol} 的缓存元数据，尝试加载...")
            try:
                with open(meta_filepath, 'r', encoding='utf-8') as f:
                    cached_metadata = json.load(f)
                
                # 验证周期是否匹配
                cached_period = cached_metadata.get('period')
                requested_period_label = self._format_period_label(period)
                
                if cached_period != requested_period_label:
                    print(f"⚠️ 缓存数据周期 ({cached_period}) 与请求周期 ({requested_period_label}) 不匹配，将重新获取。")
                    refresh_needed = True
                else:
                    data_filename = cached_metadata.get('data_file')
                    if data_filename:
                        filepath = os.path.join(stock_dir, data_filename)
                        if os.path.exists(filepath):
                            print(f"✅ 成功加载缓存数据: {filepath}")
                            cached_df = pd.read_csv(filepath)
                            # 加载后，确保timestamps列是datetime对象
                            cached_df['timestamps'] = pd.to_datetime(cached_df['timestamps'])

                            if min_fresh_days is not None:
                                freshness_threshold = datetime.now() - timedelta(days=min_fresh_days)
                                latest_timestamp = cached_df['timestamps'].max()
                                # 统一为 naive datetime 以兼容 yfinance 等返回时区感知时间戳的数据源
                                if hasattr(latest_timestamp, 'tzinfo') and latest_timestamp.tzinfo is not None:
                                    latest_timestamp = latest_timestamp.tz_localize(None)
                                if pd.isna(latest_timestamp) or latest_timestamp < freshness_threshold:
                                    refresh_needed = True
                                    stale_refresh_triggered = True
                                    if fallback_days is not None:
                                        query_start_date = (datetime.now() - timedelta(days=fallback_days)).strftime('%Y-%m-%d')
                                        query_end_date = datetime.now().strftime('%Y-%m-%d')
                                    print(
                                        f"⚠️ 缓存数据最新时间 {latest_timestamp} 早于 {min_fresh_days} 天前，"
                                        "将重新获取数据。"
                                    )
                                else:
                                    print("✅ 缓存数据满足新鲜度要求。")
                            
                            if not refresh_needed:
                                return cached_df, filepath, cached_metadata
                        else:
                            print("⚠️ 缓存数据文件不存在，准备重新获取。")
                            refresh_needed = True
                    else:
                        print("⚠️ 缓存元数据不完整或数据文件丢失。")
                        refresh_needed = True
            except Exception as e:
                print(f"⚠️ 加载缓存失败: {e}")

        if refresh_needed:
            print("ℹ️ 将从网络获取最新数据...")
            if stale_refresh_triggered and fallback_days is not None:
                query_start_date = (datetime.now() - timedelta(days=fallback_days)).strftime('%Y-%m-%d')
                query_end_date = datetime.now().strftime('%Y-%m-%d')
                print(f"   触发过期刷新，时间范围设置为: {query_start_date} 至 {query_end_date}")
            elif fallback_days is not None and query_start_date is None:
                query_start_date = (datetime.now() - timedelta(days=fallback_days)).strftime('%Y-%m-%d')
                query_end_date = datetime.now().strftime('%Y-%m-%d')
                print(f"   使用 fallback_days={fallback_days}，时间范围设置为: {query_start_date} 至 {query_end_date}")
            elif query_end_date is None:
                query_end_date = datetime.now().strftime('%Y-%m-%d')
        else:
            print("ℹ️ 未找到有效缓存，将从网络获取新数据。")
            if fallback_days is not None and query_start_date is None:
                query_start_date = (datetime.now() - timedelta(days=fallback_days)).strftime('%Y-%m-%d')
                print(f"   使用 fallback_days={fallback_days}，时间范围设置为: {query_start_date} 至 {query_end_date}")
            if query_end_date is None:
                query_end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"📅 最终查询时间范围: {query_start_date} 至 {query_end_date}")
        
        # 根据数据源获取数据（使用字典映射优化）
        source_map = {
            'akshare': self.fetch_from_akshare,
            'yfinance': self.fetch_from_yfinance,
            'baostock': self.fetch_from_baostock,
            'tushare': getattr(self, 'fetch_from_tushare', None),
            'jqdatasdk': getattr(self, 'fetch_from_jqdatasdk', None)
        }
        
        source_lower = source.lower()
        fetch_func = source_map.get(source_lower)
        
        if fetch_func is None:
            print(f"❌ 不支持的数据源: {source}")
            print("支持的数据源: akshare, yfinance, baostock, tushare, jqdatasdk")
            return None, None, None
        
        try:
            df, stock_name = fetch_func(symbol, query_start_date, query_end_date, period)
        except Exception as e:
            print(f"❌ 从 {source} 获取数据时出错: {e}")
            return None, None, None
        
        if df is None or df.empty:
            print("获取数据失败")
            return None, None, None

        # 检查是否需要分段拉取更多数据
        if fallback_days is not None and fallback_days > 30:  # 只有当请求天数较多时才考虑分段拉取
            # 计算预期的数据量
            expected_rows = self._estimate_expected_rows(fallback_days, period)
            actual_rows = len(df)

            # 如果实际数据量远小于预期，尝试分段拉取
            if actual_rows < expected_rows * DATA_AMOUNT_CHECK_RATIO:
                print(f"⚠️ 数据量不足: 实际{actual_rows}条，预期约{expected_rows}条，尝试分段拉取...")
                print(f"   数据时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
                df = self._fetch_with_chunks(symbol, source, query_start_date, query_end_date, period, fallback_days)
                if df is None or df.empty:
                    print("❌ 分段拉取失败")
                    return None, None, None
                else:
                    print(f"✅ 分段拉取成功: 从{actual_rows}条增加到{len(df)}条")
                    print(f"   新数据时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")
            else:
                print(f"✅ 数据量正常: {actual_rows}条 (预期约{expected_rows}条)")

        # 额外检查：如果用户要求较长历史但数据仍然很少，强制分段拉取
        if fallback_days is not None and fallback_days >= 90 and len(df) < MIN_DATA_FOR_CHUNK:
            print(f"强制分段拉取: 请求{fallback_days}天数据但只有{len(df)}条，尝试获取更长历史...")
            df = self._fetch_with_chunks(symbol, source, query_start_date, query_end_date, period, fallback_days)
            if df is None or df.empty:
                print("强制分段拉取失败")
                return None, None, None

        # 保存数据
        filepath, metadata = None, None
        if save:
            period_label = self._format_period_label(period)
            latest_timestamp = df['timestamps'].iloc[-1]
            filepath, metadata = self.save_data(df, symbol, period_label, stock_name, latest_timestamp)

        return df, filepath, metadata

    def _estimate_expected_rows(self, days, period):
        """估算指定天数和周期的预期数据行数"""
        if period == 'D':
            # 日线数据：每个交易日1条
            return max(int(days * TRADING_DAYS_RATIO), int(days * 0.5))

        try:
            minutes_per_period = int(period)
            rows_per_day = TRADING_MINUTES_PER_DAY // minutes_per_period
            expected_rows = days * rows_per_day * TRADING_DAYS_RATIO
            return int(expected_rows)
        except ValueError:
            # 如果无法解析周期，返回保守估计
            return days * 50  # 假设每天约50条数据

    def _fetch_with_chunks(self, symbol, source, start_date, end_date, period, required_days):
        """
        分段拉取数据以获取更多历史数据

        Args:
            symbol (str): 股票代码
            source (str): 数据源
            start_date (str): 开始日期
            end_date (str): 结束日期
            period (str): 周期
            required_days (int): 需要拉取的天数

        Returns:
            pd.DataFrame: 合并后的数据
        """
        # 根据数据源调整分段策略
        source_lower = source.lower()
        chunk_days = CHUNK_DAYS_MAP.get(source_lower, CHUNK_DAYS_MAP['default'])
        max_attempts = MAX_ATTEMPTS_MAP.get(source_lower, MAX_ATTEMPTS_MAP['default'])

        # 计算结束日期（通常是今天）
        end_dt = datetime.now() if end_date is None else pd.to_datetime(end_date)

        print(f"分段拉取模式: 从最近时间开始向前拉取，每段 {chunk_days} 天，最多 {max_attempts} 段")

        frames = []
        current_end = end_dt
        attempts = 0
        consecutive_empty = 0  # 连续空数据段计数

        while attempts < max_attempts and consecutive_empty < MAX_CONSECUTIVE_EMPTY:
            # 从最近时间开始向前拉取
            current_start = current_end - timedelta(days=chunk_days - 1)
            # 确保不早于 start_date（如果指定了的话）
            if start_date and pd.to_datetime(start_date) > current_start:
                current_start = pd.to_datetime(start_date)

            start_str = current_start.strftime('%Y-%m-%d')
            end_str = current_end.strftime('%Y-%m-%d')

            print(f"  正在拉取段 {attempts + 1}: {start_str} 至 {end_str}")

            # 根据数据源调用相应的获取方法（使用字典映射优化）
            source_map = {
                'akshare': self.fetch_from_akshare,
                'yfinance': self.fetch_from_yfinance,
                'baostock': self.fetch_from_baostock
            }
            
            fetch_func = source_map.get(source.lower())
            if fetch_func is None:
                print(f"分段拉取不支持数据源: {source}")
                break
            
            try:
                chunk_df, _ = fetch_func(symbol, start_str, end_str, period)

                if chunk_df is not None and not chunk_df.empty:
                    frames.append(chunk_df)
                    print(f"    获取到 {len(chunk_df)} 条数据")
                    consecutive_empty = 0  # 重置连续空数据计数
                else:
                    print(f"    该段无数据")
                    consecutive_empty += 1

            except Exception as e:
                print(f"    拉取失败: {e}")
                consecutive_empty += 1

            # 向前移动到下一段
            current_end = current_start - timedelta(days=1)
            attempts += 1

            # 如果已经达到最早的可用数据，就停止
            if start_date and current_end < pd.to_datetime(start_date):
                print("已达到指定的最早日期，停止拉取")
                break

            # 添加延迟避免请求过快
            delay = REQUEST_DELAY_MAP.get(source.lower(), REQUEST_DELAY_MAP['default'])
            time.sleep(delay)

        if frames:
            # 合并所有数据段
            combined_df = pd.concat(frames, ignore_index=True)

            # 去重并排序
            combined_df = combined_df.drop_duplicates(subset=['timestamps']).sort_values('timestamps').reset_index(drop=True)

            print(f"分段拉取完成，共获取 {len(combined_df)} 条数据，覆盖时间: {combined_df['timestamps'].min()} 至 {combined_df['timestamps'].max()}")
            return combined_df
        else:
            print("分段拉取失败，没有获取到任何数据")
            return None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Kronos 股票数据获取器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 获取A股数据 (默认)
  python stock_data_fetcher.py --symbol 000001

  # 获取美股数据
  python stock_data_fetcher.py --symbol AAPL --source yfinance --period 5m

  # 使用baostock获取A股数据 (推荐，稳定可靠)
  python stock_data_fetcher.py --symbol 300708 --source baostock --days 180

  # 指定时间范围和数据量
  python stock_data_fetcher.py --symbol 000001 --days 365 --period 5

  # 强制重新获取数据
  python stock_data_fetcher.py --symbol 000001 --force

注意:
  使用 baostock 前需要先运行: python -c "import baostock as bs; bs.login()"
  并注册账号获取token: https://www.baostock.com/
        """
    )

    parser.add_argument(
        "--symbol", "-s",
        required=True,
        help="股票代码 (A股: 000001, 美股: AAPL)"
    )

    parser.add_argument(
        "--source",
        choices=["akshare", "yfinance", "baostock", "tushare", "jqdatasdk"],
        default="baostock",
        help="数据源 (默认: baostock，推荐用于A股)"
    )

    parser.add_argument(
        "--period", "-p",
        default="5",
        help="时间周期 (分钟线: 1,5,15,30,60; 日线: D; yfinance用: 1m,5m等)"
    )

    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="获取最近 N 天的历史数据 (默认: 30)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制重新获取数据，忽略缓存"
    )

    parser.add_argument(
        "--start-date",
        help="指定开始日期 (格式: YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        help="指定结束日期 (格式: YYYY-MM-DD)"
    )

    return parser.parse_args()


def main():
    """主函数 - 支持命令行参数"""
    args = parse_arguments()

    # 创建数据获取器
    fetcher = StockDataFetcher()

    print("=" * 60)
    print(f"🎯 获取股票数据: {args.symbol} ({args.source})")
    print("=" * 60)

    # 计算时间范围
    if args.start_date and args.end_date:
        start_date = args.start_date
        end_date = args.end_date
    else:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')

    print(f"📅 时间范围: {start_date} 至 {end_date}")
    print(f"⏰ 时间周期: {args.period}")
    print(f"🔄 强制刷新: {'是' if args.force else '否'}")
    print("=" * 60)

    # 获取数据
    df, filepath, metadata = fetcher.get_stock_data(
        symbol=args.symbol,
        source=args.source,
        start_date=start_date,
        end_date=end_date,
        period=args.period,
        save=True,
        force_refetch=args.force,
        min_fresh_days=7 if not args.force else None,  # 非强制刷新时检查新鲜度
        fallback_days=args.days
    )

    if df is not None and filepath is not None:
        print("\n" + "=" * 60)
        print("✅ 数据获取成功!")
        print("=" * 60)
        print(f"📊 数据条数: {len(df)}")
        print(f"💾 保存路径: {filepath}")
        print(f"📅 数据时间范围: {df['timestamps'].min()} 至 {df['timestamps'].max()}")

        print(f"\n📋 数据预览:")
        print(df.head())

        if metadata:
            print(f"\n📄 元数据:")
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
    else:
        print("\n❌ 数据获取失败!")
        return 1

    return 0


# 兼容旧版直接运行
def demo():
    """演示函数 - 保持向后兼容"""
    print("运行演示模式...")

    # 创建数据获取器
    fetcher = StockDataFetcher()

    print("=" * 50)
    print("获取A股数据演示")
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


if __name__ == "__main__":
    try:
        # 检查是否有命令行参数
        if len(os.sys.argv) > 1 and os.sys.argv[1] not in ['demo', '--help', '-h']:
            # 有参数，使用命令行模式
            exit(main())
        else:
            # 无参数或明确指定demo，运行演示
            demo()
    except KeyboardInterrupt:
        print("\n用户中断执行")
        exit(1)
    except Exception as e:
        print(f"执行出错: {e}")
        exit(1)
