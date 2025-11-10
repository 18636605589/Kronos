#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================
=== Kronos 股票预测系统 - 统一执行脚本 ===
================================================

这是您的主要预测入口点。
您只需修改下面的 `PREDICTION_CONFIG` 部分，
然后直接运行此脚本即可。

用法:
    python my_stock_predictor/run_my_prediction.py                # 默认预测未来
    python my_stock_predictor/run_my_prediction.py --mode backtest  # 仅执行回测
"""

import argparse
import math
import os
import sys
import re
import pandas as pd
from datetime import datetime, timedelta

# 确保脚本可以找到我们创建的模块
# 这将当前文件所在的目录添加到Python的搜索路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_data_fetcher import StockDataFetcher
from stock_predictor import StockPredictor

# ==============================================================================
# === 预测配置 (您需要修改的部分) ===
# ==============================================================================
PREDICTION_CONFIG = {
    # --- 股票信息 ---
    "symbol": "300708",          # 股票代码 (例如: A股 '600519', 美股 'NVDA')
    "source": "baostock",        # 数据源 ('baostock' for A股推荐, 'akshare' for A股备用, 'yfinance' for 美股/全球)
    
    # --- 数据获取时间范围 ---
    "start_date": None,         # 数据开始日期 (None 表示自动根据 fallback_fetch_days 计算)
    "end_date": None,           # 数据结束日期 (None 表示使用当前日期)
    "period": "5",              # 数据频率 ('5', '15', '30', '60' for 分钟, 'D' for 日线)

    # --- 预测参数 (使用带有单位的时间字符串) ---
    "lookback_duration": "140d",   # 回溯时长 (单位: d=天, h=小时, M=月) - 调整为140天以适应数据量
    "pred_len_duration": "5d",   # 预测时长 (单位: d=天, h=小时, M=月)

    # --- 模型高级参数 (通常无需修改) ---
    "T": 0.8,                   # 采样温度 (越高越多变，越低越保守)
    "top_p": 0.6,               # 核采样概率
    "sample_count": 5,          # 预测路径数量
    # --- 新增: 是否强制刷新 ---
    "force_refetch": False,     # 设置为 True 可忽略本地缓存，强制从网络获取最新数据
    # --- 数据新鲜度控制 ---
    "min_data_freshness_days": 7,   # 允许的最大数据滞后天数
    "fallback_fetch_days": 180,     # 当数据过旧时重新拉取的时间范围(天数)
}
# ==============================================================================

class UnifiedPredictor:
    def __init__(self):
        self.fetcher = StockDataFetcher()

    def _calculate_steps(self, duration_str, period):
        """
        根据时间周期字符串和数据频率计算所需的步数(数据点数量)。
        """
        if not isinstance(duration_str, str):
            print(f"❌ 错误: 时间周期 '{duration_str}' 必须是字符串。")
            return None

        duration_str = duration_str.lower().strip()
        match = re.match(r"(\d+)([dhm])", duration_str)

        if not match:
            print(f"❌ 错误: 无法解析时间周期字符串 '{duration_str}'。请使用如 '30d', '4h', '1M' 的格式。")
            return None

        value, unit = int(match.group(1)), match.group(2)

        # --- 基于数据频率(period)进行计算 ---
        if period == 'D': # 日线数据
            if unit == 'd':
                return value
            elif unit == 'm':
                return value * 21  # 假设每月21个交易日
            else: # 'h'
                print(f"⚠️ 警告: 日线数据频率不支持按小时('{duration_str}')计算，将按天处理。")
                return value
        
        else: # 分钟数据
            try:
                minutes_per_step = int(period)
                # 假设A股每天交易4小时 = 240分钟
                steps_per_day = 240 // minutes_per_step
                
                if unit == 'd':
                    return value * steps_per_day
                elif unit == 'm':
                    return value * 21 * steps_per_day # 按每月21个交易日计算
                elif unit == 'h':
                    return value * (60 // minutes_per_step)

            except (ValueError, ZeroDivisionError):
                print(f"❌ 错误: 无效的分钟线周期 '{period}'。")
                return None

    def run_prediction(self, config):
        """
        根据配置运行完整的获取数据和预测流程。
        """
        print("🚀 开始执行股票预测流程...")
        print("="*60)
        print(f"🎯 目标股票: {config['symbol']} ({config['source']})")
        print("="*60)

        is_future_mode = config.get("forecast_future", False)

        # === 新增: 智能计算回溯和预测步数 ===
        print("🧠 正在智能计算回溯和预测步数...")
        lookback_steps = self._calculate_steps(config['lookback_duration'], config['period'])
        pred_len_steps = self._calculate_steps(config['pred_len_duration'], config['period'])
        
        if lookback_steps is None or pred_len_steps is None:
            print("❌ 无法解析时间周期字符串，流程终止。")
            return
            
        print(f"   - 数据频率: {config['period']}")
        print(f"   - 回溯时长 '{config['lookback_duration']}' -> 计算为 {lookback_steps} 个数据点")
        print(f"   - 预测时长 '{config['pred_len_duration']}' -> 计算为 {pred_len_steps} 个数据点")
        print("="*60)

        required_points_total = lookback_steps + pred_len_steps
        minimum_points_needed = lookback_steps if is_future_mode else required_points_total

        # === 步骤 1: 获取数据 ===
        print("📊 正在获取数据...")
        
        # 将'period'转换为'akshare'和'yfinance'能理解的格式
        period_map = {'5': '5m', '15': '15m', '30': '30m', '60': '60m', 'D': '1d'}
        fetch_period = config['period']
        if config['source'] == 'yfinance':
            fetch_period = period_map.get(config['period'], '1d')

        df, filepath, metadata = self.fetcher.get_stock_data(
            symbol=config['symbol'],
            source=config['source'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            period=fetch_period,
            save=True,
            force_refetch=config.get('force_refetch', False),
            min_fresh_days=config.get('min_data_freshness_days'),
            fallback_days=config.get('fallback_fetch_days')
        )

        if filepath is None or df is None:
            print("❌ 获取数据失败，流程终止。")
            return

        if len(df) < minimum_points_needed:
            print(f"⚠️ 当前数据点 {len(df)} 少于所需的 {minimum_points_needed}，尝试扩展抓取范围...")
            minimum_days = self._estimate_required_days(minimum_points_needed, config['period'])
            fallback_days = config.get('fallback_fetch_days')
            if fallback_days is None:
                fallback_days = minimum_days
            else:
                fallback_days = max(fallback_days, minimum_days)

            df, filepath, metadata = self.fetcher.get_stock_data(
                symbol=config['symbol'],
                source=config['source'],
                start_date=None,
                end_date=None,
                period=fetch_period,
                save=True,
                force_refetch=True,
                min_fresh_days=config.get('min_data_freshness_days'),
                fallback_days=fallback_days
            )

            if filepath is None or df is None:
                print("❌ 扩展抓取仍失败，流程终止。")
                return

            if len(df) < minimum_points_needed:
                print(f"❌ 扩展后数据量 {len(df)} 仍不足以支持当前配置(需要 {minimum_points_needed})，请调整参数。")
                return

        print(f"✅ 数据获取成功，已保存/加载于: {filepath}")
        print("="*60)

        # === 新增：智能数据裁剪 ===
        print("="*60)
        print("✂️ 正在根据数据量智能裁剪...")
        original_rows = len(df)
        print(f"   - 用于分析的原始数据共有 {original_rows} 条。")

        if original_rows > 5000:
            max_rows = 10000
            # 截取最新的数据
            df = df.tail(max_rows).reset_index(drop=True)
            print(f"   - 数据量大于 5000，已截取最新的 {len(df)} 条数据用于后续处理。")
        else:
            print(f"   - 数据量小于或等于 5000，将使用全部数据。")
 
        # === 步骤 2: 准备预测 ===
        print("🤖 正在准备预测...")

        if is_future_mode:
            # --- 未来预测模式 ---
            print("   - 模式: 未来预测")
            # 预测的输入数据是所有我们能获取到的历史数据
            x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = df['timestamps']
            # 生成未来的时间戳
            y_timestamp = self._generate_future_timestamps(df['timestamps'].iloc[-1], pred_len_steps, config['period'])
            if y_timestamp is None:
                print("❌ 生成未来时间戳失败，流程终止。")
                return
            print(f"   - 已生成 {len(y_timestamp)} 个未来时间点用于预测。")
        else:
            # --- 回测模式 ---
            print("   - 模式: 回测 (与历史数据对比)")
            # 从历史数据中切分出输入和用于对比的真实标签
            if len(df) < required_points_total:
                print(f"❌ 错误: 数据不足以进行回测。所需数据点: {required_points_total}, 实际拥有: {len(df)}")
                return

            subset_df = df.tail(required_points_total).reset_index(drop=True)
            x_df = subset_df.loc[:lookback_steps-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
            x_timestamp = subset_df.loc[:lookback_steps-1, 'timestamps']
            y_timestamp = subset_df.loc[lookback_steps:lookback_steps+pred_len_steps-1, 'timestamps']

        predictor = StockPredictor()
        
        results = predictor.run_prediction_pipeline(
            historical_df=df, # 传入完整的历史数据
            x_df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            is_future_forecast=is_future_mode,
            symbol=config['symbol'],
            pred_len=pred_len_steps,
            T=config['T'],
            top_p=config['top_p'],
            sample_count=config['sample_count'],
            plot_lookback=lookback_steps
        )
    
        if results is None:
            print("❌ 预测失败，流程终止。")
            return
    
        print("="*60)
        print("🎉 预测流程全部完成！")
        print(f"📈 预测图表已保存至: {results['files']['plot_path']}")
        print(f"📄 预测数据已保存至: {results['files']['csv_path']}")
        print("="*60)

    def _generate_future_timestamps(self, last_timestamp, steps, period):
        """
        生成未来的交易时间戳 (重写以修复bug)。
        """
        from pandas.tseries.offsets import BDay
        
        timestamps = []
        current_time = pd.to_datetime(last_timestamp)
        
        if period == 'D':
            future_days = pd.date_range(start=current_time + BDay(), periods=steps, freq=BDay())
            return future_days

        try:
            minutes_per_step = int(period)
        except ValueError:
            print(f"❌ 错误: 无法将周期 '{period}' 转换为分钟数。")
            return None

        while len(timestamps) < steps:
            # 1. 时间递增
            current_time += timedelta(minutes=minutes_per_step)

            # 2. 检查是否需要跳到下一天
            # 如果当前时间超过下午3点，或者进入了新的一天
            if current_time.time() > datetime.strptime("15:00", "%H:%M").time() or \
               current_time.date() > (timestamps[-1].date() if timestamps else last_timestamp.date()):
                
                # 计算下一个交易日
                next_day = pd.to_datetime(current_time.date())
                if current_time.weekday() >= 4 or current_time.time() > datetime.strptime("15:00", "%H:%M").time(): # 周五或周末，或当天收盘后
                    next_day = next_day + BDay()
                
                # 重置到下一个交易日的开盘时间
                current_time = next_day.replace(hour=9, minute=30, second=0, microsecond=0)

            # 3. 处理午休 (11:30 -> 13:00)
            if datetime.strptime("11:30", "%H:%M").time() < current_time.time() < datetime.strptime("13:00", "%H:%M").time():
                current_time = current_time.replace(hour=13, minute=0, second=0, microsecond=0)

            # 4. 检查是否在交易时间内
            time_of_day = current_time.time()
            is_morning = datetime.strptime("09:30", "%H:%M").time() <= time_of_day <= datetime.strptime("11:30", "%H:%M").time()
            is_afternoon = datetime.strptime("13:00", "%H:%M").time() <= time_of_day <= datetime.strptime("15:00", "%H:%M").time()

            if is_morning or is_afternoon:
                timestamps.append(current_time)
        
        return pd.to_datetime(timestamps)
    
    def _estimate_required_days(self, required_points, period):
        """根据周期估算需要的最少交易日数"""
        if required_points <= 0:
            return 1

        if period == 'D':
            return max(required_points, 1)

        try:
            minutes_per_step = int(period)
            if minutes_per_step <= 0:
                raise ValueError
            trading_minutes_per_day = 240
            steps_per_day = max(trading_minutes_per_day // minutes_per_step, 1)
            return max(math.ceil(required_points / steps_per_day), 1)
        except ValueError:
            return max(required_points, 1)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Kronos 股票预测统一脚本")
    parser.add_argument(
        "--mode",
        choices=["future", "backtest"],
        default="future",
        help="选择执行模式: future=预测未来, backtest=历史回测"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    runtime_config = PREDICTION_CONFIG.copy()
    is_future_mode = args.mode == "future"

    runtime_config["forecast_future"] = is_future_mode

    if is_future_mode:
        runtime_config["end_date"] = datetime.now().strftime('%Y-%m-%d')
    elif runtime_config.get("end_date") is None:
        runtime_config["end_date"] = datetime.now().strftime('%Y-%m-%d')

    mode_label = "预测未来趋势" if is_future_mode else "回测历史数据"
    print(f"================== 模式: {mode_label} ==================")
    UnifiedPredictor().run_prediction(runtime_config)
