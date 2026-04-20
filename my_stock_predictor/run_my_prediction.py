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
    python my_stock_predictor/run_my_prediction.py --mode tune     # 自动寻找最佳参数 (fun run_tuning)
    python my_stock_predictor/run_my_prediction.py --mode future   # 仅执行未来预测
    python my_stock_predictor/run_my_prediction.py --mode backtest  # 仅执行回测
    python my_stock_predictor/run_my_prediction.py --mode rolling   # 仅执行滚动回测
"""

import argparse
import json
import logging
import math
import os
import sys
import re
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time as dt_time

# 确保脚本可以找到我们创建的模块
# 这将当前文件所在的目录添加到Python的搜索路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_data_fetcher import StockDataFetcher
from stock_predictor import StockPredictor
from utils.technical_analysis import TechnicalAnalyzer
from constants import (
    TRADING_MINUTES_PER_DAY,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_RATIO,
    PERIOD_MAP,
    REQUIRED_COLUMNS,
)

# ==============================================================================
# 模块级常量: A 股交易时间点（避免在循环中反复 strptime）
# ==============================================================================
MARKET_OPEN = dt_time(9, 30)
MARKET_LUNCH_START = dt_time(11, 30)
MARKET_LUNCH_END = dt_time(13, 0)
MARKET_CLOSE = dt_time(15, 0)

# 脚本所在目录, 统一用于结果文件路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# === 预测配置 ===
# ==============================================================================
# 设置设备选择:
# 'auto' = 自动检测 (推荐)
# 'cpu' = 强制使用CPU (稳定但较慢)
# 'cuda' = NVIDIA GPU
# 'mps' = Apple Silicon GPU (如果遇到MPS内存问题，改用'cpu')
# os.environ['DEVICE'] = 'cpu'  # 你有32GB内存，CPU模式应该没问题

# 设备选择配置
# 选项1: 自动检测 (推荐，优先使用MPS，内存不足时自动切换CPU)
os.environ['DEVICE'] = 'auto'

# 选项2: 直接使用CPU (稳定但较慢，适合大内存使用场景)
# os.environ['DEVICE'] = 'cpu'

# 选项3: 强制使用MPS (仅在确认MPS内存充足时使用)
# os.environ['DEVICE'] = 'mps'

"""
超短线 / 日内交易 (当前配置)
适合人群：盯盘时间多，喜欢抓日内波动，做 T+0 或隔日超短线。

特点：反应极快，但噪音多，容易被假动作骗。
推荐参数：
python
"period": "5",              # 5分钟线
"lookback_duration": "20d", # 回溯20天 (约1000个点)
"pred_len_duration": "1d",  # 预测未来1天 (约48个点)
"T": 0.2,                   # 低温，求稳
===================================================
短线波段 (最推荐新手/上班族)
适合人群：每天看一眼，持股 3-5 天，抓周级别的波段。

特点：过滤了日内的细微噪音，信号更稳，胜率通常比5分钟线更高。
推荐参数：
python
"period": "60",             # 60分钟线 (1小时)
"lookback_duration": "60d", # 回溯60天 (约240个点，覆盖一个季度)
"pred_len_duration": "5d",  # 预测未来5天 (约20个点，一周)
"T": 0.7,                   # 稍微给一点灵活性
===================================================
适合人群：不常看盘，持股 1-3 个月，抓大趋势。

特点：非常稳健，忽略短期波动，只看大方向。
推荐参数：
python
"period": "D",              # 日线
"lookback_duration": "250d",# 回溯250天 (约1年)
"pred_len_duration": "20d", # 预测未来20天 (约1个月)
"T": 0.4,                   # 允许模型发挥更多“想象力”来捕捉趋势
"""

PREDICTION_CONFIG = {
    # --- 股票信息 ---
    "symbol": "000876",          # 股票代码 (例如: A股 '600519', 美股 'NVDA')
    "source": "baostock",        # 数据源 ('baostock' for A股推荐, 'akshare' for A股备用, 'yfinance' for 美股/全球)
    
    # --- 数据获取时间范围 ---
    "start_date": None,         # 数据开始日期 (None 表示自动根据 fallback_fetch_days 计算)
    "end_date": None,           # 数据结束日期 (None 表示使用当前日期)
    "period": "D",             # 数据频率 ('5', '15', '30', '60' for 分钟, 'D' for 日线) - 切换为日线以减少日内噪音

    # --- 预测参数 (使用带有单位的时间字符串) ---
    # 回溯不宜过长：Kronos 内部用历史均值做 z-score，若股价涨跌幅度大，
    # 过长回溯导致均值远离当前价位，预测会回归到偏低/偏高的历史均值。
    # 60d 是通用默认值；对近期涨跌幅超 50% 的强趋势股可缩短到 30d。
    "lookback_duration": "60d",    # 60天约60个点（日线模式）
    # 1d 约4-8个点（更短的预测长度更能抵御均值回归的误差，提升真实胜率）
    "pred_len_duration": "3d",    # 预测时长 (单位: d=天, h=小时, M=月) - 预测未来3天

    # --- 数据降维 (实验性) ---
    # 如果开启，模型在预处理时只保留 'close' 价格列，移除高开低和成交量等特征
    # 对于纯时序大模型（如 Kronos），单变量有时候比多变量噪音更少，方向预测更准。
    "use_close_only": False,

    # --- 模型采样参数 ---
    # 以下参数由 tune 模式自动调优得出（--mode tune），基于前复权数据最佳 MAPE=0.11%
    # 更换股票后建议重新运行 tune 找到针对新股票的最佳参数
    "T": 0.7,                      # 采样温度（tune 最佳结果）
    "top_p": 0.95,                 # 核采样概率（tune 最佳结果）
    "sample_count": 10,            # 预测路径数量：10路平均更稳定
    "enable_adaptive_tuning": False,  # 禁用自适应调优，使用上面手动设定的参数

    # --- 数据预处理 ---
    # Kronos 内部已处理尺度和分布，输入应尽量保持原始
    # 过度预处理（IQR替换28%数据+波动率过滤+EWM平滑）是导致 MAPE 40% 的根因
    "enable_advanced_preprocessing": False,  # 关闭高级预处理（归一化/趋势调整/波动率过滤）
    "price_normalization": "none",           # 不做归一化
    "trend_adjustment": False,               # 不做趋势调整
    "volatility_filter": False,              # 关闭波动率过滤，避免破坏真实波动信息

    # --- 新增: 是否强制刷新 ---
    "force_refetch": False,     # 设置为 True 可忽略本地缓存，强制从网络获取最新数据
    # --- 数据新鲜度控制 ---
    "min_data_freshness_days": 5,   # 允许的最大数据滞后天数
    "fallback_fetch_days": 300,     # 增加到300天，确保获取足够的历史数据
    
    # --- 图表显示优化 ---
    "plot_lookback_days": 30,       # 图表显示的历史天数 (显示完整回溯期)
    "enable_focus_mode": True,       # 启用专注模式，只显示预测相关区域
    "prediction_highlight": True,    # 高亮预测区域

    # --- 滚动回测配置（--mode rolling）---
    # 将历史数据按时间顺序切成 N 段，每段独立跑回测，取均值评估真实泛化性能
    "rolling_windows": 5,            # 滚动窗口数量（建议 3-8，越多越可信但越慢）
    "rolling_use_latest_ratio": 0.4, # 保留最近该比例的数据不用于 tune（作为验证集）
}
# ==============================================================================

class UnifiedPredictor:
    def __init__(self):
        self.fetcher = StockDataFetcher()

    # ==========================================================================
    # 公共工具方法（供 run_prediction / run_tuning / run_rolling_backtest 复用）
    # ==========================================================================

    @staticmethod
    def _resolve_fetch_period(period, source):
        """根据数据源将内部 period 编码转换为对应 fetcher 的参数格式"""
        if source == 'yfinance':
            return PERIOD_MAP.get(period, '1d')
        return period

    def _fetch_data(self, config, *, force_refetch=None, fallback_days=None,
                    use_config_dates=True):
        """
        统一的数据获取入口.

        Args:
            config: 预测配置字典
            force_refetch: 覆盖 config['force_refetch'], None 表示使用 config 的值
            fallback_days: 覆盖 config['fallback_fetch_days']
            use_config_dates: False 时忽略 config 里的 start/end date, 按 fallback 拉取

        Returns:
            (df, filepath, metadata) 三元组, 与原 fetcher.get_stock_data 相同
        """
        source = config.get('source', 'baostock')
        fetch_period = self._resolve_fetch_period(config['period'], source)

        effective_force = force_refetch if force_refetch is not None \
            else config.get('force_refetch', False)
        effective_fallback = fallback_days if fallback_days is not None \
            else config.get('fallback_fetch_days')

        return self.fetcher.get_stock_data(
            symbol=config['symbol'],
            source=source,
            start_date=config.get('start_date') if use_config_dates else None,
            end_date=config.get('end_date') if use_config_dates else None,
            period=fetch_period,
            save=True,
            force_refetch=effective_force,
            min_fresh_days=config.get('min_data_freshness_days'),
            fallback_days=effective_fallback
        )

    @staticmethod
    def _split_backtest_window(subset_df, lookback_steps, pred_len_steps, use_close_only):
        """
        将回测窗口切分为 (x_df, x_timestamp, y_timestamp, ground_truth).

        subset_df 必须已经按时间排序, 长度 >= lookback_steps + pred_len_steps.
        ground_truth 的 index 被设置为 y_timestamp.values, 方便与预测结果 align.
        """
        input_cols = ['close'] if use_close_only else list(REQUIRED_COLUMNS)
        x_df = subset_df.iloc[:lookback_steps][input_cols].copy()
        x_timestamp = subset_df.iloc[:lookback_steps]['timestamps'].copy()

        end_idx = lookback_steps + pred_len_steps
        y_timestamp = subset_df.iloc[lookback_steps:end_idx]['timestamps'].copy()
        ground_truth = subset_df.iloc[lookback_steps:end_idx][list(REQUIRED_COLUMNS)].copy()
        ground_truth.index = y_timestamp.values

        return x_df, x_timestamp, y_timestamp, ground_truth

    @staticmethod
    def _estimate_pred_days(pred_len_duration, period, pred_len_steps):
        """估算预测时长对应的"日"数, 用于绘图缩放"""
        if 'd' in pred_len_duration:
            try:
                return max(1, int(pred_len_duration.replace('d', '')))
            except ValueError:
                pass

        period_str = str(period)
        period_int = int(period_str) if period_str.isdigit() else 60
        if period_int > 0:
            points_per_day = TRADING_MINUTES_PER_DAY / period_int
        else:
            points_per_day = 4
        return max(1, int(pred_len_steps / points_per_day))

    def _adjust_plot_lookback_days(self, config, pred_len_steps, context_label=""):
        """
        针对短线预测, 智能压缩图表显示的历史天数, 避免预测区被挤到边缘.
        若满足触发条件则就地更新 config['plot_lookback_days'] 并打印提示.

        Args:
            config: 预测配置字典
            pred_len_steps: 预测步数
            context_label: 可选的上下文标签, 如 "滚动回测"; 为空时附带"超短线视觉效果"提示
        """
        if not config.get('enable_focus_mode', True):
            return

        pred_days = self._estimate_pred_days(
            config['pred_len_duration'], config['period'], pred_len_steps
        )
        original_lookback = config.get('plot_lookback_days', 30)
        if original_lookback <= pred_days * 5:
            return

        smart_lookback = max(3, pred_days * 5)
        config['plot_lookback_days'] = smart_lookback
        if context_label:
            suffix = "。"
            print(f"   - 👁️ 已智能调整{context_label}图表显示历史天数为 "
                  f"{smart_lookback} 天 (原{original_lookback}天){suffix}")
        else:
            print(f"   - 👁️ 已智能调整图表显示历史天数为 {smart_lookback} 天 "
                  f"(原{original_lookback}天)，以优化超短线视觉效果。")

    def _build_predictor(self, results_subdir, enable_adaptive_tuning):
        """统一的 StockPredictor 构造, 避免三处地方重复写 results_dir 拼接"""
        results_dir = os.path.join(SCRIPT_DIR, results_subdir)
        return StockPredictor(
            device=os.environ.get('DEVICE', 'auto'),
            results_dir=results_dir,
            enable_adaptive_tuning=enable_adaptive_tuning
        )

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
                return value * TRADING_DAYS_PER_MONTH
            else: # 'h'
                print(f"⚠️ 警告: 日线数据频率不支持按小时('{duration_str}')计算，将按天处理。")
                return value
        
        else: # 分钟数据
            try:
                minutes_per_step = int(period)
                steps_per_day = TRADING_MINUTES_PER_DAY // minutes_per_step
                
                if unit == 'd':
                    return value * steps_per_day
                elif unit == 'm':
                    return value * TRADING_DAYS_PER_MONTH * steps_per_day
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

        # === 步骤 1: 获取数据（一次获取，避免重复请求） ===
        print("📊 正在获取数据...")
        print(f"   - 当前模式: {'未来预测' if is_future_mode else '回测'}")
        print(f"   - 至少需要 {minimum_points_needed} 个数据点")

        df, filepath, metadata = self._fetch_data(config)

        if df is None:
            print("❌ 获取数据失败，流程终止。")
            print("🔧 可能的解决方案:")
            print("  1. 检查网络连接是否正常")
            print(f"  2. 检查股票代码 '{config['symbol']}' 是否正确")
            print(f"  3. 检查数据源 '{config['source']}' 是否可用")
            return

        # 数据不足时尝试扩展获取范围
        if len(df) < minimum_points_needed:
            print(f"⚠️ 当前数据点 {len(df)} 少于所需的 {minimum_points_needed}，尝试扩展抓取范围...")
            minimum_days = self._estimate_required_days(minimum_points_needed, config['period'])
            fallback_days = max(config.get('fallback_fetch_days') or 0, minimum_days)

            df, filepath, metadata = self._fetch_data(
                config,
                force_refetch=True,
                fallback_days=fallback_days,
                use_config_dates=False
            )

            if df is None or len(df) < minimum_points_needed:
                actual = len(df) if df is not None else 0
                print(f"❌ 数据量不足: 实际{actual}条, 需要{minimum_points_needed}条")
                print("🔧 建议: 减少 lookback_duration/pred_len_duration 或使用更粗的时间粒度")
                return

        print(f"✅ 数据获取成功: {len(df)} 条数据")
        if filepath:
            print(f"   文件: {filepath}")
        print("="*60)

        # === 数据量检查和智能裁剪 ===
        print("="*60)
        print("✂️ 正在检查数据量是否满足预测需求...")
        original_rows = len(df)
        print(f"   - 用于分析的原始数据共有 {original_rows} 条。")

        # 计算所需的最少数据量
        required_total = lookback_steps + pred_len_steps
        print(f"   - 预测配置需要至少 {required_total} 个数据点 (回溯{lookback_steps} + 预测{pred_len_steps})")

        # 对于回测模式，需要更多数据
        if not is_future_mode:
            if original_rows < required_total:
                print(f"   - ⚠️ 回测模式数据不足，将自动切换为未来预测模式")
                print(f"     (需要{required_total}点，实际{original_rows}点)")
                is_future_mode = True
                config["forecast_future"] = True
            else:
                print("   - ✅ 回测模式数据充足")
        else:
            print("   - ✅ 未来预测模式")

        # 智能裁剪数据（保留足够的历史数据）
        # 只需要 required_total 条数据即可完成预测/回测, 无需 5000 这种固定下限
        if original_rows > required_total:
            if is_future_mode:
                # 未来预测: 保留最新的数据, 但设一个上限避免过量
                keep_rows = min(original_rows, 10000)
            else:
                # 回测: 在必需窗口之上再留 1000 条缓冲
                keep_rows = required_total + 1000

            df = df.tail(keep_rows).reset_index(drop=True)
            print(f"   - 已裁剪数据至最新的 {len(df)} 条，保留足够的历史信息。")
        else:
            print(f"   - 数据量适中 ({len(df)} 条)，无需裁剪。")

        # === 新增: 计算并显示当前技术指标 ===
        print("="*60)
        print("📈 计算当前技术指标 (基于历史数据)...")
        try:
            # 计算指标
            tech_df = TechnicalAnalyzer.add_all_indicators(df)
            last_row = tech_df.iloc[-1]
            
            print(f"   - 当前价格: {last_row['close']:.2f}")
            print(f"   - MA5:  {last_row['MA5']:.2f}")
            print(f"   - MA10: {last_row['MA10']:.2f}")
            print(f"   - MA20: {last_row['MA20']:.2f}")
            print(f"   - MACD: {last_row['MACD']:.4f} (Signal: {last_row['MACD_Signal']:.4f}, Hist: {last_row['MACD_Hist']:.4f})")
            print(f"   - RSI:  {last_row['RSI']:.2f}")
            print(f"   - KDJ:  K={last_row['K']:.1f}, D={last_row['D']:.1f}, J={last_row['J']:.1f}")
            print(f"   - BOLL: 上轨={last_row['BB_Upper']:.2f}, 中轨={last_row['BB_Middle']:.2f}, 下轨={last_row['BB_Lower']:.2f}")
            
            # 获取模型预测趋势（如果有的话）
            # 注意：此时模型还没跑，我们只能先基于技术面分析，或者等模型跑完再结合
            # 这里我们先做纯技术面分析，等模型跑完后再做结合分析会更准确，但为了用户体验，先在这里展示技术面信号
            
            analysis_result = TechnicalAnalyzer.analyze_market_condition(tech_df)
            
            print("-" * 40)
            print("🔍 技术面信号:")
            for signal in analysis_result['signals']:
                print(f"   ✅ {signal}")
            
            if analysis_result['warnings']:
                print("⚠️ 风险警示:")
                for warning in analysis_result['warnings']:
                    print(f"   ⚠️ {warning}")
                    
            print("-" * 40)
            
        except Exception as e:
            print(f"   ⚠️ 技术指标计算失败: {e}")
 
        # === 步骤 2: 准备预测 ===
        print("🤖 正在准备预测...")

        ground_truth = None  # 初始化ground_truth变量
        
        use_close_only = config.get('use_close_only', False)

        if is_future_mode:
            # --- 未来预测模式 ---
            print("   - 模式: 未来预测")
            # 【修复】严格按配置裁剪模型输入上下文, 避免传入全部历史数据导致窗口不一致
            subset_df = df.tail(lookback_steps).reset_index(drop=True)
            input_cols = ['close'] if use_close_only else list(REQUIRED_COLUMNS)
            x_df = subset_df[input_cols].copy()
            x_timestamp = subset_df['timestamps'].copy()
            # 生成未来的时间戳
            y_timestamp = self._generate_future_timestamps(
                df['timestamps'].iloc[-1], pred_len_steps, config['period']
            )
            if y_timestamp is None:
                print("❌ 生成未来时间戳失败，流程终止。")
                return
            print(f"   - 已生成 {len(y_timestamp)} 个未来时间点用于预测。")
        else:
            # --- 回测模式 ---
            print("   - 模式: 回测 (与历史数据对比)")
            if len(df) < required_points_total:
                print(f"❌ 错误: 数据不足以进行回测。所需数据点: {required_points_total}, 实际拥有: {len(df)}")
                return

            subset_df = df.tail(required_points_total).reset_index(drop=True)
            x_df, x_timestamp, y_timestamp, ground_truth = self._split_backtest_window(
                subset_df, lookback_steps, pred_len_steps, use_close_only
            )
            print(f"   - ✅ 已准备回测数据并保存真实值用于验证")

        # 智能动态计算绘图的历史天数, 避免短线预测挤在图表边缘
        self._adjust_plot_lookback_days(config, pred_len_steps)

        predictor = self._build_predictor(
            results_subdir="prediction_results",
            enable_adaptive_tuning=config.get('enable_adaptive_tuning', True)
        )

        print(f"✅ StockPredictor初始化完成")
        
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
            plot_lookback=lookback_steps,
            enable_advanced_preprocessing=config.get('enable_advanced_preprocessing', False),
            price_normalization=config.get('price_normalization', 'none'),
            trend_adjustment=config.get('trend_adjustment', False),
            volatility_filter=config.get('volatility_filter', False),
            config=config  # 传递完整配置字典用于图表设置
        )
    
        if results is None:
            print("❌ 预测失败，流程终止。")
            return

        # === 预测结果验证 ===
        self._validate_prediction_results(results, config, ground_truth)

        # === 明确区分预测和回测的结果输出 ===
        print("="*60)
        
        # === 新增: 结合模型预测的最终建议 ===
        if is_future_mode:
            try:
                # 获取模型预测趋势
                pred_start = results['prediction']['close'].iloc[0]
                pred_end = results['prediction']['close'].iloc[-1]
                model_trend = 'up' if pred_end > pred_start else 'down'
                
                # 重新计算包含模型趋势的综合分析
                tech_df = TechnicalAnalyzer.add_all_indicators(df) # 使用原始df重新计算
                final_analysis = TechnicalAnalyzer.analyze_market_condition(tech_df, model_prediction_trend=model_trend)
                
                print("💡 智能交易建议 (模型 + 技术面):")
                print(f"   {final_analysis['advice']}")
                print("="*60)
            except Exception as e:
                print(f"   ⚠️ 生成最终建议失败: {e}")
                
        if is_future_mode:
            print("🎯 未来预测模式完成！")
            print("📁 结果保存在专门的预测文件夹中:")
        else:
            print("📊 历史回测模式完成！")
            print("📁 结果保存在专门的回测文件夹中:")

        print(f"   📈 图表文件: {os.path.basename(results['files']['plot_path'])}")
        print(f"   📄 数据文件: {os.path.basename(results['files']['csv_path'])}")
        print(f"   📂 完整路径: {os.path.dirname(results['files']['plot_path'])}")
        print("="*60)

    def _generate_future_timestamps(self, last_timestamp, steps, period):
        """
        生成未来的 A 股交易时间戳。
        优先使用 chinese_calendar 库跳过法定节假日（春节/五一/国庆等），
        未安装时自动降级为普通工作日逻辑。
        """
        # 尝试导入中国节假日库
        try:
            import chinese_calendar
            def _is_trading_day(dt):
                """判断是否为 A 股交易日（工作日 + 非法定节假日）"""
                return chinese_calendar.is_workday(dt.date())
            _calendar_source = "chinese_calendar（含法定节假日）"
        except ImportError:
            def _is_trading_day(dt):
                return dt.weekday() < 5
            _calendar_source = "普通工作日（未安装 chinese_calendar）"
            print(f"⚠️ 未检测到 chinese_calendar 库，使用{_calendar_source}。\n"
                  "   建议安装: pip install chinese-calendar")

        print(f"   - 交易日历: {_calendar_source}")
        timestamps = []
        current_time = pd.to_datetime(last_timestamp)

        if period == 'D':
            # 日线：逐天检查，跳过非交易日
            candidate = current_time + timedelta(days=1)
            while len(timestamps) < steps:
                if _is_trading_day(candidate):
                    timestamps.append(pd.Timestamp(candidate.date()))
                candidate += timedelta(days=1)
            return pd.to_datetime(timestamps)

        try:
            minutes_per_step = int(period)
        except ValueError:
            print(f"❌ 错误: 无法将周期 '{period}' 转换为分钟数。")
            return None

        while len(timestamps) < steps:
            # 1. 时间递增
            current_time += timedelta(minutes=minutes_per_step)

            # 2. 超过收盘（15:00）或跨天，跳到下一个 A 股交易日开盘
            last_date = timestamps[-1].date() if timestamps else last_timestamp.date()
            if current_time.time() > MARKET_CLOSE or current_time.date() > last_date:
                # 找到下一个交易日（跳过周末和法定节假日）
                next_day = pd.Timestamp(current_time.date()) + timedelta(days=1)
                while not _is_trading_day(next_day):
                    next_day += timedelta(days=1)

                # 重置到开盘时间
                current_time = next_day.to_pydatetime().replace(
                    hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute,
                    second=0, microsecond=0
                )

            # 3. 处理午休 (11:30 -> 13:00)
            if MARKET_LUNCH_START < current_time.time() < MARKET_LUNCH_END:
                current_time = current_time.replace(
                    hour=MARKET_LUNCH_END.hour, minute=MARKET_LUNCH_END.minute,
                    second=0, microsecond=0
                )

            # 4. 检查是否在交易时间内
            time_of_day = current_time.time()
            is_morning = MARKET_OPEN <= time_of_day <= MARKET_LUNCH_START
            is_afternoon = MARKET_LUNCH_END <= time_of_day <= MARKET_CLOSE

            if is_morning or is_afternoon:
                timestamps.append(current_time)

        return pd.to_datetime(timestamps)

    def _validate_prediction_results(self, results, config, ground_truth=None):
        """
        验证预测结果是否在合理范围内，区分预测和回测
        
        Args:
            results: 预测结果字典
            config: 配置字典
            ground_truth: 真实数据(仅回测模式),  DataFrame with index as timestamps
        """
        print("="*60)
        
        pred_df = results['prediction']
        analysis = results['analysis']
        is_future_mode = config.get("forecast_future", False)
        
        if is_future_mode:
            # 未来预测：验证合理性
            print("🔍 正在验证未来预测的合理性...")
            self._validate_reasonability(pred_df, analysis)
        else:
            # 回测：计算与真实值的准确性指标
            print("🔍 正在验证回测准确性...")
            if ground_truth is None:
                print("   ⚠️ 警告: 回测模式但未提供真实数据，只能进行合理性验证")
                self._validate_reasonability(pred_df, analysis)
            else:
                hist_last = analysis.get('historical_last_close') if analysis else None
                self._validate_backtest_accuracy(pred_df, ground_truth, historical_last_close=hist_last)
        
        print("="*60)
    
    def _validate_reasonability(self, pred_df, analysis):
        """验证预测合理性（用于未来预测或缺少ground truth的情况）"""
        # 获取预测数据的统计信息
        pred_close = pred_df['close']
        pred_mean = pred_close.mean()
        pred_std = pred_close.std()
        pred_min = pred_close.min()
        pred_max = pred_close.max()

        # 获取历史数据的最后收盘价作为基准
        historical_last_close = analysis['historical_last_close']

        print(f"   - 历史最后收盘价: {historical_last_close:.2f}")
        print(f"   - 预测均值: {pred_mean:.2f}")
        print(f"   - 预测范围: {pred_min:.2f} - {pred_max:.2f}")

        # 计算预测偏差
        deviation_percentage = abs(pred_mean - historical_last_close) / historical_last_close * 100

        # 设置合理的偏差阈值 (30%以内认为是合理的)
        max_reasonable_deviation = 30.0

        if deviation_percentage > max_reasonable_deviation:
            print(f"   ⚠️ 警告: 预测结果偏差过大 ({deviation_percentage:.1f}%)")
            print("   建议检查数据质量或调整模型参数")
        elif deviation_percentage > 15:
            print(f"   ⚠️ 注意: 预测结果偏差中等 ({deviation_percentage:.1f}%)")
            print("   建议微调参数以获得更准确的预测")
        else:
            print(f"   ✅ 预测结果在合理范围内 (偏差: {deviation_percentage:.1f}%)")

        # 检查预测的波动性是否合理
        volatility_ratio = pred_std / pred_mean if pred_mean != 0 else 0
        if volatility_ratio > 0.1:  # 如果波动率超过10%
            print(f"   ⚠️ 注意: 预测波动较大 (波动率: {volatility_ratio:.1%})")
            print("   可能需要降低采样参数以获得更稳定的预测")
    
    def _validate_backtest_accuracy(self, pred_df, ground_truth, historical_last_close=None):
        """计算回测准确性指标（与真实历史数据对比）"""
        true_close, pred_close = ground_truth['close'].align(pred_df['close'], join='inner')
        if len(true_close) == 0:
            print("⚠️ 预测与真实数据索引无交集，无法计算回测指标")
            return

        rmse = np.sqrt(np.mean((true_close - pred_close) ** 2))
        mae = np.mean(np.abs(true_close - pred_close))
        mape = np.mean(np.abs((true_close - pred_close) / true_close)) * 100

        # 【修复】方向准确率：更贴合交易实战的绝对方向评估（期末点位相对于期初基点的涨跌）
        if historical_last_close is not None:
            # 以历史最后一个真实收盘点位作为入场参考点
            base_price = historical_last_close
            true_dir = 1 if true_close.iloc[-1] > base_price else -1
            pred_dir = 1 if pred_close.iloc[-1] > base_price else -1
            direction_accuracy = 100.0 if true_dir == pred_dir else 0.0
        elif len(true_close) >= 2:
            # 如果未提供 historical_last_close，则用序列内起止点对比
            true_dir = 1 if true_close.iloc[-1] > true_close.iloc[0] else -1
            pred_dir = 1 if pred_close.iloc[-1] > pred_close.iloc[0] else -1
            direction_accuracy = 100.0 if true_dir == pred_dir else 0.0
        else:
            direction_accuracy = 0.0
        
        print(f"📊 回测准确性指标:")
        print(f"   - RMSE (均方根误差): {rmse:.4f}")
        print(f"   - MAE (平均绝对误差): {mae:.4f}")
        print(f"   - MAPE (平均绝对百分比误差): {mape:.2f}%")
        print(f"   - 方向准确率: {direction_accuracy:.1f}%")
        
        # 评估准确性等级（二维评价： MAPE 和 方向准确率）
        print(f"\n📈 综合准确性评级:")
        if mape < 3 and direction_accuracy >= 55:
            print(f"   ✅ 优秀 (MAPE < 3% 且 胜率 >= 55%)")
            print(f"   🎯 方向与点位双优，可以信赖该模型")
        elif mape < 5 and direction_accuracy >= 50:
            print(f"   ✅ 良好 (MAPE < 5% 且 胜率 >= 50%)")
            print(f"   👍 预测较为准确，可作为交易辅助参考")
        elif mape < 10:
            print(f"   ⚠️ 一般 (MAPE < 10% 但方向准确率低于预期)")
            print(f"   💡 虽然点位偏差不大，但涨跌方向参考价值低")
        else:
            print(f"   ❌ 较差 (MAPE >= 10% 或 严重背离)")
            print(f"   🔧 模型在此时断完全失效，不可参考")
            
        # 额外的细节信息
        price_range = true_close.max() - true_close.min()
        print(f"\n📉 详细统计:")
        print(f"   - 真实价格范围: {true_close.min():.2f} - {true_close.max():.2f} (波动: {price_range:.2f})")
        print(f"   - 预测价格范围: {pred_close.min():.2f} - {pred_close.max():.2f}")
        if price_range > 0:
            print(f"   - 相对误差 (RMSE/价格范围): {rmse/price_range*100:.2f}%")
        else:
            print(f"   - 相对误差: 真实数据无波动，无法计算")

    def run_tuning(self, config):
        """
        自动调优参数：遍历T和top_p组合，寻找最佳MAPE
        """
        print("🚀 开始自动参数调优...")
        print("="*60)
        
        # 1. 获取数据 (复用 run_prediction 的逻辑 - 简化版)
        lookback_steps = self._calculate_steps(config['lookback_duration'], config['period'])
        pred_len_steps = self._calculate_steps(config['pred_len_duration'], config['period'])
        required_total = lookback_steps + pred_len_steps
        
        print(f"📊 正在获取数据用于调优 (回溯: {lookback_steps}, 预测: {pred_len_steps})...")

        df, filepath, _ = self._fetch_data(config)

        if df is None:
            print(f"❌ 未能获取到数据，无法进行调优。")
            return

        # 检查数据量是否满足预测所需的最低要求（lookback + pred_len）
        if len(df) < required_total:
            print(f"❌ 数据量不足 ({len(df)})，调优所需最少 {required_total} 条。请尝试增加 fallback_fetch_days 或缩短 lookback_duration。")
            return

        # 裁剪数据: 最多保留 30000 条 (多多益善，但有上限)
        max_limit = 30000
        if len(df) > max_limit:
            print(f"✂️ 数据量 ({len(df)}) 超过上限 {max_limit}，截取最新的 {max_limit} 条用于调优...")
            df = df.tail(max_limit).reset_index(drop=True)
        else:
            print(f"✅ 使用全部可用数据 ({len(df)} 条) 进行调优...")
        
        # 修复数据泄露：tune 应当使用过去的数据找参数，保留最新数据做验证
        use_ratio = config.get('rolling_use_latest_ratio', 0.4)
        if use_ratio >= 1.0 or use_ratio <= 0:
            use_ratio = 0.4
            
        # 最少需要一个完整的回测窗口（lookback + pred）
        # 我们把最新的一部分数据（如40%或至少一个预测窗口长度）切掉不给 tune 用
        reserve_len_for_val = max(int(len(df) * use_ratio), pred_len_steps)
        if len(df) - reserve_len_for_val < required_total:
            print(f"⚠️ 数据总长度({len(df)})扣除验证集({reserve_len_for_val})后不足 required_total({required_total})")
            print("   降级保护：仅切掉最后一个预测窗口作为防泄露隔离")
            reserve_len_for_val = pred_len_steps

        if len(df) - reserve_len_for_val < required_total:
             print("❌ 数据极度匮乏，无法在防泄露前提下进行 tune，请增大 fallback_fetch_days。")
             return

        tune_df = df.iloc[:-reserve_len_for_val].reset_index(drop=True)
        print(f"✂️ 为防数据泄露，保留最新 {reserve_len_for_val} 条数据仅作验证，本次 tune 使用前 {len(tune_df)} 条数据。")

        subset_df = tune_df.tail(required_total).reset_index(drop=True)
        x_df, x_timestamp, y_timestamp, ground_truth = self._split_backtest_window(
            subset_df, lookback_steps, pred_len_steps,
            use_close_only=config.get('use_close_only', False)
        )

        # 3. 定义参数网格
        T_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        top_p_list = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        best_mape = float('inf')
        best_params = None
        results = []

        total_combinations = len(T_list) * len(top_p_list)
        print(f"🔍 将测试 {total_combinations} 组参数组合...")
        print("-" * 60)
        print(f"{'T':<6} | {'top_p':<6} | {'MAPE':<10} | {'Status'}")
        print("-" * 60)

        # 初始化预测器
        tuning_results_dir = os.path.join(SCRIPT_DIR, "tuning_results")
        predictor = self._build_predictor(
            results_subdir="tuning_results",
            enable_adaptive_tuning=False
        )

        # 4. 遍历参数
        count = 0
        for T in T_list:
            for top_p in top_p_list:
                count += 1

                try:
                    # 运行预测: 临时抑制日志输出以保持整洁
                    predictor.logger.setLevel(logging.WARNING)
                    
                    pred_results = predictor.run_prediction_pipeline(
                        historical_df=tune_df,
                        x_df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        is_future_forecast=False,
                        symbol=config['symbol'],
                        pred_len=pred_len_steps,
                        T=T,
                        top_p=top_p,
                        sample_count=3,
                        plot_lookback=lookback_steps,
                        enable_advanced_preprocessing=config.get('enable_advanced_preprocessing', False),
                        price_normalization=config.get('price_normalization', 'none'),
                        trend_adjustment=config.get('trend_adjustment', False),
                        volatility_filter=config.get('volatility_filter', False),
                        config=config
                    )
                    
                    predictor.logger.setLevel(logging.INFO) # 恢复日志
                    
                    if pred_results:
                        pred_df = pred_results['prediction']
                        # 显式对齐索引，避免时间戳错位导致 MAPE 计算出 NaN
                        true_close = ground_truth['close']
                        pred_close = pred_df['close']
                        true_close_aligned, pred_close_aligned = true_close.align(pred_close, join='inner')
                        if len(true_close_aligned) == 0:
                            print(f"{T:<6.1f} | {top_p:<6.1f} | {'NoOverlap':<10} | ❌")
                            continue
                        mape = np.mean(np.abs((true_close_aligned - pred_close_aligned) / true_close_aligned)) * 100
                        
                        results.append({'T': T, 'top_p': top_p, 'mape': mape})
                        print(f"{T:<6.1f} | {top_p:<6.1f} | {mape:<9.2f}% | ✅")
                        
                        if mape < best_mape:
                            best_mape = mape
                            best_params = {'T': T, 'top_p': top_p}
                    else:
                        print(f"{T:<6.1f} | {top_p:<6.1f} | {'Failed':<10} | ❌")
                        
                except Exception as e:
                    print(f"{T:<6.1f} | {top_p:<6.1f} | {'Error':<10} | ❌ ({str(e)})")
        
        print("-" * 60)
        
        # 保存详细报告
        if results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(tuning_results_dir, config['symbol'], 'tuning_reports')
            os.makedirs(report_dir, exist_ok=True)
            
            # 1. 保存为 CSV (方便Excel查看)
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('mape') # 按效果排序
            csv_path = os.path.join(report_dir, f"tuning_results_{timestamp}.csv")
            results_df.to_csv(csv_path, index=False)
            
            # 2. 保存最佳参数为 JSON
            best_result = results_df.iloc[0].to_dict()
            json_path = os.path.join(report_dir, f"best_params_{timestamp}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(best_result, f, indent=4, ensure_ascii=False)
                
            print(f"\n📄 详细调优报告已保存:")
            print(f"   - CSV表格: {csv_path}")
            print(f"   - 最佳参数: {json_path}")

        if best_params:
            print(f"\n🏆 调优完成！最佳参数组合:")
            print(f"   T = {best_params['T']}")
            print(f"   top_p = {best_params['top_p']}")
            print(f"   最佳 MAPE = {best_mape:.2f}%")
            print("\n💡 建议更新 run_my_prediction.py 中的 PREDICTION_CONFIG:")
            print(f"    \"T\": {best_params['T']},")
            print(f"    \"top_p\": {best_params['top_p']},")
        else:
            print("\n❌ 调优失败，未找到有效参数组合。")
            print("="*60)

    def _estimate_required_days(self, required_points, period):
        """根据周期估算需要的最少日历天数（非交易日数）"""
        if required_points <= 0:
            return 1

        if period == 'D':
            # 交易日 -> 日历天：除以交易日占比
            return max(math.ceil(required_points / TRADING_DAYS_RATIO), 1)

        try:
            minutes_per_step = int(str(period).rstrip('mMhHdD'))
            if minutes_per_step <= 0:
                raise ValueError
            steps_per_day = max(TRADING_MINUTES_PER_DAY // minutes_per_step, 1)
            trading_days = math.ceil(required_points / steps_per_day)
            return max(math.ceil(trading_days / TRADING_DAYS_RATIO), 1)
        except ValueError:
            return max(math.ceil(required_points / TRADING_DAYS_RATIO), 1)

    def run_rolling_backtest(self, config):
        """
        滚动回测（Rolling Backtest）
        将历史数据按时间顺序切分为 N 个窗口，每段独立跑回测，
        最终联合所有窗口的 MAPE 和方向准确率平均来评估模型的真实泛化性能。
        相比单窗口回测，结果更可靠。
        """
        print("="*60)
        print("🔄 开始滚动回测（Rolling Backtest）...")
        print("="*60)

        period       = config.get('period', 'D')
        symbol       = config.get('symbol', '000001')
        n_windows    = config.get('rolling_windows', 5)
        T            = config.get('T', 0.9)
        top_p        = config.get('top_p', 0.6)
        sample_count = config.get('sample_count', 5)
        use_close_only = config.get('use_close_only', False)

        lookback_steps = self._calculate_steps(config['lookback_duration'], period)
        pred_len_steps = self._calculate_steps(config['pred_len_duration'], period)
        if not lookback_steps or not pred_len_steps:
            print("❌ 时间参数解析失败")
            return

        # 单个窗口所需点数
        window_size = lookback_steps + pred_len_steps
        # 总共需要的最少数据点数: N 个窗口排列
        required_total = window_size * n_windows

        required_days = self._estimate_required_days(required_total, period)
        df, _, _ = self._fetch_data(
            config,
            fallback_days=max(config.get('fallback_fetch_days', 300) or 300, required_days + 30),
            use_config_dates=False
        )

        if df is None or len(df) < window_size * 2:
            print(f"❌ 数据不足，滚动回测需要至少 {window_size * 2} 个数据点")
            print(f"   建议：增大 fallback_fetch_days 或减少 rolling_windows")
            return

        # 智能动态计算绘图的历史天数, 避免短线预测挤在图表边缘
        self._adjust_plot_lookback_days(config, pred_len_steps, context_label="滚动回测")

        # 根据实际数据量自动调整窗口数
        actual_n = min(n_windows, len(df) // window_size)
        if actual_n < n_windows:
            print(f"⚠️ 数据量不足 {n_windows} 个窗口，实际可用 {actual_n} 个窗口")
        if actual_n < 2:
            print("❌ 数据不足以运行最小滚动回测，请增加数据量")
            return

        print(f"   股票: {symbol} | 周期: {period} | 回溯: {lookback_steps}点 | 预测: {pred_len_steps}点")
        print(f"   实际可用窗口数: {actual_n} （共 {len(df)} 个数据点）")
        print(f"   参数: T={T}, top_p={top_p}, sample_count={sample_count}")
        print("-"*60)

        results = []

        # 循环外创建一次预测器，避免模型重复加载
        predictor = self._build_predictor(
            results_subdir="prediction_results",
            enable_adaptive_tuning=config.get('enable_adaptive_tuning', True)
        )
        predictor.logger.setLevel(logging.WARNING)

        for i in range(actual_n):
            start_idx = len(df) - window_size * (actual_n - i)
            end_idx   = start_idx + window_size
            slice_df  = df.iloc[start_idx:end_idx].reset_index(drop=True)

            win_label = f"窗口{i+1}/{actual_n}"
            ts_start  = slice_df['timestamps'].iloc[0]
            ts_end    = slice_df['timestamps'].iloc[-1]
            print(f"\n[{win_label}] {ts_start} → {ts_end}")

            # slice_df 长度正好是 window_size = lookback_steps + pred_len_steps,
            # 因此 _split_backtest_window 的切分等价于原本的 slice_df.iloc[lookback_steps:]
            x_df, x_timestamp, y_timestamp, ground_truth = self._split_backtest_window(
                slice_df, lookback_steps, pred_len_steps, use_close_only
            )

            # historical_df 需要足够多的点通过 validate_data，取该窗口结尾前的全量数据
            hist_end = end_idx
            hist_start = max(0, hist_end - max(window_size, 120))
            historical_df = df.iloc[hist_start:hist_end].reset_index(drop=True)

            try:
                pred_results = predictor.run_prediction_pipeline(
                    historical_df=historical_df,
                    x_df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    is_future_forecast=False,
                    symbol=symbol,
                    pred_len=pred_len_steps,
                    T=T,
                    top_p=top_p,
                    sample_count=sample_count,
                    plot_lookback=lookback_steps,
                    enable_advanced_preprocessing=config.get('enable_advanced_preprocessing', False),
                    price_normalization=config.get('price_normalization', 'none'),
                    trend_adjustment=config.get('trend_adjustment', False),
                    volatility_filter=config.get('volatility_filter', False),
                    config=config
                )

                if pred_results:
                    pred_df = pred_results['prediction']
                    true_close, pred_close = ground_truth['close'].align(pred_df['close'], join='inner')

                    if len(true_close) == 0:
                        print(f"  [{win_label}] 序列索引无重叠，跳过")
                        continue

                    # MAPE
                    mape = float(np.mean(np.abs((true_close - pred_close) / true_close)) * 100)

                    # 【修复】方向准确率：期末数据点相对于预测前历史最后一个数据点的绝对涨跌方向
                    hist_before_pred = historical_df[historical_df['timestamps'] < pred_df.index.min()]
                    if not hist_before_pred.empty:
                        base_price = hist_before_pred['close'].iloc[-1]
                        true_is_up = true_close.iloc[-1] > base_price
                        pred_is_up = pred_close.iloc[-1] > base_price
                        dir_acc = 100.0 if true_is_up == pred_is_up else 0.0
                    elif len(true_close) >= 2:
                        true_is_up = true_close.iloc[-1] > true_close.iloc[0]
                        pred_is_up = pred_close.iloc[-1] > pred_close.iloc[0]
                        dir_acc = 100.0 if true_is_up == pred_is_up else 0.0
                    else:
                        dir_acc = 0.0

                    results.append({
                        'window': win_label,
                        'start': str(ts_start),
                        'end': str(ts_end),
                        'mape': mape,
                        'dir_acc': dir_acc
                    })
                    print(f"  MAPE: {mape:.2f}% | 方向准确率: {dir_acc:.1f}%")
                else:
                    print(f"  [{win_label}] 预测失败，跳过")

            except Exception as e:
                print(f"  [{win_label}] 错误: {e}")

        # 汇总输出
        print("\n" + "="*60)
        print("📊 滚动回测综合结果")
        print("="*60)

        if not results:
            print("❌ 没有成功运行的窗口，请检查数据量和参数。")
            return

        mapes   = [r['mape'] for r in results]
        dir_accs = [r['dir_acc'] for r in results]
        avg_mape    = float(np.mean(mapes))
        avg_dir_acc = float(np.mean(dir_accs))
        std_mape    = float(np.std(mapes))

        print(f"   窗口数: {len(results)} | 每窗 {lookback_steps}回溯 + {pred_len_steps}预测")
        print(f"\n   平均 MAPE         : {avg_mape:.2f}% (标准差 {std_mape:.2f}%)")
        print(f"   平均方向准确率   : {avg_dir_acc:.1f}%")

        if avg_mape < 5:
            grade = "✅ 优秀"
        elif avg_mape < 10:
            grade = "⚠️ 一般"
        elif avg_mape < 20:
            grade = "🟡 较差"
        else:
            grade = "❌ 差"
        print(f"   整体评级         : {grade}")

        if avg_dir_acc >= 55:
            dir_grade = "✅ 有参考价值"
        elif avg_dir_acc >= 50:
            dir_grade = "⚠️ 接近随机"
        else:
            dir_grade = "❌ 不如抛硬币"
        print(f"   方向信号可信度 : {dir_grade}")

        print("\n   各窗口详细:")
        print(f"   {'#':>3}  {'MAPE':>8}  {'Dir%':>6}  时间结束")
        for r in results:
            print(f"   {r['window']:>3}  {r['mape']:>7.2f}%  {r['dir_acc']:>5.1f}%  {r['end']}")

        # 保存报告
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(
            SCRIPT_DIR, 'prediction_results', symbol, 'rolling_backtest'
        )
        os.makedirs(report_dir, exist_ok=True)

        summary = {
            'symbol': symbol, 'period': period,
            'n_windows': len(results), 'lookback_steps': lookback_steps, 'pred_len_steps': pred_len_steps,
            'parameters': {'T': T, 'top_p': top_p, 'sample_count': sample_count},
            'avg_mape': avg_mape, 'std_mape': std_mape, 'avg_dir_accuracy': avg_dir_acc,
            'window_results': results, 'timestamp': timestamp_str
        }
        json_path = os.path.join(report_dir, f"rolling_report_{timestamp_str}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        results_df = pd.DataFrame(results)
        csv_path = os.path.join(report_dir, f"rolling_results_{timestamp_str}.csv")
        results_df.to_csv(csv_path, index=False)

        print(f"\n📄 滚动回测报告已保存:")
        print(f"   {json_path}")
        print(f"   {csv_path}")
        print("="*60)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Kronos 股票预测统一脚本")
    parser.add_argument(
        "--mode",
        choices=["future", "backtest", "tune", "rolling"],
        default="future",
        help="选择执行模式: future=预测未来, backtest=历史回测, tune=自动参数调优, rolling=滚动回测(推荐)"
    )
    # 网络模式互斥组
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--offline",
        action="store_true",
        help="启用离线模式，只使用本地缓存的模型，不尝试网络更新"
    )
    mode_group.add_argument(
        "--online",
        action="store_true",
        help="启用在线模式，尝试更新模型，失败时使用本地缓存（默认行为）"
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="强制更新模型，忽略更新间隔检查"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    runtime_config = PREDICTION_CONFIG.copy()
    
    # 模式处理
    if args.mode == "tune":
        is_future_mode = False
        runtime_config["forecast_future"] = False
        mode_label = "🎛️ 自动参数调优模式"
        mode_desc = "自动寻找最佳 T 和 top_p 参数组合"
        result_folder = "tuning_results"
    elif args.mode == "rolling":
        is_future_mode = False
        runtime_config["forecast_future"] = False
        mode_label = "🔄 滚动回测模式"
        mode_desc = "多窗口滚动验证，评估模型真实泛化能力"
        result_folder = "rolling_backtest"
    else:
        is_future_mode = args.mode == "future"
        runtime_config["forecast_future"] = is_future_mode
        
        if is_future_mode:
            mode_label = "🎯 未来预测模式"
            mode_desc = "基于历史数据预测未来股价走势"
            result_folder = "future_forecast"
        else:
            mode_label = "📊 历史回测模式"
            mode_desc = "使用历史数据验证预测准确性"
            result_folder = "backtest"

    # 设置模型加载模式
    if args.offline:
        os.environ['KRONOS_OFFLINE_MODE'] = 'true'
        print("🔌 启用离线模式: 只使用本地缓存的模型")
    elif args.online or args.force_update:
        os.environ['KRONOS_OFFLINE_MODE'] = 'false'
        if args.force_update:
            os.environ['KRONOS_FORCE_UPDATE'] = 'true'
            print("🔄 启用强制更新模式: 将强制下载最新模型")
        else:
            print("🌐 启用在线模式: 智能检查更新，失败时使用本地缓存")
    else:
        os.environ['KRONOS_OFFLINE_MODE'] = 'false'

    print("="*60)
    print(f"   {mode_label}")
    print(f"   {mode_desc}")
    if args.mode not in ("tune", "rolling"):
        print(f"   📁 结果将保存至: prediction_results/{runtime_config['symbol']}/{result_folder}/")
    print("="*60)
    
    predictor = UnifiedPredictor()
    if args.mode == "tune":
        predictor.run_tuning(runtime_config)
    elif args.mode == "rolling":
        predictor.run_rolling_backtest(runtime_config)
    else:
        predictor.run_prediction(runtime_config)
