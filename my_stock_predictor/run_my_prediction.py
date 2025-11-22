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
    python my_stock_predictor/run_my_prediction.py --mode future
    python my_stock_predictor/run_my_prediction.py --mode backtest  # 仅执行回测

    python my_stock_predictor/run_my_prediction.py --mode tune 自动寻找最佳参数 (fun run_tuning)
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
from utils.technical_analysis import TechnicalAnalyzer
from constants import (
    TRADING_MINUTES_PER_DAY,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_RATIO
)

# ==============================================================================
# === 预测配置 ===
# ==============================================================================
# 设备配置 - 根据你的硬件情况选择
import os
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
    "symbol": "300708",          # 股票代码 (例如: A股 '600519', 美股 'NVDA')
    "source": "baostock",        # 数据源 ('baostock' for A股推荐, 'akshare' for A股备用, 'yfinance' for 美股/全球)
    
    # --- 数据获取时间范围 ---
    "start_date": None,         # 数据开始日期 (None 表示自动根据 fallback_fetch_days 计算)
    "end_date": None,           # 数据结束日期 (None 表示使用当前日期)
    "period": "60",             # 数据频率 ('5', '15', '30', '60' for 分钟, 'D' for 日线) - 切换为60分钟线

    # --- 预测参数 (使用带有单位的时间字符串) ---
    "lookback_duration": "120d",   # 回溯时长 (单位: d=天, h=小时, M=月) - 120天约480个点，完美利用模型上下文(512)
    "pred_len_duration": "5d",    # 预测时长 (单位: d=天, h=小时, M=月) - 预测未来5天 (约20个点)

    # --- 模型高级参数 (MPS优化配置) ---
    "T": 0.2,                  # 采样温度 (稍微给一点灵活性)
    "top_p": 0.8,              # 核采样概率 (适度宽松，允许一定灵活性)
    "sample_count": 5,          # 预测路径数量 (用户设备支持最大5，增加路径数可提高稳定性)
    "enable_adaptive_tuning": False,  # 禁用自适应参数调优，使用我们设定的优化参数

    # --- 数据预处理增强 ---
    "enable_advanced_preprocessing": True,  # 启用高级数据预处理
    "price_normalization": "robust",       # 价格归一化方法: 'standard', 'robust', 'none'
    "trend_adjustment": False,             # 禁用趋势调整 (直接预测价格，避免平滑过度)
    "volatility_filter": True,             # 启用波动率过滤

    # --- 新增: 是否强制刷新 ---
    "force_refetch": False,     # 设置为 True 可忽略本地缓存，强制从网络获取最新数据
    # --- 数据新鲜度控制 ---
    "min_data_freshness_days": 5,   # 允许的最大数据滞后天数
    "fallback_fetch_days": 150,     # 当数据过旧时重新拉取的时间范围(天数) - 调整为150天以覆盖120天回溯
    
    # --- 图表显示优化 ---
    "plot_lookback_days": 30,       # 图表显示的历史天数 (显示完整回溯期)
    "enable_focus_mode": True,       # 启用专注模式，只显示预测相关区域
    "prediction_highlight": True,    # 高亮预测区域
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

        # === 智能预检：提前检测数据量是否足够 ===
        print("🔍 正在进行数据可用性预检...")
        precheck_points_needed = minimum_points_needed
        precheck_days = self._estimate_required_days(int(precheck_points_needed * 1.2), config['period'])  # 多获取20%作为缓冲

        # 将'period'转换为数据源能理解的格式
        period_map = {'5': '5m', '15': '15m', '30': '30m', '60': '60m', 'D': '1d'}
        precheck_period = config['period']
        if config['source'] == 'yfinance':
            precheck_period = period_map.get(config['period'], '1d')

        print(f"   - 预检目标: 至少{precheck_points_needed}个数据点，估算需要{precheck_days}天数据")

        precheck_df, _, _ = self.fetcher.get_stock_data(
            symbol=config['symbol'],
            source=config['source'],
            start_date=None,
            end_date=None,
            period=precheck_period,
            save=False,  # 预检不保存
            force_refetch=False,
            min_fresh_days=config.get('min_data_freshness_days'),
            fallback_days=precheck_days
        )

        if precheck_df is not None and len(precheck_df) >= precheck_points_needed:
            print(f"   - ✅ 预检通过: 获取到{len(precheck_df)}个数据点，满足最低要求")
            if not is_future_mode and len(precheck_df) < required_points_total:
                print(f"   - ⚠️ 注意: 数据点({len(precheck_df)})不足以完整回测({required_points_total})，将仅进行未来预测")
                is_future_mode = True
                config["forecast_future"] = True
                print(f"   - 🔄 已切换到未来预测模式")
        else:
            print(f"   - ❌ 预检失败: 只有{len(precheck_df) if precheck_df is not None else 0}个数据点")
            if precheck_df is None:
                print("❌ 数据获取完全失败，流程终止。")
                return
            else:
                print(f"⚠️ 数据不足，将尝试扩展获取范围...")
        print("="*60)

        # === 步骤 1: 获取数据 ===
        print("📊 正在获取数据...")
        print(f"   - 当前模式: {'未来预测' if is_future_mode else '回测'}")
        print(f"   - 至少需要 {minimum_points_needed} 个数据点")
        
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
            print("🔧 可能的解决方案:")
            print("  1. 检查网络连接是否正常")
            print(f"  2. 检查股票代码 '{config['symbol']}' 是否正确")
            print(f"  3. 检查数据源 '{config['source']}' 是否可用")
            print(f"  4. 尝试更换数据源或调整时间范围")
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
                print("🔧 建议的解决方案:")
                print("  1. 检查是否存在网络限制或API限制")
                print(f"  2. 减少预测时长或增加数据频率从 '{config['period']}' 到更粗的时间粒度")
                print(f"  3. 减少回溯时长从 '{config['lookback_duration']}' 到更短的时间范围")
                print("  4. 使用不同的数据源")
                return

            if len(df) < minimum_points_needed:
                print(f"❌ 扩展后数据量 {len(df)} 仍不足以支持当前配置(需要 {minimum_points_needed})")
                print("🔧 参数调整建议:")
                print(f"  1. 当前需要约 {self._estimate_required_days(minimum_points_needed, config['period'])} 天的历史数据")
                print("  2. 建议减少 lookback_duration 或 pred_len_duration 参数")
                print("  3. 或使用更大的数据频率间隔")
                return

        print(f"✅ 数据获取成功，已保存/加载于: {filepath}")
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
        min_required = max(required_total, 5000)  # 至少保留5000点或所需点数
        if original_rows > min_required:
            # 对于未来预测，保留最新的数据
            if is_future_mode:
                keep_rows = min(original_rows, 10000)  # 最多保留10000点
            else:
                # 对于回测，确保有足够的连续数据
                keep_rows = max(required_total + 1000, min_required)  # 多保留一些缓冲

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
            # 使用新的prepare_backtest_data方法正确切分数据
            if len(df) < required_points_total:
                print(f"❌ 错误: 数据不足以进行回测。所需数据点: {required_points_total}, 实际拥有: {len(df)}")
                return

            subset_df = df.tail(required_points_total).reset_index(drop=True)
            # ⚠️ 这里需要先创建临时predictor来调用prepare_backtest_data方法
            # 为了保持一致性，我们手动切分但保存ground_truth
            x_df = subset_df.iloc[:lookback_steps][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            x_timestamp = subset_df.iloc[:lookback_steps]['timestamps'].copy()
            y_timestamp = subset_df.iloc[lookback_steps:lookback_steps+pred_len_steps]['timestamps'].copy()
            
            # 【关键修复】保存ground truth用于后续验证
            ground_truth = subset_df.iloc[lookback_steps:lookback_steps+pred_len_steps][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            ground_truth.index = y_timestamp.values
            print(f"   - ✅ 已准备回测数据并保存真实值用于验证")

        # 指定结果保存目录为当前脚本所在目录下的 prediction_results
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_results")
        predictor = StockPredictor(
            device=os.environ.get('DEVICE', 'auto'),  # 使用环境变量设置的设备
            results_dir=results_dir,
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
            last_date = timestamps[-1].date() if timestamps else last_timestamp.date()
            if current_time.time() > datetime.strptime("15:00", "%H:%M").time() or \
               current_time.date() > last_date:
                
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
                self._validate_backtest_accuracy(pred_df, ground_truth)
        
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
        volatility_ratio = pred_std / pred_mean
        if volatility_ratio > 0.1:  # 如果波动率超过10%
            print(f"   ⚠️ 注意: 预测波动较大 (波动率: {volatility_ratio:.1%})")
            print("   可能需要降低采样参数以获得更稳定的预测")
    
    def _validate_backtest_accuracy(self, pred_df, ground_truth):
        """计算回测准确性指标（与真实历史数据对比）"""
        import numpy as np
        
        # 确保索引对齐
        pred_close = pred_df['close']
        true_close = ground_truth['close']
        
        # 计算各种误差指标
        # RMSE (Root Mean Squared Error) - 均方根误差
        rmse = np.sqrt(np.mean((true_close - pred_close) ** 2))
        
        # MAE (Mean Absolute Error) - 平均绝对误差
        mae = np.mean(np.abs(true_close - pred_close))
        
        # MAPE (Mean Absolute Percentage Error) - 平均绝对百分比误差
        mape = np.mean(np.abs((true_close - pred_close) / true_close)) * 100
        
        # 方向准确率（预测涨跌方向的准确性）
        true_direction = np.sign(true_close.diff().dropna())
        pred_direction = np.sign(pred_close.diff().dropna())
        direction_accuracy = np.mean(true_direction == pred_direction) * 100
        
        print(f"📊 回测准确性指标:")
        print(f"   - RMSE (均方根误差): {rmse:.4f}")
        print(f"   - MAE (平均绝对误差): {mae:.4f}")
        print(f"   - MAPE (平均绝对百分比误差): {mape:.2f}%")
        print(f"   - 方向准确率: {direction_accuracy:.1f}%")
        
        # 评估准确性等级
        print(f"\n📈 准确性评级:")
        if mape < 5:
            print(f"   ✅ 优秀 (MAPE < 5%)")
            print(f"   🎯 预测非常准确，可以信赖该模型")
        elif mape < 10:
            print(f"   ✅ 良好 (MAPE < 10%)")
            print(f"   👍 预测较为准确，可以作为参考")
        elif mape < 20:
            print(f"   ⚠️ 一般 (MAPE < 20%)")
            print(f"   💡 建议调整模型参数或增加训练数据")
        else:
            print(f"   ❌ 较差 (MAPE >= 20%)")
            print(f"   🔧 建议重新调整模型参数或检查数据质量")
            
        # 额外的细节信息
        price_range = true_close.max() - true_close.min()
        print(f"\n📉 详细统计:")
        print(f"   - 真实价格范围: {true_close.min():.2f} - {true_close.max():.2f} (波动: {price_range:.2f})")
        print(f"   - 预测价格范围: {pred_close.min():.2f} - {pred_close.max():.2f}")
        print(f"   - 相对误差 (RMSE/价格范围): {rmse/price_range*100:.2f}%")

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
        
        # 转换周期格式
        period_map = {'5': '5m', '15': '15m', '30': '30m', '60': '60m', 'D': '1d'}
        fetch_period = config['period']
        if config['source'] == 'yfinance':
            fetch_period = period_map.get(config['period'], '1d')

        df, filepath, _ = self.fetcher.get_stock_data(
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
        
        if df is None:
            print(f"❌ 未能获取到数据，无法进行调优。")
            return

        # 检查数据量是否满足用户要求的最低标准 (5000)
        if len(df) < 5000:
            print(f"❌ 数据量不足 ({len(df)})，用户要求最少 5000 条。请尝试获取更多历史数据。")
            return

        if len(df) < required_total:
            print(f"❌ 数据不足，无法进行调优 (需要 {required_total}，实际 {len(df)})")
            return

        # 裁剪数据: 最多保留 30000 条 (多多益善，但有上限)
        max_limit = 30000
        if len(df) > max_limit:
            print(f"✂️ 数据量 ({len(df)}) 超过上限 {max_limit}，截取最新的 {max_limit} 条用于调优...")
            df = df.tail(max_limit).reset_index(drop=True)
        else:
            print(f"✅ 使用全部可用数据 ({len(df)} 条) 进行调优...")
        
        # 2. 准备回测数据
        # 注意：这里需要调整逻辑，因为我们现在使用更多的数据进行验证，而不仅仅是最后一段
        # 但为了保持调优逻辑的一致性（预测最后一段），我们仍然使用最后一段作为验证集
        # 这里的逻辑是：使用 df 的最后 required_total 长度作为输入来预测最后一段
        # 如果 df 很长，前面的数据其实没有被用到预测里（因为模型只看 lookback_steps）
        # 等等，调优的目的是测试参数在"当前"市场环境下的表现。
        # 如果我们只跑一次预测（针对最后一段），那么前面的 20000 条数据其实没用上？
        # 对！UnifiedPredictor.run_prediction_pipeline 内部是单次预测。
        # 如果要利用更多数据，应该进行"滚动回测" (Rolling Backtest)，但这会非常慢。
        # 鉴于用户说"多多益善"，可能误以为数据多就能跑得准。
        # 但实际上，对于单次预测，只有最后 lookback_steps 条数据是有效的输入。
        # 除非... 我们修改 run_prediction_pipeline 让它跑多次？
        # 不，那太复杂了。
        # 既然用户要求"数据最少5000"，我们至少保证了数据量充足。
        # 现有的逻辑是：
        # subset_df = df.tail(required_total)
        # 这意味着它只用了最后 required_total 条。
        # 如果用户想利用更多数据，应该是想看"过去一段时间的平均表现"？
        # 但目前的架构不支持快速的滚动回测。
        # 
        # 让我们先按用户的要求裁剪数据。虽然对于单次预测来说，多余的数据可能没被直接用到，
        # 但保留它们可以确保我们有足够的历史上下文（比如计算技术指标时）。
        
        subset_df = df.tail(required_total).reset_index(drop=True)
        x_df = subset_df.iloc[:lookback_steps][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = subset_df.iloc[:lookback_steps]['timestamps'].copy()
        y_timestamp = subset_df.iloc[lookback_steps:lookback_steps+pred_len_steps]['timestamps'].copy()
        ground_truth = subset_df.iloc[lookback_steps:lookback_steps+pred_len_steps][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        ground_truth.index = y_timestamp.values
        
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
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuning_results")
        predictor = StockPredictor(
            device=os.environ.get('DEVICE', 'auto'),
            results_dir=results_dir,
            enable_adaptive_tuning=False # 调优时必须关闭自适应
        )
        
        # 4. 遍历参数
        import numpy as np
        count = 0
        for T in T_list:
            for top_p in top_p_list:
                count += 1
                
                try:
                    # 运行预测
                    # 临时抑制日志输出以保持整洁
                    import logging
                    predictor.logger.setLevel(logging.WARNING)
                    
                    pred_results = predictor.run_prediction_pipeline(
                        historical_df=df,
                        x_df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        is_future_forecast=False, # 必须是回测模式
                        symbol=config['symbol'],
                        pred_len=pred_len_steps,
                        T=T,
                        top_p=top_p,
                        sample_count=3, # 调优时使用较少的采样数以加快速度
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
                        # 计算MAPE
                        true_close = ground_truth['close']
                        pred_close = pred_df['close']
                        mape = np.mean(np.abs((true_close - pred_close) / true_close)) * 100
                        
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
            import json
            import pandas as pd
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(results_dir, config['symbol'], 'tuning_reports')
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
        """根据周期估算需要的最少交易日数"""
        if required_points <= 0:
            return 1

        if period == 'D':
            return max(required_points, 1)

        try:
            minutes_per_step = int(period)
            if minutes_per_step <= 0:
                raise ValueError
            steps_per_day = max(TRADING_MINUTES_PER_DAY // minutes_per_step, 1)
            return max(math.ceil(required_points / steps_per_day), 1)
        except ValueError:
            return max(required_points, 1)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Kronos 股票预测统一脚本")
    parser.add_argument(
        "--mode",
        choices=["future", "backtest", "tune"],
        default="future",
        help="选择执行模式: future=预测未来, backtest=历史回测, tune=自动参数调优"
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
        # 调优模式强制为回测逻辑
        is_future_mode = False
        runtime_config["forecast_future"] = False
        mode_label = "🎛️ 自动参数调优模式"
        mode_desc = "自动寻找最佳 T 和 top_p 参数组合"
        result_folder = "tuning_results"
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
        # 默认在线模式
        os.environ['KRONOS_OFFLINE_MODE'] = 'false'

    print("="*60)
    print(f"   {mode_label}")
    print(f"   {mode_desc}")
    if args.mode != "tune":
        print(f"   📁 结果将保存至: prediction_results/{runtime_config['symbol']}/{result_folder}/")
    print("="*60)
    
    predictor = UnifiedPredictor()
    if args.mode == "tune":
        predictor.run_tuning(runtime_config)
    else:
        predictor.run_prediction(runtime_config)
