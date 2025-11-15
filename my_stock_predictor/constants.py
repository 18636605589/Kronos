#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
常量定义模块
集中管理项目中使用的所有常量，提高代码可维护性
"""

# ==============================================================================
# 交易时间常量
# ==============================================================================
TRADING_HOURS = {
    'MORNING_START': '09:30',
    'MORNING_END': '11:30',
    'AFTERNOON_START': '13:00',
    'AFTERNOON_END': '15:00',
    'LUNCH_START': '11:30',
    'LUNCH_END': '13:00'
}

# A股交易时间点
TRADING_OPEN_HOUR = 9
TRADING_OPEN_MINUTE = 30
TRADING_CLOSE_HOUR = 15
TRADING_CLOSE_MINUTE = 0
LUNCH_START_HOUR = 11
LUNCH_START_MINUTE = 30
LUNCH_END_HOUR = 13
LUNCH_END_MINUTE = 0

# ==============================================================================
# 数据计算常量
# ==============================================================================
TRADING_MINUTES_PER_DAY = 240  # A股每天交易4小时 = 240分钟
TRADING_DAYS_PER_MONTH = 21    # 每月约21个交易日
TRADING_DAYS_RATIO = 0.7       # 交易日占比约70%

# ==============================================================================
# 数据处理常量
# ==============================================================================
DEFAULT_SMOOTH_ALPHA = 0.1     # 指数移动平均平滑系数
OUTLIER_THRESHOLD = 3.0        # 异常值检测阈值(标准差倍数)
MAX_NAN_RATIO = 0.05           # 最大允许NaN比例
MIN_DATA_POINTS = 100          # 最少数据点数
OUTLIER_RATIO_THRESHOLD = 0.05 # 异常值比例阈值(5%)

# ==============================================================================
# 预测参数常量
# ==============================================================================
DEFAULT_T = 0.5                # 默认采样温度
DEFAULT_TOP_P = 0.5            # 默认核采样概率
DEFAULT_SAMPLE_COUNT = 1       # 默认采样次数

# 预测参数范围
T_MIN = 0.1
T_MAX = 2.0
TOP_P_MIN = 0.1
TOP_P_MAX = 0.95
SAMPLE_COUNT_MIN = 1
SAMPLE_COUNT_MAX = 10

# ==============================================================================
# 波动性和质量评估常量
# ==============================================================================
VOLATILITY_HIGH_THRESHOLD = 0.02      # 高波动性阈值
VOLATILITY_LOW_THRESHOLD = 0.005      # 低波动性阈值
MAX_REASONABLE_DEVIATION = 30.0       # 最大合理偏差百分比(%)
MEDIUM_DEVIATION_THRESHOLD = 15.0     # 中等偏差阈值(%)
VOLATILITY_RATIO_THRESHOLD = 0.1      # 波动率阈值(10%)
PRICE_CHANGE_OUTLIER_THRESHOLD = 0.5  # 价格变化异常阈值(50%)
PRICE_CHANGE_WARNING_THRESHOLD = 0.1  # 价格变化警告阈值(10%)

# 波动率合理性范围
VOLATILITY_RATIO_MIN = 0.5
VOLATILITY_RATIO_MAX = 2.0

# 年化交易日数
ANNUAL_TRADING_DAYS = 252

# ==============================================================================
# 数据获取常量
# ==============================================================================
DEFAULT_MIN_FRESH_DAYS = 5     # 默认数据新鲜度要求(天)
DEFAULT_FALLBACK_DAYS = 360   # 默认回退拉取天数
DEFAULT_CHUNK_DAYS_AKSHARE = 25  # akshare分段拉取每段天数
DEFAULT_CHUNK_DAYS_BAOSTOCK = 60  # baostock分段拉取每段天数
DEFAULT_MAX_ATTEMPTS = 12      # 默认最大尝试次数
MAX_CONSECUTIVE_EMPTY = 3      # 最大连续空数据段数

# ==============================================================================
# 数据分段拉取常量
# ==============================================================================
CHUNK_DAYS_MAP = {
    'akshare': 25,
    'baostock': 60,
    'yfinance': 30,
    'default': 30
}

MAX_ATTEMPTS_MAP = {
    'akshare': 8,
    'baostock': 12,
    'yfinance': 12,
    'default': 12
}

# ==============================================================================
# 数据质量阈值
# ==============================================================================
DATA_AMOUNT_CHECK_RATIO = 0.6  # 数据量检查比例(实际/预期)
MIN_DATA_FOR_CHUNK = 500       # 触发分段拉取的最小数据量

# ==============================================================================
# 图表绘制常量
# ==============================================================================
MAX_XTICKS = 15                # 最大X轴刻度数
DEFAULT_PLOT_LOOKBACK = 1500   # 默认绘图回溯点数
MIN_DATA_FOR_PLOT = 5000      # 绘图所需最少数据点
MAX_DATA_FOR_PLOT = 10000      # 绘图最多保留数据点

# 时间轴刻度策略
TIME_AXIS_STRATEGIES = {
    'within_24h': {
        'base_freq': 'H',
        'dense_freq': '15min',
        'label_format': '%m-%d %H:%M'
    },
    'within_week': {
        'base_freq': 'D',
        'dense_freq': 'H',
        'label_format': '%m-%d %H:00'
    },
    'within_month': {
        'base_freq': 'W',
        'dense_freq': 'D',
        'label_format': '%m-%d'
    },
    'longer': {
        'base_freq': 'M',
        'dense_freq': 'W',
        'label_format': '%m-%d'
    }
}

# 密集区域时间范围比例
DENSE_REGION_RATIO = 0.1       # 预测开始前后10%时间范围
CURRENT_TIME_DENSE_RATIO = 0.15  # 当前时间前后15%时间范围
ZOOM_MARGIN_RATIO = 0.2        # 局部放大视图边距比例

# ==============================================================================
# 数据列名常量
# ==============================================================================
REQUIRED_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']
TIMESTAMP_COLUMN = 'timestamps'
PRICE_COLUMNS = ['open', 'high', 'low', 'close']

# ==============================================================================
# 数据源映射
# ==============================================================================
PERIOD_MAP = {
    '5': '5m',
    '15': '15m',
    '30': '30m',
    '60': '60m',
    'D': '1d'
}

# ==============================================================================
# 请求延迟常量(秒)
# ==============================================================================
REQUEST_DELAY_MAP = {
    'akshare': 1.0,
    'baostock': 0.3,
    'yfinance': 0.5,
    'default': 0.5
}

