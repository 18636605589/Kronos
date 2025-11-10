# 🚀 Kronos股票预测系统使用指南

## 📋 系统概述

这是一个基于Kronos模型的完整股票预测系统，包含数据获取、预测分析和结果可视化功能。

**🎯 当前默认配置**
- **目标股票**: 创业板 300708 (伟明环保)
- **数据源**: baostock (推荐，稳定可靠)
- **历史时长**: 120天 (约5760个5分钟数据点)
- **预测时长**: 3天 (约144个5分钟数据点)
- **模型参数**: T=0.8, top_p=0.6, sample_count=5 (优化配置)

## 📦 系统组件

### 1. 数据获取模块 (`stock_data_fetcher.py`)
- 支持从多个数据源获取股票数据
- 自动格式化为Kronos模型所需格式
- 支持A股和美股数据获取

### 2. 预测分析模块 (`stock_predictor.py`)
- 基于Kronos模型进行股票预测
- 提供完整的预测流程
- 包含结果分析和可视化

### 3. 完整示例 (`stock_prediction_demo.py`)
- 展示完整的使用流程
- 包含多种预测参数组合
- 提供数据分析演示

## 🛠️ 安装依赖

### 基础依赖
```bash
pip install pandas numpy matplotlib torch safetensors
```

### 可选依赖（数据获取）
```bash
# A股数据获取
pip install akshare

# 美股数据获取
pip install yfinance
```

## 🚀 快速开始

### 方法1：运行统一脚本
```bash
python my_stock_predictor/run_my_prediction.py              # 默认预测未来 (300708股票)
python my_stock_predictor/run_my_prediction.py --mode backtest  # 仅回测历史
```

- 默认以未来预测模式执行，预测**创业板股票 300708**，如需回测请使用 `--mode backtest`。
- 通过修改 `run_my_prediction.py` 中的 `PREDICTION_CONFIG` 控制股票、时间范围、新鲜度策略等参数。
- 当前默认配置：**120天历史数据**，**3天预测时长**，使用 **baostock 数据源**。
- 当 `start_date` / `end_date` 为 `None` 时，会自动根据 `fallback_fetch_days=180` 确定抓取时间范围。
- 当缓存数据超过 `min_data_freshness_days=7` 限制时，脚本会自动重新拉取最近180天的数据。
- 若本地数据不足以覆盖所需回溯/预测窗口，脚本会自动扩展抓取区间后重试，确保回测/预测正常运行。
- 当使用 akshare 数据源且数据量不足时，会自动启用分段拉取模式（每段25天），获取更长的历史数据。
- Apple Silicon (M 系列芯片) 会自动使用 MPS 加速，提升推理速度。
- 模型参数已优化：`T=0.8`, `top_p=0.6`, `sample_count=5` 以获得更稳定的预测结果。

### 方法2：分步使用

#### 第一步：获取股票数据

**方法1：命令行运行**
```bash
# 获取A股数据 (默认180天)
python my_stock_predictor/stock_data_fetcher.py --symbol 300708

# 获取美股数据
python my_stock_predictor/stock_data_fetcher.py --symbol AAPL --source yfinance --period 5m

# 获取1年的历史数据
python my_stock_predictor/stock_data_fetcher.py --symbol 300708 --days 365

# 强制重新获取数据
python my_stock_predictor/stock_data_fetcher.py --symbol 300708 --force

# 指定具体时间范围
python my_stock_predictor/stock_data_fetcher.py --symbol 300708 --start-date 2024-01-01 --end-date 2024-12-31
```

**方法2：Python代码调用**
```python
from stock_data_fetcher import StockDataFetcher

fetcher = StockDataFetcher(data_dir='my_stock_predictor/stock_data')
# 默认使用 baostock 数据源（推荐A股）
df, filepath, metadata = fetcher.get_stock_data(
    symbol='300708',        # 创业板股票 (默认示例)
    source='baostock',      # 默认数据源，也可使用 'akshare' 作为备用
    start_date='2024-01-01',
    end_date='2024-01-31',
    period='5',             # 5分钟数据；若使用 yfinance 请传 '5m'
    save=True,
    force_refetch=True,     # 可按需启用强制刷新
    min_fresh_days=7,       # 可选：要求缓存数据须在7天内
    fallback_days=180       # 可选：过期后拉取近180天的数据
)
```

#### 第二步：准备并运行预测
```python
import pandas as pd
from stock_predictor import StockPredictor

predictor = StockPredictor(device='cpu')

lookback_steps = 1500  # 历史数据点数量 (默认: 1500, 约6.25天)
pred_len_steps = 96    # 预测数据点数量 (默认: 96, 约8小时)

window_df = df.tail(lookback_steps + pred_len_steps).reset_index(drop=True)
x_df = window_df.loc[:lookback_steps - 1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = window_df.loc[:lookback_steps - 1, 'timestamps']
y_timestamp = window_df.loc[lookback_steps:lookback_steps + pred_len_steps - 1, 'timestamps']

results = predictor.run_prediction_pipeline(
    historical_df=df,
    x_df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    is_future_forecast=False,  # True 表示未来预测，False 表示回测
    symbol='300708',
    pred_len=pred_len_steps,
    T=0.8,           # 采样温度 (默认: 0.8，更稳定的预测)
    top_p=0.6,       # 核采样概率 (默认: 0.6，更保守的预测)
    sample_count=5   # 采样次数 (默认: 5，多次采样取平均)
)
```

## 📊 数据格式

系统要求的数据格式（CSV文件）：
```csv
timestamps,open,high,low,close,volume,amount
2024-01-01 09:30:00,10.50,10.55,10.48,10.52,1000,10520
2024-01-01 09:35:00,10.52,10.58,10.50,10.55,1200,12660
...
```

## ⚙️ 参数说明

### 数据获取参数
- `symbol`: 股票代码
- `source`: 数据源 ('akshare', 'yfinance', 'baostock', 'tushare', 'jqdatasdk')

### 支持的数据源对比

| 数据源 | 适用市场 | 免费额度 | 数据质量 | 历史深度 | 安装方式 | 默认推荐 |
|--------|----------|----------|----------|----------|----------|----------|
| **baostock** | A股专业 | 完全免费 | 高 | 完整 | `pip install baostock` + 注册 | ✅ **默认** |
| akshare | A股为主 | 完全免费 | 中等 | 有限(~1个月分钟线) | `pip install akshare` | 备用 |
| yfinance | 美股为主 | 完全免费 | 中等 | 完整 | `pip install yfinance` | 美股专用 |
| tushare | A股专业 | 免费额度有限 | 高 | 完整 | `pip install tushare` + 注册 | 可选 |
| jqdatasdk | 全球专业 | 收费 | 极高 | 完整 | `pip install jqdatasdk` + 付费 | 专业版 |
- `start_date`: 开始日期
- `end_date`: 结束日期
- `period`: 时间周期 ('1', '5', '15', '30', '60', 'D')
- `force_refetch`: 是否忽略缓存重新拉取数据
- `min_data_freshness_days`: 缓存允许的最大滞后天数
- `fallback_fetch_days`: 数据过期时重新拉取的时间范围（天数）

### 预测参数
- `lookback_duration`: 统一脚本使用的回溯时长（如 `120d`, `6h`）默认120天
- `pred_len_duration`: 统一脚本使用的预测时长（如 `3d`, `8h`）默认3天
- `lookback`: 自定义流程中的历史窗口长度（数据点数）默认1500
- `pred_len`: 自定义流程中的预测长度（数据点数）默认96
- `T`: 采样温度（0.1-2.0，越低越保守）默认0.8
- `top_p`: 核采样概率（0.1-1.0）默认0.6
- `sample_count`: 采样次数（多次采样取平均）默认5

## 📁 输出文件

### 数据文件
- `stock_data/<symbol>/`: 获取的原始数据与缓存元数据

### 预测结果
- `prediction_results/<symbol>/`: 预测数据 (`*.csv`)、图表 (`*.png`)、元数据 (`*.json`)

### 文件格式
- `*.csv`: 预测数据
- `*.png`: 预测图表
- `*.json`: 预测元数据和分析报告

## 🎯 使用示例

### 示例：预测美股 (未来模式)
```python
import pandas as pd
from stock_data_fetcher import StockDataFetcher
from stock_predictor import StockPredictor

fetcher = StockDataFetcher()
df, _, _ = fetcher.get_stock_data('AAPL', source='yfinance', period='5m', force_refetch=True)

predictor = StockPredictor()

pred_len_steps = 96  # 约等于 2 个交易日（5m 数据）
future_timestamps = pd.date_range(df['timestamps'].iloc[-1], periods=pred_len_steps + 1, freq='5T')[1:]

results = predictor.run_prediction_pipeline(
    historical_df=df,
    x_df=df[['open', 'high', 'low', 'close', 'volume', 'amount']],
    x_timestamp=df['timestamps'],
    y_timestamp=future_timestamps,
    is_future_forecast=True,
    symbol='AAPL',
    pred_len=pred_len_steps,
    T=0.8,
    top_p=0.6,
    sample_count=5
)
```

### 示例：使用 baostock 获取A股数据
```python
import baostock as bs
# 首次使用需要登录
lg = bs.login()
print(f"登录状态: {lg.error_msg}")

fetcher = StockDataFetcher()
df, _, _ = fetcher.get_stock_data('300708', source='baostock', period='5', force_refetch=True)

predictor = StockPredictor()
# 使用优化后的参数进行预测
results = predictor.run_prediction_pipeline(
    historical_df=df,
    x_df=df[['open', 'high', 'low', 'close', 'volume', 'amount']],
    x_timestamp=df['timestamps'],
    y_timestamp=pd.date_range(df['timestamps'].iloc[-1], periods=97, freq='5T')[1:],
    is_future_forecast=True,
    symbol='300708',
    pred_len=96,
    T=0.8,
    top_p=0.6,
    sample_count=5
)
```

## 🔧 故障排除

### SSL 连接错误 (akshare)
如果遇到 `SSLError` 或网络连接问题：
1. **使用 baostock 数据源**（推荐）：
   ```bash
   python my_stock_predictor/stock_data_fetcher.py --symbol 300708 --source baostock --days 180
   ```
2. **检查网络连接**，尝试使用代理或VPN
3. **等待一段时间**，akshare 的 API 可能暂时不稳定

### 数据获取失败
- **akshare**: 尝试使用 baostock 或 tushare 数据源
- **baostock**: 需要注册账号并获取 token，首次使用需要调用 `bs.login()`
- **网络问题**: 检查防火墙和代理设置

### baostock 时间戳解析错误
如果遇到时间戳解析错误：
1. 确认已正确安装 baostock: `pip install baostock`
2. 确认已登录: `import baostock as bs; bs.login()`
3. 检查股票代码格式，A股需要添加市场前缀:
   - 沪市: `sh.600000`
   - 深市: `sz.000001`
   - 创业板: `sz.300708`

## 📈 预测结果解读

### 价格分析
- **趋势**: 上涨/下跌
- **价格变化**: 绝对值和百分比
- **波动性**: 价格标准差

### 成交量分析
- **平均成交量**: 预测期间的平均值
- **成交量趋势**: 相对于历史的变化

### 可视化图表
- **价格图表**: 历史价格 vs 预测价格
- **成交量图表**: 历史成交量 vs 预测成交量

## ⚠️ 注意事项

1. **数据质量**: 确保输入数据完整且格式正确
2. **模型限制**: Kronos模型的最大上下文长度为512
3. **预测准确性**: 预测结果仅供参考，不构成投资建议
4. **网络连接**: 数据获取需要稳定的网络连接
5. **计算资源**: 预测过程可能需要较长时间

## 🔧 故障排除

### 常见问题

1. **模型加载失败**
   - 检查网络连接
   - 确保已安装所有依赖

2. **数据获取失败**
   - 检查股票代码是否正确
   - 确认数据源是否可用

3. **预测失败**
   - 检查数据格式是否正确
   - 确保数据量足够

4. **内存不足**
   - 减少lookback参数
   - 使用更小的数据范围

## 📞 技术支持

如遇到问题，请检查：
1. 依赖是否正确安装
2. 网络连接是否正常
3. 数据格式是否符合要求
4. 系统资源是否充足

## 📄 许可证

本系统基于Kronos项目，遵循MIT许可证。
