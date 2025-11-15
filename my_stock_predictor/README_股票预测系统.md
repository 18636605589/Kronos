# 🚀 Kronos股票预测系统使用指南

## 📋 系统概述

my_stock_predictor/
├── run_my_prediction.py          # 🚀 主运行脚本（统一入口）
├── stock_data_fetcher.py         # 📊 数据获取模块
├── stock_predictor.py            # 🔮 预测核心模块
├── README_股票预测系统.md         # 📖 使用文档
│
├── stock_data/                   # 💾 本地数据缓存
│   └── 300708/                   # 股票代码目录
│       ├── 300708_5min_*.csv     # 股票数据文件
│       └── metadata.json         # 数据元信息
│
├── prediction_results/
│   └── 300708/                  # 股票代码
│   ├── future_forecast/       # 未来预测结果
│   │   ├── 300708_forecast_chart_20241114_120000.png
│   │   ├── 300708_forecast_data_20241114_120000.csv
│   │   └── 300708_forecast_metadata_20241114_120000.json
│   └── backtest/              # 历史回测结果
│       ├── 300708_backtest_chart_20241114_120000.png
│       ├── 300708_backtest_data_20241114_120000.csv
│       └── 300708_backtest_metadata_20241114_120000.json            # 预测元数据
│
└── my_stock_predictor/          # 📊 子数据目录（可能是历史遗留）
    └── stock_data/

这是一个基于Kronos模型的完整股票预测系统，包含数据获取、预测分析和结果可视化功能。

**🎯 当前默认配置（超高精度优化版）**
- **目标股票**: 创业板 300708 (伟明环保)
- **数据源**: baostock (推荐，稳定可靠)
- **历史时长**: 300天 (约14400个5分钟数据点)
- **预测时长**: 3天 (约720个5分钟数据点)
- **模型参数**: T=0.05, top_p=0.05, sample_count=1 (超高精度配置)
- **数据处理**: 启用指数移动平均平滑处理

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

### ⚠️ macOS 用户注意
macOS 系统默认不允许直接安装 Python 包，需要使用虚拟环境或 `--user` 标志。

### 方法1：使用虚拟环境（强烈推荐）

#### 自动设置（推荐）
```bash
cd /Users/zilong/Documents/项目/Python/Kronos
chmod +x setup_venv.sh
./setup_venv.sh
```

#### 手动设置
```bash
cd /Users/zilong/Documents/项目/Python/Kronos

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install baostock akshare yfinance

# 验证安装
python3 -c "import pandas; import numpy; import torch; print('✅ 核心依赖已安装')"
```

**每次使用前需要激活虚拟环境**:
```bash
source venv/bin/activate
python3 my_stock_predictor/run_my_prediction.py
```

### 方法2：使用 --user 标志（不推荐，但可用）
```bash
cd /Users/zilong/Documents/项目/Python/Kronos
pip3 install --user -r requirements.txt
pip3 install --user baostock akshare yfinance
```

### 方法3：使用 --break-system-packages（不推荐）
```bash
cd /Users/zilong/Documents/项目/Python/Kronos
pip3 install --break-system-packages -r requirements.txt
pip3 install --break-system-packages baostock akshare yfinance
```

### 方法4：手动安装（虚拟环境中）

#### 基础依赖
```bash
pip install pandas numpy matplotlib torch safetensors
pip install einops==0.8.1 huggingface_hub==0.33.1 tqdm==4.67.1
```

#### 可选依赖（数据获取）
```bash
# A股数据获取 (多个选择)
pip install baostock     # 推荐：稳定，专业 A股数据
pip install akshare      # 备用：免费 A股数据
pip install tushare      # 可选：专业 A股数据平台

# 美股数据获取
pip install yfinance     # 美股和全球股票数据
```

### 验证安装
```bash
python3 -c "import pandas; import numpy; import torch; print('✅ 核心依赖已安装')"
```

## 🚀 快速开始

### 方法1：运行统一脚本
```bash
python my_stock_predictor/run_my_prediction.py              # 默认预测未来 (300708股票)
python my_stock_predictor/run_my_prediction.py --mode backtest  # 仅回测历史
```

- 默认以未来预测模式执行，预测**创业板股票 300708**，如需回测请使用 `--mode backtest`。
- 通过修改 `run_my_prediction.py` 中的 `PREDICTION_CONFIG` 控制股票、时间范围、新鲜度策略等参数。
- 当前默认配置：**300天历史数据**，**3天预测时长**，使用 **baostock 数据源**。
- 当 `start_date` / `end_date` 为 `None` 时，会自动根据 `fallback_fetch_days=360` 确定抓取时间范围。
- 当缓存数据超过 `min_data_freshness_days=5` 限制时，脚本会自动重新拉取最近360天的数据。
- 若本地数据不足以覆盖所需回溯/预测窗口，脚本会自动扩展抓取区间后重试，确保回测/预测正常运行。
- 当使用 akshare 数据源且数据量不足时，会自动启用分段拉取模式（每段25天），获取更长的历史数据。
- Apple Silicon (M 系列芯片) 会自动使用 MPS 加速，提升推理速度。
- 模型参数已优化：`T=0.05`, `top_p=0.05`, `sample_count=1` 以获得超高精度的预测结果。
- 数据处理已启用指数移动平均平滑处理，减少噪声提高预测稳定性。

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
- `start_date`: 开始日期
- `end_date`: 结束日期
- `period`: 时间周期 ('1', '5', '15', '30', '60', 'D')
- `force_refetch`: 是否忽略缓存重新拉取数据
- `min_data_freshness_days`: 缓存允许的最大滞后天数
- `fallback_fetch_days`: 数据过期时重新拉取的时间范围（天数）

### 预测参数
- `lookback_duration`: 统一脚本使用的回溯时长（如 `300d`, `6h`, `1M`）默认300天
- `pred_len_duration`: 统一脚本使用的预测时长（如 `3d`, `8h`, `1M`）默认3天
- `lookback`: 自定义流程中的历史窗口长度（数据点数）默认1500
- `pred_len`: 自定义流程中的预测长度（数据点数）默认96
- `T`: 采样温度（0.01-2.0，越低越保守）默认0.05
- `top_p`: 核采样概率（0.01-1.0）默认0.05
- `sample_count`: 采样次数（多次采样取平均）默认1
- `enable_adaptive_tuning`: 是否启用自适应参数调优（True/False）默认False
  - 启用时：系统根据数据波动性和预测长度自动调整T、top_p、sample_count参数
  - 禁用时：严格使用用户指定的参数值

### 时间单位说明
系统支持以下时间单位用于配置预测时长：
- `d`: 天 (例如: `30d` 表示30天)
- `h`: 小时 (例如: `8h` 表示8小时)
- `M`: 月 (例如: `1M` 表示1个月)

**示例**:
- `lookback_duration: "300d"` - 使用300天的历史数据
- `pred_len_duration: "3d"` - 预测未来3天
- `lookback_duration: "24h"` - 使用24小时的历史数据
- `pred_len_duration: "2h"` - 预测未来2小时

### 预测精度优化说明

系统已实施多项优化措施以提高预测精度：

#### 🔧 参数优化
- **超低采样温度** (T=0.05): 大幅降低预测的随机性，使结果更保守稳定
- **极低核采样概率** (top_p=0.05): 限制预测的多样性，专注于最可能的预测路径
- **单次采样** (sample_count=1): 避免多次采样带来的变异性，确保结果的一致性

#### 📊 数据处理优化
- **指数移动平均平滑**: 对价格数据应用轻微平滑处理(alpha=0.1)，减少市场噪声
- **异常值检测**: 自动识别和处理价格异常值
- **历史数据扩展**: 使用300天历史数据提供更丰富的学习样本

#### 🎯 预测策略优化
- **短时预测**: 将预测时长从8天缩短到3天，提高短期预测精度
- **禁用自适应调优**: 防止系统自动调整参数导致的不一致性
- **严格参数控制**: 确保每次预测使用相同的参数配置

#### 📈 质量验证
- **偏差分析**: 自动检测预测结果是否在合理范围内
- **波动性评估**: 比较预测波动性与历史波动性
- **智能建议**: 根据预测质量提供具体的参数调整建议

## 📁 输出文件

### 数据文件
- `my_stock_predictor/stock_data/<symbol>/`: 获取的原始数据与缓存元数据
- 包含：`*.csv` (原始数据), `metadata.json` (数据元信息)

### 预测结果
- `my_stock_predictor/prediction_results/<symbol>/`: 预测数据 (`*.csv`)、图表 (`*.png`)、元数据 (`*.json`)

### 文件格式
- `*.csv`: 预测数据 (包含时间戳、预测价格、成交量等)
- `*.png`: 预测图表 (智能时间轴显示)
- `*.json`: 预测元数据和分析报告 (包含统计信息、模型参数等)

## 🎯 使用示例

### 运行模式说明

系统支持两种预测模式：

#### 🚀 未来预测模式 (默认)
- **用途**: 预测未来的股价走势
- **数据需求**: 只需要足够的历史数据作为输入
- **输出**: 基于历史数据预测未来价格
- **命令**: `python run_my_prediction.py` 或 `python run_my_prediction.py --mode future`

#### 📊 回测模式
- **用途**: 用历史数据验证预测准确性
- **数据需求**: 需要足够的历史数据来切分训练集和测试集
- **输出**: 预测结果与真实历史数据对比
- **命令**: `python run_my_prediction.py --mode backtest`

**注意**: 如果回测模式的数据不足，系统会自动切换为未来预测模式。

### 示例：预测美股 (未来模式)
```python
import pandas as pd
from stock_data_fetcher import StockDataFetcher
from stock_predictor import StockPredictor

fetcher = StockDataFetcher()
df, _, _ = fetcher.get_stock_data('AAPL', source='yfinance', period='5m', force_refetch=True)

predictor = StockPredictor(enable_adaptive_tuning=False)

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
    T=0.1,
    top_p=0.1,
    sample_count=2
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
# 使用高精度参数进行预测
results = predictor.run_prediction_pipeline(
    historical_df=df,
    x_df=df[['open', 'high', 'low', 'close', 'volume', 'amount']],
    x_timestamp=df['timestamps'],
    y_timestamp=pd.date_range(df['timestamps'].iloc[-1], periods=97, freq='5T')[1:],
    is_future_forecast=True,
    symbol='300708',
    pred_len=96,
    T=0.1,
    top_p=0.1,
    sample_count=2
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

### baostock 数据获取问题
如果遇到数据获取问题：
1. 确认已正确安装 baostock: `pip install baostock`
2. 首次使用需要登录: `import baostock as bs; bs.login()`
3. 检查网络连接，baostock 对网络要求较高
4. 检查股票代码格式，A股需要添加市场前缀:
   - 沪市: `sh.600000`
   - 深市: `sz.000001`
   - 创业板: `sz.300708`
5. 如果遇到频率限制，建议适当增加请求间隔

## 📈 预测结果解读

### 价格分析
- **趋势**: 上涨/下跌
- **价格变化**: 绝对值和百分比
- **波动性**: 价格标准差

### 成交量分析
- **平均成交量**: 预测期间的平均值
- **成交量趋势**: 相对于历史的变化

### 可视化图表 - 智能时间轴

#### 🎯 核心特性
- **智能时间轴**: 根据时间跨度和关键时间点自动调整刻度密度
- **关键时间标记**: 自动标识预测开始、当前时间、历史结束等重要节点
- **局部放大视图**: 当时间跨度较大时，在预测区域提供放大镜视图

#### 📊 图表组件
- **价格图表**: 历史价格 vs 预测价格 vs 真实价格（回测模式）
- **成交量图表**: 历史成交量 vs 预测成交量 vs 真实成交量（回测模式）

#### 🎨 视觉元素
- 🟠 **橙色虚线**: 预测开始时间标记
- 🟣 **紫色虚线**: 当前时间标记（未来预测模式）
- ⚪ **灰色虚线**: 历史数据结束标记
- 📦 **橙色边框子图**: 预测区域局部放大（时间跨度>30天时自动显示）

#### 📅 时间轴策略
| 时间跨度 | 基础刻度 | 密集区域刻度 | 标签格式 |
|---------|---------|-------------|---------|
| 1天内 | 每小时 | 每15分钟 | MM-DD HH:MM |
| 1周内 | 每天 | 每小时 | MM-DD HH:00 |
| 1月内 | 每周 | 每天 | MM-DD |
| 更长时间 | 每月 | 每周 | MM-DD |

**智能调节**: 系统优先在预测时间前后24小时内显示最密集的刻度，确保关键信息清晰可见。

#### 💡 使用建议
- **短时间预测**（1-5天）: 时间轴会显示小时级别的细节，方便观察日内走势
- **中期预测**（1周-1月）: 重点关注预测开始前后的日间变化和趋势转折
- **长期预测**（1月以上）: 利用局部放大视图仔细观察预测区域的细节变化
- **回测分析**: 对比预测价格与真实价格在关键时间点的表现差异

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

## 💡 使用提示

### 脚本运行路径
所有命令都需要在项目根目录 (`/Users/zilong/Documents/项目/Python/Kronos`) 下运行：

```bash
cd /Users/zilong/Documents/项目/Python/Kronos
python my_stock_predictor/run_my_prediction.py
```

### 配置修改
要修改预测参数，请编辑 `run_my_prediction.py` 文件顶部的 `PREDICTION_CONFIG` 字典。

### 数据缓存
系统会在 `my_stock_predictor/stock_data/` 目录下缓存数据，避免重复下载。

## 📞 技术支持

如遇到问题，请检查：
1. 依赖是否正确安装
2. 网络连接是否正常
3. 数据格式是否符合要求
4. 系统资源是否充足
5. 脚本运行路径是否正确

## 📄 许可证

本系统基于Kronos项目，遵循MIT许可证。
