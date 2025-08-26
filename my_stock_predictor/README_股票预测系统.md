# 🚀 Kronos股票预测系统使用指南

## 📋 系统概述

这是一个基于Kronos模型的完整股票预测系统，包含数据获取、预测分析和结果可视化功能。

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

### 方法1：运行完整演示
```bash
python stock_prediction_demo.py
```

### 方法2：分步使用

#### 第一步：获取股票数据
```python
from stock_data_fetcher import StockDataFetcher

# 创建数据获取器
fetcher = StockDataFetcher(data_dir="stock_data")

# 获取A股数据
df, filepath = fetcher.get_stock_data(
    symbol="000001",  # 平安银行
    source='akshare',
    start_date='2024-01-01',
    end_date='2024-01-31',
    period='5',  # 5分钟数据
    save=True
)
```

#### 第二步：进行预测
```python
from stock_predictor import StockPredictor

# 创建预测器
predictor = StockPredictor(device="cpu")

# 运行预测
results = predictor.run_prediction_pipeline(
    data_file=filepath,
    symbol="000001",
    lookback=400,    # 使用400个历史数据点
    pred_len=120,    # 预测120个时间点
    T=1.0,          # 采样温度
    top_p=0.9,      # 核采样概率
    sample_count=1  # 采样次数
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
- `source`: 数据源 ('akshare' 或 'yfinance')
- `start_date`: 开始日期
- `end_date`: 结束日期
- `period`: 时间周期 ('1', '5', '15', '30', '60', 'D')

### 预测参数
- `lookback`: 回看窗口大小（历史数据点数）
- `pred_len`: 预测长度（未来数据点数）
- `T`: 采样温度（0.1-2.0，越低越保守）
- `top_p`: 核采样概率（0.1-1.0）
- `sample_count`: 采样次数（多次采样取平均）

## 📁 输出文件

### 数据文件
- `stock_data/`: 获取的股票数据
- `analysis_data/`: 分析用的数据

### 预测结果
- `prediction_results/`: 预测结果和图表
- `custom_predictions/`: 自定义参数预测结果

### 文件格式
- `*.csv`: 预测数据
- `*.png`: 预测图表
- `*.json`: 预测元数据和分析报告

## 🎯 使用示例

### 示例1：获取并预测A股
```python
# 获取平安银行数据
fetcher = StockDataFetcher()
df, filepath = fetcher.get_stock_data("000001", source='akshare')

# 进行预测
predictor = StockPredictor()
results = predictor.run_prediction_pipeline(filepath, "000001")
```

### 示例2：预测美股
```python
# 获取苹果公司数据
fetcher = StockDataFetcher()
df, filepath = fetcher.get_stock_data("AAPL", source='yfinance')

# 进行预测
predictor = StockPredictor()
results = predictor.run_prediction_pipeline(filepath, "AAPL")
```

### 示例3：使用现有数据
```python
# 直接使用CSV文件进行预测
predictor = StockPredictor()
results = predictor.run_prediction_pipeline(
    "your_data.csv", 
    "YOUR_SYMBOL"
)
```

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
