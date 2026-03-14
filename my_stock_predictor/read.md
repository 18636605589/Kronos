my_stock_predictor/
├── run_my_prediction.py      # 主入口脚本（统一执行）
├── stock_predictor.py        # 核心预测器类（2156行，最复杂）
├── stock_data_fetcher.py     # 数据获取模块（支持多数据源）
├── constants.py              # 集中管理所有常量
├── utils/
│   ├── technical_analysis.py # 技术指标计算工具类
│   └── logger_config.py      # 日志配置
├── stock_data/               # 本地缓存的股票数据
│   └── 300627/               # 按股票代码组织
├── prediction_results/       # 预测结果输出
└── tuning_results/           # 参数调优结果
