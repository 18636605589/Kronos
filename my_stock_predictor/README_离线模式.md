# Kronos 股票预测系统 - 离线模式使用指南

## 📋 功能概述

Kronos 股票预测系统现在支持**离线模式**和**智能更新模式**：

- **离线模式**: 只使用本地缓存的模型，完全不需要网络
- **在线模式**: 尝试更新模型，失败时自动降级到本地缓存
- **智能切换**: 自动检测网络状况和缓存状态

## 🚀 快速开始

### 方法1：使用启动脚本（推荐）

```bash
cd my_stock_predictor
./start_predictor.sh
```

启动脚本会提供交互式菜单选择运行模式。

### 方法2：命令行参数

```bash
# 离线模式（推荐用于无网络环境）
python run_my_prediction.py --offline

# 在线模式（智能更新，默认7天检查一次）
python run_my_prediction.py --online
python run_my_prediction.py  # 默认行为

# 强制更新模式（忽略更新间隔，强制下载最新模型）
python run_my_prediction.py --force-update
```

## 🔧 工作原理

### 离线模式流程
```
1. 检查本地缓存是否存在模型文件
2. 如果存在，直接加载本地模型
3. 如果不存在，报错并提示解决方案
```

### 在线模式流程（智能更新）
```
1. 检查是否在更新间隔内（默认7天）
2. 如果在间隔内，直接使用本地缓存
3. 如果超过间隔，检查本地缓存作为备用
4. 从网络下载/更新最新模型
5. 更新版本信息记录
6. 如果网络失败，回退到本地缓存
```

## 📁 模型缓存和版本管理

### 模型缓存位置
模型文件存储在：
```
~/.cache/huggingface/hub/
├── models--NeoQuasar--Kronos-Tokenizer-base/
└── models--NeoQuasar--Kronos-base/
```

### 版本跟踪文件
系统会在项目目录创建版本跟踪文件：
```
my_stock_predictor/.model_version.json
```
记录最后更新时间和模型信息。

## ⚙️ 高级配置

### 环境变量

```bash
# 设置更新间隔（天数）
export KRONOS_UPDATE_INTERVAL_DAYS=7

# 强制更新（忽略间隔检查）
export KRONOS_FORCE_UPDATE=true

# 启用离线模式
export KRONOS_OFFLINE_MODE=true
```

### 自定义更新间隔

```bash
# 每3天检查一次更新
export KRONOS_UPDATE_INTERVAL_DAYS=3 && python run_my_prediction.py

# 每30天检查一次更新
export KRONOS_UPDATE_INTERVAL_DAYS=30 && python run_my_prediction.py
```

## 🛠️ 故障排除

### 离线模式报错："离线模式加载失败"

**原因**: 本地没有缓存的模型文件

**解决方案**:
1. 首次运行时先使用在线模式下载模型：
   ```bash
   export HF_HUB_DISABLE_SSL_VERIFICATION=1
   export REQUESTS_CA_BUNDLE=""
   export SSL_CERT_FILE=""
   python run_my_prediction.py --online
   ```

2. 检查缓存目录是否存在：
   ```bash
   ls -la ~/.cache/huggingface/hub/
   ```

### 网络连接问题

如果遇到SSL或网络错误，可以设置环境变量：

```bash
export HF_HUB_DISABLE_SSL_VERIFICATION=1
export REQUESTS_CA_BUNDLE=""
export SSL_CERT_FILE=""
export CURL_CA_BUNDLE=""
export HF_HUB_DISABLE_PROGRESS_BARS=1
```

## 📊 性能对比

| 模式 | 首次运行时间 | 后续运行时间 | 网络依赖 | 模型版本 | 更新频率 |
|------|-------------|-------------|----------|----------|----------|
| 离线模式 | 最快 | 最快 | 无 | 本地缓存版本 | 从不更新 |
| 在线模式 | 中等 | 最快 | 有 | 智能更新 | 7天检查 |
| 强制更新 | 最慢 | 中等 | 必须 | 最新版本 | 每次强制 |

## 💡 使用建议

1. **日常使用**: 使用默认在线模式（智能更新，7天检查一次）
2. **定期更新**: 如果需要更频繁更新，设置`KRONOS_UPDATE_INTERVAL_DAYS=1`
3. **无网络环境**: 使用离线模式（确保模型已缓存）
4. **强制更新**: 只有在需要最新模型时使用`--force-update`
5. **稳定版本**: 如果模型更新后出现问题，可以临时使用离线模式

## 🔄 自动更新策略

系统实现了智能的降级策略：
- ✅ **优先网络更新** → 获取最新模型
- 🔄 **网络失败** → 使用本地缓存
- ❌ **缓存不存在** → 报错提示

这样确保系统在各种网络条件下都能正常工作。
