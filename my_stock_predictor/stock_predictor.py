#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票预测分析模块
基于Kronos模型进行股票预测并保存结果
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import torch
import warnings
import logging
from typing import Optional, Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# 延迟导入matplotlib，避免初始化冲突
# import matplotlib.pyplot as plt  # 将在需要时导入

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError as e:
    print(f"错误: 无法导入Kronos模型: {e}")
    print("请确保model目录存在且包含必要的文件")
    raise

# 导入常量
from utils.technical_analysis import TechnicalAnalyzer
from constants import (
    REQUIRED_COLUMNS,
    TIMESTAMP_COLUMN,
    PRICE_COLUMNS,
    DEFAULT_SMOOTH_ALPHA,
    OUTLIER_THRESHOLD,
    MAX_NAN_RATIO,
    MIN_DATA_POINTS,
    DEFAULT_PLOT_LOOKBACK_DAYS,
    FOCUS_MODE_MARGIN_DAYS
)

class StockPredictor:
    """股票预测器"""
    
    def __init__(self, device="auto", max_context=512, results_dir="prediction_results", enable_adaptive_tuning=True):
        """
        初始化预测器

        Args:
            device (str): 计算设备，可为 'cpu'、'cuda:0' 或 'auto'
            max_context (int): 最大上下文长度
            results_dir (str): 结果保存目录
            enable_adaptive_tuning (bool): 是否启用自适应参数调优
        """
        # 初始化日志
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.requested_device = device
        self.device = self._resolve_device(device)
        self.max_context = max_context
        self.results_dir = results_dir
        self.enable_adaptive_tuning = enable_adaptive_tuning

        # 模型相关
        self.model = None
        self.tokenizer = None
        self.predictor = None

        # 性能监控
        self.performance_stats = {
            'predictions_count': 0,
            'total_inference_time': 0,
            'memory_peak': 0,
            'errors_count': 0
        }
        
        # 确保结果目录存在
        self.ensure_results_dir()
        
        # 初始化模型
        try:
            self.load_model()
        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
            raise RuntimeError(f"无法初始化Kronos模型: {str(e)}")

        # 数据预处理配置
        self.data_config = {
            'required_columns': ['open', 'high', 'low', 'close', 'volume', 'amount'],
            'timestamp_column': 'timestamps',
            'max_nan_ratio': 0.05,  # 最大允许的NaN比例
            'min_data_points': 100,  # 最少数据点数
            'outlier_threshold': 3.0  # 异常值检测阈值(标准差倍数)
        }

        # 归一化参数存储（用于逆变换）
        self.normalization_params = {'method': 'none', 'params': {}}
        
        self.logger.info("StockPredictor初始化完成")

    def get_performance_stats(self):
        """获取性能统计信息"""
        return self.performance_stats.copy()

    def reset_performance_stats(self):
        """重置性能统计"""
        self.performance_stats = {
            'predictions_count': 0,
            'total_inference_time': 0,
            'memory_peak': 0,
            'errors_count': 0
        }

    def optimize_memory_usage(self):
        """内存优化"""
        import gc

        try:
            # 强制垃圾回收
            gc.collect()

            # 如果使用GPU，清空缓存
            if self.device.startswith('cuda'):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        self.logger.info("已清空GPU缓存")
                except ImportError:
                    pass
            elif self.device == 'mps':
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                        self.logger.info("已清空MPS缓存")
                except ImportError:
                    pass

            # 额外的内存优化措施
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            self.logger.info(f"当前内存使用: {memory_mb:.1f} MB")

            # 如果内存使用过高，尝试释放更多资源
            if memory_mb > 8000:  # 超过8GB
                self.logger.warning("内存使用过高，尝试深度清理...")

                # 清理可能存在的临时变量
                if hasattr(self, 'temp_data'):
                    delattr(self, 'temp_data')

                # 再次垃圾回收
                gc.collect()

                # 在MPS上尝试更激进的清理
                if self.device == 'mps':
                    try:
                        import torch
                        # 强制同步
                        torch.mps.synchronize()
                        torch.mps.empty_cache()
                        self.logger.info("已执行MPS深度清理")
                    except:
                        pass

        except Exception as e:
            self.logger.warning(f"内存优化过程中出现警告: {str(e)}")
    
    def ensure_results_dir(self):
        """确保结果目录存在"""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"创建结果目录: {self.results_dir}")
    
    def _resolve_device(self, device):
        """根据当前环境解析实际使用的设备"""
        normalized = (device or "auto").lower()

        print(f"🔍 设备检测: 请求设备='{normalized}'")

        if normalized == "auto":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("✅ 检测到 Apple Silicon MPS，自动使用 'mps' 设备。")
                print("⚠️  注意: MPS可能遇到内存限制，如失败会自动切换到CPU")
                print("💡 如需直接使用CPU，请设置: os.environ['DEVICE'] = 'cpu'")
                return "mps"
            elif torch.cuda.is_available():
                print("✅ 检测到可用的 CUDA，自动使用 'cuda:0' 设备。")
                return "cuda:0"
            else:
                print("ℹ️ 未检测到 GPU 加速，将使用 CPU。")
                return "cpu"

        if normalized.startswith("cuda"):
            if torch.cuda.is_available():
                return device
            print("⚠️ 请求使用 CUDA，但当前环境不支持，已自动回退到 CPU。")
            return "cpu"

        if normalized == "mps":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("✅ 使用 MPS (Apple Silicon) 设备。")
                print("💡 如果遇到内存问题，可以切换到CPU模式。")
                return "mps"
            print("⚠️ 请求使用 MPS (Apple Silicon)，但当前环境不支持，已自动回退到 CPU。")
            return "cpu"

        print(f"ℹ️ 使用指定设备: {device}")
        return device
    
    def validate_data(self, df: pd.DataFrame, context: str = "general",
                      min_points_override: Optional[int] = None) -> Tuple[bool, str]:
        """
        验证数据质量和完整性

        Args:
            df: 输入数据框
            context: 验证上下文描述
            min_points_override: 最小数据点数覆盖值，用于对预测输入等场景放宽限制

        Returns:
            (is_valid, error_message)
        """
        try:
            # 1. 检查核心列是否存在
            if 'close' not in df.columns:
                return False, "缺少核心列: close"
                
            # 检查时间戳列
            if self.data_config['timestamp_column'] not in df.columns:
                return False, f"缺少时间戳列: {self.data_config['timestamp_column']}"
                
            # 兼容单变量模式：如果只传入了极少的列，则跳过完整列检查
            if len(df.columns) <= 2 or 'open' not in df.columns:
                self.logger.info("检测到非全特征数据输入(如单变量模式)，跳过严格列校验。")
            else:
                missing_cols = set(self.data_config['required_columns']) - set(df.columns)
                if missing_cols:
                    return False, f"缺少必要列: {missing_cols}"

            # 3. 检查数据量是否足够
            min_required = min_points_override if min_points_override is not None else self.data_config['min_data_points']
            if len(df) < min_required:
                return False, f"数据点严重不足: {len(df)} < {min_required} (至少需要{min_required}个点)"
            
            # 软性警告：如果数据量少于推荐值但多于最小值
            if len(df) < 500:
                 self.logger.warning(f"数据量较少 ({len(df)}), 可能影响模型效果 (推荐 > 500)")

            # 4. 检查NaN值比例
            check_cols = [c for c in self.data_config['required_columns'] if c in df.columns]
            if check_cols:
                nan_ratio = df[check_cols].isnull().sum().sum() / (len(df) * len(check_cols))
                if nan_ratio > self.data_config['max_nan_ratio']:
                    return False, f"NaN值比例过高: {nan_ratio:.2%} > {self.data_config['max_nan_ratio']:.2%}"

            # 5. 检查价格数据合理性
            price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in df.columns]
            for col in price_cols:
                if (df[col] <= 0).any():
                    return False, f"发现非正价格值在列 {col}"

                # 检查high >= max(open, close), low <= min(open, close)
                # 只有在这些列都存在时才检查
                if col == 'high' and 'open' in df.columns and 'close' in df.columns:
                    invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
                    if invalid_high.any():
                        return False, f"发现high价格低于open或close"
                elif col == 'low' and 'open' in df.columns and 'close' in df.columns:
                    invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
                    if invalid_low.any():
                        return False, f"发现low价格高于open或close"

            # 6. 检查时间戳排序
            if not df[self.data_config['timestamp_column']].is_monotonic_increasing:
                return False, "时间戳不是单调递增的"

            return True, "数据验证通过"

        except Exception as e:
            return False, f"数据验证过程中发生错误: {str(e)}"

    def preprocess_data(self, df: pd.DataFrame, detect_outliers: bool = True,
                       enable_advanced: bool = False, normalization: str = "none",
                       trend_adjustment: bool = False, volatility_filter: bool = False,
                       min_points_override: Optional[int] = None) -> pd.DataFrame:
        """
        数据预处理和清理（增强版）

        Args:
            df: 原始数据框
            detect_outliers: 是否检测和处理异常值
            enable_advanced: 是否启用高级预处理
            normalization: 归一化方法 ('standard', 'robust', 'none')
            trend_adjustment: 是否启用趋势调整
            volatility_filter: 是否启用波动率过滤
            min_points_override: 最小数据点数覆盖值

        Returns:
            处理后的数据框
        """
        try:
            logger.info("开始数据预处理...")

            is_valid, error_msg = self.validate_data(df, "preprocessing_input",
                                                      min_points_override=min_points_override)
            if not is_valid:
                raise ValueError(f"输入数据验证失败: {error_msg}")

            processed_df = df.copy()

            # 1. 确保数值列类型正确（先于 NaN 填充，避免 coerce 产生的 NaN 被遗漏）
            numeric_cols = [c for c in REQUIRED_COLUMNS if c in processed_df.columns]
            for col in numeric_cols:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

            # 2. 处理缺失值
            for col in numeric_cols:
                if processed_df[col].isnull().any():
                    processed_df[col] = processed_df[col].ffill().bfill()
                    if processed_df[col].isnull().any():
                        median_val = processed_df[col].median()
                        processed_df[col] = processed_df[col].fillna(median_val)
                        logger.warning(f"列 {col} 使用中位数 {median_val:.2f} 填充剩余NaN值")

            # 2. 异常值检测和处理（仅在启用高级预处理时执行）
            # Kronos 基础模型对原始数据表现更好，IQR 检测容易误删正常股价
            if enable_advanced and detect_outliers:
                processed_df = self._handle_outliers(processed_df)

            # 3. 高级预处理（可选）
            if enable_advanced:
                processed_df = self._advanced_preprocessing(
                    processed_df, normalization, trend_adjustment, volatility_filter
                )

            # 4. 数据平滑处理（仅在启用高级预处理时执行）
            # 平滑会引入滞后，对基础模型预测产生系统性偏差
            if enable_advanced:
                processed_df = self._smooth_price_data(processed_df)

            # 5. 确保时间戳类型正确
            processed_df[TIMESTAMP_COLUMN] = pd.to_datetime(processed_df[TIMESTAMP_COLUMN])

            # 6. 排序数据
            processed_df = processed_df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

            # 7. 最终验证处理后的数据
            is_valid, error_msg = self.validate_data(processed_df, "preprocessing_output",
                                                      min_points_override=min_points_override)
            if not is_valid:
                logger.warning(f"预处理后数据存在问题: {error_msg}，但继续执行")

            logger.info(f"数据预处理完成，处理后数据量: {len(processed_df)}")
            return processed_df

        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise

    def _advanced_preprocessing(self, df: pd.DataFrame, normalization: str = "none",
                               trend_adjustment: bool = False, volatility_filter: bool = False) -> pd.DataFrame:
        """
        高级数据预处理方法

        Args:
            df: 输入数据框
            normalization: 归一化方法
            trend_adjustment: 是否趋势调整
            volatility_filter: 是否波动率过滤

        Returns:
            处理后的数据框
        """
        processed_df = df.copy()

        # 1. 价格归一化
        if normalization != "none":
            processed_df = self._normalize_prices(processed_df, method=normalization)

        # 2. 趋势调整
        if trend_adjustment:
            processed_df = self._adjust_trend(processed_df)

        # 3. 波动率过滤
        if volatility_filter:
            processed_df = self._filter_volatility(processed_df)

        return processed_df

    def _normalize_prices(self, df: pd.DataFrame, method: str = "robust") -> pd.DataFrame:
        """价格归一化 - 保存参数用于后续逆变换"""
        normalized_df = df.copy()
        price_cols = PRICE_COLUMNS
        
        # 初始化归一化参数存储
        self.normalization_params = {'method': method, 'params': {}}

        for col in price_cols:
            if method == "standard":
                # Z-score标准化
                mean_val = df[col].mean()
                std_val = df[col].std()
                self.normalization_params['params'][col] = {
                    'mean': float(mean_val), 
                    'std': float(std_val)
                }
                if std_val > 0:
                    normalized_df[col] = (df[col] - mean_val) / std_val
            elif method == "robust":
                # 稳健标准化（使用中位数和IQR）
                median_val = df[col].median()
                q75, q25 = df[col].quantile([0.75, 0.25])
                iqr = q75 - q25
                self.normalization_params['params'][col] = {
                    'median': float(median_val),
                    'iqr': float(iqr)
                }
                if iqr > 0:
                    normalized_df[col] = (df[col] - median_val) / iqr

        logger.info(f"已应用{method}价格归一化并保存参数")
        logger.debug(f"归一化参数: {self.normalization_params}")
        return normalized_df

    def _adjust_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """趋势调整 - 去除长期趋势，突出周期性变化"""
        adjusted_df = df.copy()

        # 计算移动平均趋势
        for col in PRICE_COLUMNS:
            trend = df[col].rolling(window=50, center=False).mean()
            # 去除趋势成分
            adjusted_df[col] = df[col] - trend + trend.mean()

        logger.info("已应用趋势调整")
        # 使用新版 pandas API，避免废弃告警
        return adjusted_df.bfill().ffill()

    def _filter_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """波动率过滤 - 高波动区间用移动平均平滑，降低噪声对模型的干扰"""
        filtered_df = df.copy()

        # 计算滚动波动率（20周期）
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20).std()

        # 高波动期权重降低（权重越低，越依赖移动平均）
        volatility_threshold = max(volatility.quantile(0.8), 1e-10)
        weights = 1 / (1 + volatility / volatility_threshold)
        # 避免 NaN 权重（序列开头）
        weights = weights.fillna(1.0)

        # 高波动区间用移动平均替代原始数据（修复原恒等变换 bug）
        for col in PRICE_COLUMNS:
            rolling_mean = df[col].rolling(window=20, center=True, min_periods=1).mean()
            filtered_df[col] = df[col] * weights + rolling_mean * (1 - weights)

        logger.info("已应用波动率过滤")
        return filtered_df.bfill().ffill()
    
    def _inverse_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        逆归一化 - 将归一化后的数据还原到原始价格尺度
        
        Args:
            df: 归一化后的数据框
            
        Returns:
            还原到原始尺度的数据框
        """
        # 如果没有进行归一化，直接返回
        if not self.normalization_params or self.normalization_params.get('method') == 'none':
            logger.debug("未进行归一化，跳过逆变换")
            return df
        
        denormalized_df = df.copy()
        method = self.normalization_params['method']
        params = self.normalization_params.get('params', {})
        
        if not params:
            logger.warning("归一化参数为空，无法进行逆变换")
            return df
        
        # 对每个价格列进行逆变换
        for col in PRICE_COLUMNS:
            if col in df.columns and col in params:
                col_params = params[col]
                if method == "standard":
                    # Z-score逆变换: x = z * std + mean
                    denormalized_df[col] = df[col] * col_params['std'] + col_params['mean']
                    logger.debug(f"列 {col} 逆标准化: mean={col_params['mean']:.4f}, std={col_params['std']:.4f}")
                elif method == "robust":
                    # Robust逆变换: x = z * IQR + median
                    denormalized_df[col] = df[col] * col_params['iqr'] + col_params['median']
                    logger.debug(f"列 {col} 逆稳健标准化: median={col_params['median']:.4f}, iqr={col_params['iqr']:.4f}")
        
        logger.info(f"已应用{method}逆归一化，数据还原到原始价格尺度")
        return denormalized_df

    def _smooth_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对价格数据进行轻微平滑，减少噪声，提高预测稳定性。
        使用 adjust=False（纯因果递推），避免使用未来数据（前瞻偏差）。
        alpha=0.3 在平滑和响应速度之间取得更好平衡（原 0.1 滞后过强）。
        """
        smoothed_df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            if col in smoothed_df.columns:
                # adjust=False: 纯因果递推，不使用未来数据
                # alpha=0.3: 减小历史滞后，使最近价格权重更高
                smoothed_df[col] = smoothed_df[col].ewm(alpha=0.3, adjust=False).mean()

        logger.info("已应用数据平滑处理（因果EWM，alpha=0.3）")
        return smoothed_df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理异常值 - 使用IQR方法进行更robust的检测
        """
        processed_df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            # 使用IQR (四分位距) 方法检测异常值，更适合股票数据
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1

            # IQR异常值检测：A股涨跌停属于正常行为，阈值从1.5放宽到2.5
            lower_bound = Q1 - 2.5 * IQR
            upper_bound = Q3 + 2.5 * IQR

            outlier_mask = (processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)

            if outlier_mask.any():
                outlier_count = outlier_mask.sum()
                logger.warning(f"列 {col} 使用IQR方法发现 {outlier_count} 个异常值")

                # 对于价格数据，使用局部回归或插值而不是简单的移动平均
                if outlier_count / len(processed_df) < 0.05:  # 异常值比例小于5%
                    # 使用线性插值替换异常值
                    processed_df[col] = processed_df[col].where(~outlier_mask, np.nan)
                    processed_df[col] = processed_df[col].interpolate(method='linear', limit_direction='both')
                else:
                    # 如果异常值太多，使用移动中位数（更robust）
                    processed_df[col] = processed_df[col].where(~outlier_mask, processed_df[col].rolling(window=7, center=True, min_periods=3).median())

                # 确保插值后没有NaN值
                if processed_df[col].isnull().any():
                    processed_df[col] = processed_df[col].ffill().bfill()

        return processed_df
    
    def load_model(self):
        """加载Kronos模型，支持离线模式和自动更新"""
        try:
            print(f"正在加载Kronos模型... (device: {self.device})")

            # 离线模式加载逻辑
            offline_mode = os.environ.get('KRONOS_OFFLINE_MODE', 'false').lower() == 'true'

            if offline_mode:
                print("🔌 启用离线模式，只使用本地缓存的模型")
                self._load_model_offline()
            else:
                print("🌐 启用在线模式，优先使用最新模型")
                self._load_model_with_update()

            # 创建预测器
            self.predictor = KronosPredictor(
                self.model,
                self.tokenizer,
                device=self.device,
                max_context=self.max_context
            )

            print("✅ Kronos大模型加载成功！")
            print(f"   📊 模型信息: {self.model.__class__.__name__}")
            print(f"   🖥️  运行设备: {self.device}")
            print(f"   🧠 最大上下文长度: {self.max_context}")

        except ImportError as e:
            print(f"❌ 模型导入失败: {e}")
            print("\n🔧 解决方案:")
            print("  1. 确保已安装Kronos相关依赖: pip install -r requirements.txt")
            print("  2. 检查model目录是否存在且包含必要的文件")
            raise RuntimeError(f"Kronos模型导入失败: {e}")

        except ConnectionError as e:
            print(f"❌ 网络连接失败: {e}")
            print("\n🔧 解决方案:")
            print("  1. 检查网络连接是否正常")
            print("  2. 设置代理: export HTTPS_PROXY=http://your-proxy:port")
            print("  3. 或下载模型到本地后使用离线模式")
            raise RuntimeError(f"模型下载失败，请检查网络连接: {e}")

        except Exception as e:
            error_type = type(e).__name__
            print(f"❌ 模型加载失败 ({error_type}): {e}")
            print("\n🔧 解决方案:")

            # 提供具体的解决方案
            if "SSL" in str(e).upper() or "CERTIFICATE" in str(e).upper():
                print("  1. SSL证书问题，尝试以下步骤:")
                print("    - 升级网络库: pip install --upgrade requests urllib3 certifi")
                print("    - 设置代理: export HTTPS_PROXY=http://your-proxy:port")
                print("    - 或临时跳过SSL验证: export HF_HUB_DISABLE_SSL_VERIFICATION=1")
                print("  2. 检查网络连接和防火墙设置")
            elif "timeout" in str(e).lower() or "connection" in str(e).lower():
                print("  1. 网络连接问题:")
                print("    - 检查网络连接是否正常")
                print("    - 设置代理服务器")
                print("    - 尝试使用VPN")
                print("  2. 下载模型到本地后离线使用")
            elif "disk" in str(e).lower() or "space" in str(e).lower():
                print("  1. 磁盘空间不足:")
                print("    - 清理磁盘空间")
                print("    - 设置HF_HOME到其他目录: export HF_HOME=/path/to/large/disk")
            elif "memory" in str(e).lower() or "cuda" in str(e).lower():
                print("  1. 内存不足:")
                print("    - 使用CPU模式: export DEVICE=cpu")
                print("    - 减少max_context参数")
                print("    - 关闭其他程序释放内存")
            else:
                print("  1. 通用解决方案:")
                print("    - 检查HuggingFace token是否正确设置")
                print("    - 尝试重新安装相关库: pip install --upgrade transformers huggingface-hub")
                print("    - 检查是否有足够的内存和磁盘空间")
                print("    - 尝试重启Python环境")

            # 如果是SSL错误，尝试备用方案
            if "SSL" in str(e).upper() and "HF_HUB_DISABLE_SSL_VERIFICATION" not in os.environ:
                print("\n🔄 尝试自动修复SSL问题...")
                try:
                    # 再次设置环境变量并重试
                    os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
                    os.environ['REQUESTS_CA_BUNDLE'] = ''
                    os.environ['SSL_CERT_FILE'] = ''
                    print("✅ 已设置跳过SSL验证的环境变量，请重新运行程序")
                except Exception as retry_e:
                    print(f"❌ 自动修复失败: {retry_e}")

            raise RuntimeError(f"Kronos模型加载失败: {e}")

    def _load_model_offline(self):
        """离线模式：只使用本地缓存的模型"""
        try:
            print("  📂 尝试加载本地缓存的Tokenizer...")
            self.tokenizer = KronosTokenizer.from_pretrained(
                "NeoQuasar/Kronos-Tokenizer-base",
                local_files_only=True
            )
            print("  ✅ Tokenizer加载成功")

            print("  🤖 尝试加载本地缓存的模型...")
            self.model = Kronos.from_pretrained(
                "NeoQuasar/Kronos-base",
                local_files_only=True
            )
            print("  ✅ 模型加载成功")

        except Exception as e:
            print(f"❌ 离线模式加载失败: {e}")
            print("🔧 解决方案:")
            print("  1. 确保模型已下载到本地缓存 (~/.cache/huggingface/hub/)")
            print("  2. 或者先运行一次在线模式下载模型")
            print("  3. 检查网络连接和磁盘空间")
            raise RuntimeError(f"离线模式加载失败: {e}")

    def _should_update_model(self):
        """检查是否应该更新模型"""
        # 检查强制更新标志
        force_update = os.environ.get('KRONOS_FORCE_UPDATE', 'false').lower() == 'true'
        if force_update:
            print("  🔄 检测到强制更新标志，将更新模型")
            return True

        # 检查更新间隔（默认7天）
        update_interval_days = int(os.environ.get('KRONOS_UPDATE_INTERVAL_DAYS', '7'))

        try:
            # 检查版本跟踪文件
            version_file = os.path.join(os.path.dirname(__file__), '.model_version.json')
            if not os.path.exists(version_file):
                print(f"  📝 首次运行，将下载最新模型")
                return True

            import json
            with open(version_file, 'r') as f:
                version_info = json.load(f)

            last_update = datetime.fromisoformat(version_info.get('last_update', '2000-01-01T00:00:00'))
            days_since_update = (datetime.now() - last_update).days

            if days_since_update >= update_interval_days:
                print(f"  ⏰ 距离上次更新已过去{days_since_update}天，将检查模型更新")
                return True
            else:
                print(f"  ✅ 模型在{update_interval_days - days_since_update}天内已更新过，跳过网络检查")
                return False

        except Exception as e:
            print(f"  ⚠️ 版本检查失败: {e}，将尝试更新")
            return True

    def _update_version_info(self):
        """更新版本信息"""
        try:
            version_file = os.path.join(os.path.dirname(__file__), '.model_version.json')
            version_info = {
                'last_update': datetime.now().isoformat(),
                'tokenizer_repo': 'NeoQuasar/Kronos-Tokenizer-base',
                'model_repo': 'NeoQuasar/Kronos-base'
            }
            import json
            with open(version_file, 'w') as f:
                json.dump(version_info, f, indent=2)
            print("  📝 已更新版本信息")
        except Exception as e:
            print(f"  ⚠️ 更新版本信息失败: {e}")

    def _load_model_with_update(self):
        """在线模式：智能更新模型，失败时使用本地缓存"""
        from datetime import datetime

        # 首先尝试加载本地缓存的模型（作为备用）
        local_tokenizer = None
        local_model = None
        local_available = False

        try:
            print("  📂 检查本地缓存...")
            local_tokenizer = KronosTokenizer.from_pretrained(
                "NeoQuasar/Kronos-Tokenizer-base",
                local_files_only=True
            )
            local_model = Kronos.from_pretrained(
                "NeoQuasar/Kronos-base",
                local_files_only=True
            )
            local_available = True
            print("  ✅ 本地缓存可用")
        except Exception as e:
            print(f"  ⚠️ 本地缓存不可用: {e}")
            print("  📥 将下载最新模型")

        # 检查是否需要更新
        if not self._should_update_model():
            # 不需要更新，直接使用本地缓存
            if local_available:
                print("  🔄 使用本地缓存的模型...")
                self.tokenizer = local_tokenizer
                self.model = local_model
                print("  ✅ 使用本地缓存版本")
                return
            else:
                print("  ⚠️ 本地缓存不可用，将强制下载")
                # 继续到网络下载流程

        # 尝试从网络更新模型
        try:
            print("  🌐 从网络下载/更新模型...")

            # 设置网络下载的环境变量（如果没有设置的话）
            if 'HF_HUB_DISABLE_SSL_VERIFICATION' not in os.environ:
                os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
                os.environ['REQUESTS_CA_BUNDLE'] = ''
                os.environ['SSL_CERT_FILE'] = ''
                os.environ['CURL_CA_BUNDLE'] = ''

            self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            print("  ✅ Tokenizer加载成功")

            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
            print("  ✅ 模型加载成功")

            # 更新版本信息
            self._update_version_info()

        except Exception as network_error:
            print(f"  ❌ 网络加载失败: {network_error}")

            if local_available:
                print("  🔄 回退到本地缓存的模型...")
                self.tokenizer = local_tokenizer
                self.model = local_model
                print("  ✅ 已切换到本地缓存版本")
            else:
                print("  💥 网络加载失败且无本地缓存，加载失败")
                raise RuntimeError(f"模型加载失败: 网络错误且无本地缓存 - {network_error}")

    def load_data(self, filepath):
        """
        加载股票数据
        
        Args:
            filepath (str): 数据文件路径
            
        Returns:
            pd.DataFrame: 股票数据
        """
        try:
            print(f"正在加载数据: {filepath}")
            
            # 读取CSV文件
            df = pd.read_csv(filepath)
            
            # 确保时间戳列存在
            if 'timestamps' not in df.columns:
                print("错误: 数据文件中缺少timestamps列")
                return None
            
            # 转换时间戳格式
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            # 确保所有必需的列都存在
            required_columns = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"警告: 缺少列: {missing_columns}")
                # 尝试用其他列名匹配
                column_mapping = {
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Amount': 'amount'
                }
                
                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns and new_col not in df.columns:
                        df[new_col] = df[old_col]
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 仅基于关键列删除无效行，避免因额外列 NaN 丢失有效数据
            key_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df.dropna(subset=key_cols)
            
            # 按时间排序
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            print(f"成功加载 {len(df)} 条数据")
            print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
            
            return df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def prepare_prediction_data(self, df, lookback=1500, pred_len=96, use_close_only=False):
        """
        准备预测数据
        
        Args:
            df (pd.DataFrame): 股票数据
            lookback (int): 回看窗口大小
            pred_len (int): 预测长度
            use_close_only (bool): 仅使用 close 列
            
        Returns:
            tuple: (输入数据, 输入时间戳, 输出时间戳)
        """
        if len(df) < lookback + pred_len:
            print(f"警告: 数据长度({len(df)})小于lookback({lookback}) + pred_len({pred_len})")
            # 调整参数
            lookback = min(lookback, len(df) // 2)
            pred_len = min(pred_len, len(df) - lookback)
            print(f"调整参数: lookback={lookback}, pred_len={pred_len}")
        
        # 准备输入数据
        if use_close_only:
            x_df = df.loc[:lookback-1, ['close']]
        else:
            x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
            
        x_timestamp = df.loc[:lookback-1, 'timestamps']
        y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
        
        return x_df, x_timestamp, y_timestamp
    
    def prepare_backtest_data(self, df, lookback=1500, pred_len=96, use_close_only=False):
        """
        准备回测数据 - 正确切分训练和测试集
        
        Args:
            df: 完整历史数据
            lookback: 训练数据长度
            pred_len: 预测长度（回测长度）
            use_close_only (bool): 仅使用 close 列
            
        Returns:
            tuple: (训练数据, 训练时间戳, 预测时间戳, 真实测试数据)
        """
        if len(df) < lookback + pred_len:
            raise ValueError(f"数据长度({len(df)})不足以进行回测，需要至少 {lookback + pred_len} 个数据点")
        
        # 训练数据：前 lookback 行
        if use_close_only:
            x_df = df.iloc[:lookback][['close']].copy()
        else:
            x_df = df.iloc[:lookback][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            
        x_timestamp = df.iloc[:lookback]['timestamps'].copy()
        
        # 预测时间戳：接下来 pred_len 行的时间戳
        y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps'].copy()
        
        # 真实数据：用于后续验证（ground truth，这里保存所有列以便进行全面的回测验证）
        ground_truth = df.iloc[lookback:lookback+pred_len][['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        ground_truth.index = y_timestamp.values  # 设置索引为时间戳
        
        logger.info(f"回测数据准备完成: 训练集 {len(x_df)} 行, 预测集 {len(ground_truth)} 行")
        logger.info(f"训练时间范围: {x_timestamp.iloc[0]} 至 {x_timestamp.iloc[-1]}")
        logger.info(f"预测时间范围: {y_timestamp.iloc[0]} 至 {y_timestamp.iloc[-1]}")
        
        return x_df, x_timestamp, y_timestamp, ground_truth
    
    def predict(self, x_df, x_timestamp, y_timestamp, pred_len, T=0.5, top_p=0.5, sample_count=1):
        """
        进行预测
        
        Args:
            x_df (pd.DataFrame): 输入特征
            x_timestamp (pd.Series): 输入时间戳
            y_timestamp (pd.Series): 预测时间戳
            pred_len (int): 预测长度
            T (float): 采样温度 举例：若预测 “明天是否下雨”，T=0.1 时，模型会坚定选择训练数据中最可能的结果（如 “下雨” 概率 80% 则直接输出）；若 T 过高（如 0.8），可能因随机采样输出低概率选项（如 “晴天”），偏离真实趋势。
            top_p (float): 核采样概率  若场景中 “真相” 高度唯一（如预测具体数值、明确分类），top_p 可降至 0.5-0.6，进一步聚焦；若数据存在轻微不确定性（如多因素影响的预测），0.7-0.8 更稳妥。
            sample_count (int): 采样次数
            
        Returns:
            pd.DataFrame: 预测结果
        """
        import time
        import os

        start_time = time.time()

        # 尝试导入psutil，如果失败则降级到基础监控
        try:
            import psutil
            psutil_available = True
            initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            psutil_available = False
            initial_memory = 0
            self.logger.warning("psutil不可用，性能监控功能将降级")

        try:
            self.logger.info("🔮 开始模型推理...")
            self.performance_stats['predictions_count'] += 1

            # 验证输入参数
            if x_df is None or x_df.empty:
                raise ValueError("输入数据为空")
            if len(y_timestamp) != pred_len:
                raise ValueError(f"预测时间戳长度({len(y_timestamp)})与预测长度({pred_len})不匹配")

            # 自适应参数调整
            T, top_p, sample_count = self._adaptive_parameter_tuning(x_df, T, top_p, sample_count, pred_len, self.enable_adaptive_tuning)

            self.logger.info(f"使用参数: T={T:.2f}, top_p={top_p:.2f}, sample_count={sample_count}")

            # 内存优化：在大预测前清理内存
            if pred_len > 200:
                self.optimize_memory_usage()

            # 执行预测 - 调用Kronos大模型进行推理
            self.logger.info(f"🧠 正在调用Kronos大模型进行预测推理 (预测长度: {pred_len})...")
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=T,
                top_p=top_p,
                sample_count=sample_count,
                verbose=True  # 显示推理过程
            )
            self.logger.info("✅ Kronos大模型推理完成！")

            if pred_df is not None:
                # 后处理预测结果
                pred_df = self._postprocess_predictions(pred_df, x_df)
                
                # 【新增】应用逆归一化，将预测结果还原到原始价格尺度
                pred_df = self._inverse_normalization(pred_df)

                # 更新性能统计
                inference_time = time.time() - start_time

                if psutil_available:
                    current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    memory_used = current_memory - initial_memory
                    self.performance_stats['total_inference_time'] += inference_time
                    self.performance_stats['memory_peak'] = max(self.performance_stats['memory_peak'], memory_used)
                    self.logger.info(f"✅ 预测完成并优化！ (耗时: {inference_time:.2f}s, 内存: {memory_used:.1f}MB)")
                else:
                    self.performance_stats['total_inference_time'] += inference_time
                    self.logger.info(f"✅ 预测完成并优化！ (耗时: {inference_time:.2f}s)")
            else:
                self.performance_stats['errors_count'] += 1
                self.logger.error("❌ 预测返回None")

            return pred_df
            
        except (MemoryError, RuntimeError) as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "memory" in error_msg:
                self.logger.error(f"检测到内存不足错误: {e}")
                self.optimize_memory_usage()

                # 如果使用MPS遇到内存不足，尝试切换到CPU
                if self.device == 'mps':
                    self.logger.warning("🔄 MPS内存不足，自动切换到CPU模式...")
                    try:
                        # 重新初始化预测器为CPU模式
                        self.device = 'cpu'
                        self.predictor = KronosPredictor(
                            self.model.to('cpu'),
                            self.tokenizer.to('cpu'),
                            device='cpu',
                            max_context=self.max_context
                        )
                        self.logger.info("✅ 已切换到CPU模式，重试预测...")

                        # 递归调用自己，使用CPU模式
                        return self.predict(x_df, x_timestamp, y_timestamp, pred_len, T, top_p, sample_count)

                    except Exception as cpu_error:
                        self.logger.error(f"❌ CPU模式也失败: {cpu_error}")
                        return None
                else:
                    self.logger.error("❌ 当前已是CPU模式，内存仍然不足")
                    return None
            else:
                # 不是内存错误，重新抛出
                raise e
        except Exception as e:
            self.performance_stats['errors_count'] += 1
            self.logger.error(f"预测失败: {str(e)}")
            self.logger.error("详细错误信息:", exc_info=True)
            return None
    
    def _adaptive_parameter_tuning(self, x_df, T, top_p, sample_count, pred_len, enable_adaptive=True):
        """
        自适应参数调优

        根据数据特征和预测长度自动调整参数
        """
        # 如果禁用自适应调整，直接返回原始参数
        if not enable_adaptive:
            self.logger.info(f"自适应参数调优已禁用，使用用户指定的参数: T={T}, top_p={top_p}")
            return T, top_p, sample_count

        # 计算数据的波动性
        close_volatility = x_df['close'].pct_change().std()

        # 根据波动性调整参数
        if close_volatility > 0.02:  # 高波动性
            T = min(T * 0.8, 0.5)  # 降低温度，使预测更保守
            top_p = min(top_p * 0.9, 0.6)  # 降低多样性
        elif close_volatility < 0.005:  # 低波动性
            T = max(T * 1.2, 0.3)  # 可以适当增加温度
            top_p = max(top_p * 1.1, 0.4)

        # 根据预测长度调整采样次数
        if pred_len > 100:  # 长序列预测
            sample_count = min(sample_count + 1, 5)  # 增加采样次数提高稳定性
        elif pred_len < 20:  # 短序列预测
            sample_count = max(sample_count - 1, 1)  # 可以减少采样次数

        # 确保参数在合理范围内
        T = max(0.1, min(T, 2.0))
        top_p = max(0.1, min(top_p, 0.95))
        sample_count = max(1, min(sample_count, 10))

        return T, top_p, sample_count

    def _postprocess_predictions(self, pred_df, x_df):
        """
        预测结果后处理

        确保预测结果的合理性和连续性
        """
        processed_df = pred_df.copy()

        # 1. 确保价格连续性（防止跳跃）
        for col in ['open', 'high', 'low', 'close']:
            if col in processed_df.columns:
                # 计算相邻预测值的变化率
                pct_change = processed_df[col].pct_change()

                # 识别异常变化（超过50%的单步变化）
                outlier_mask = pct_change.abs() > 0.5

                if outlier_mask.any():
                    logger.warning(f"检测到 {outlier_mask.sum()} 个异常价格变化，已进行平滑处理")

                    # 使用移动平均平滑异常值
                    processed_df[col] = processed_df[col].where(
                        ~outlier_mask,
                        processed_df[col].rolling(window=3, center=True, min_periods=1).mean()
                    )

        if 'volume' in processed_df.columns:
            processed_df['volume'] = processed_df['volume'].clip(lower=0)
        if 'amount' in processed_df.columns:
            processed_df['amount'] = processed_df['amount'].clip(lower=0)

        # 3. A股涨跌停价格约束（–10%/日，防止预测超出交易规则上限）
        try:
            last_close = float(x_df['close'].iloc[-1])
            if last_close > 0 and len(pred_df.index) >= 1:
                # 从预测时间戳推断数据频率
                if len(processed_df.index) >= 2:
                    step_minutes = (processed_df.index[1] - processed_df.index[0]).total_seconds() / 60
                else:
                    step_minutes = 60

                if step_minutes >= 1440:
                    # 日线：逐步滚动约束 ±10%/天
                    per_step_limit = 0.10
                    for col in ['open', 'high', 'low', 'close']:
                        if col not in processed_df.columns:
                            continue
                        prev_price = last_close
                        clipped = []
                        for val in processed_df[col]:
                            v = float(val)
                            v = min(v, prev_price * (1 + per_step_limit))
                            v = max(v, prev_price * (1 - per_step_limit))
                            clipped.append(v)
                            prev_price = v  # 下一步以当前预测价为基准
                        processed_df[col] = clipped
                    logger.info("已应用日线涨跌停约束（逐日滚动 ±10%）")
                else:
                    # 分钟线：按预测总天数设置整体允许区间
                    steps_per_day = max(1, 240 / step_minutes)
                    num_trading_days = max(1, len(processed_df) / steps_per_day)
                    total_limit = 0.10 * num_trading_days
                    for col in ['open', 'high', 'low', 'close']:
                        if col in processed_df.columns:
                            processed_df[col] = processed_df[col].clip(
                                lower=last_close * (1 - total_limit),
                                upper=last_close * (1 + total_limit)
                            )
                    logger.info(f"已应用分钟线价格区间约束（预测{num_trading_days:.0f}交易日区间 ±{total_limit:.0%}）")
        except Exception as e:
            logger.warning(f"A股涨跌停约束应用失败（不影响预测结果）: {e}")

        # 4. 确保 OHLC 关系合理（放在涨跌停约束之后，保证最终一致性）
        if all(c in processed_df.columns for c in ['open', 'high', 'low', 'close']):
            processed_df['high'] = processed_df[['open', 'close', 'high']].max(axis=1)
            processed_df['low'] = processed_df[['open', 'close', 'low']].min(axis=1)

        return processed_df

    def _calculate_smart_xticks(self, timestamps, pred_start_time=None, max_ticks=12, is_future_forecast=False):
        """
        计算智能时间轴刻度，保证不拥挤、不重叠
        """
        if timestamps.empty:
            return [], []

        timestamps = pd.to_datetime(timestamps)
        start_time = timestamps.min()
        end_time = timestamps.max()
        total_days = (end_time - start_time).total_seconds() / 86400

        if total_days <= 0:
            return [start_time], [start_time.strftime('%m-%d')]

        pred_start = pd.to_datetime(pred_start_time) if pred_start_time is not None else None

        # 根据总天数选择合适的刻度间隔
        if total_days <= 7:
            freq = 'D'
        elif total_days <= 30:
            freq = '3D'
        elif total_days <= 90:
            freq = 'W'
        elif total_days <= 365:
            freq = '2W'
        else:
            freq = 'MS'

        base_ticks = list(pd.date_range(start=start_time, end=end_time, freq=freq))

        # 确保首尾时间点在刻度中
        if base_ticks and base_ticks[0] > start_time:
            base_ticks.insert(0, start_time)
        if base_ticks and base_ticks[-1] < end_time:
            base_ticks.append(end_time)

        # 加入预测开始时间作为关键刻度
        if pred_start is not None and start_time <= pred_start <= end_time:
            base_ticks.append(pred_start)

        all_ticks = sorted(set(base_ticks))

        # 去掉过于接近的刻度（间距小于总跨度的5%则合并，保留关键刻度）
        min_gap = pd.Timedelta(days=max(1, total_days * 0.05))
        filtered = []
        for tick in all_ticks:
            is_key = pred_start is not None and tick == pred_start
            if not filtered:
                filtered.append(tick)
            elif tick - filtered[-1] < min_gap:
                prev_is_key = pred_start is not None and filtered[-1] == pred_start
                if is_key and not prev_is_key:
                    filtered[-1] = tick
            else:
                filtered.append(tick)
        all_ticks = filtered

        # 限制最大刻度数量（均匀采样，但保留关键刻度）
        if len(all_ticks) > max_ticks:
            key_indices = set()
            if pred_start is not None:
                for i, t in enumerate(all_ticks):
                    if t == pred_start:
                        key_indices.add(i)
            key_indices.add(0)
            key_indices.add(len(all_ticks) - 1)

            other_indices = [i for i in range(len(all_ticks)) if i not in key_indices]
            remaining = max_ticks - len(key_indices)
            if remaining > 0 and other_indices:
                step = max(1, len(other_indices) // remaining)
                sampled = set(other_indices[::step][:remaining])
            else:
                sampled = set()
            keep = sorted(key_indices | sampled)
            all_ticks = [all_ticks[i] for i in keep]

        # 生成标签：统一使用短格式
        tick_labels = []
        for tick in all_ticks:
            is_key = pred_start is not None and abs(tick - pred_start) < pd.Timedelta(hours=12)
            if total_days <= 7:
                tick_labels.append(tick.strftime('%m-%d'))
            elif total_days <= 180:
                tick_labels.append(tick.strftime('%m-%d'))
            else:
                tick_labels.append(tick.strftime('%Y-%m'))

        return all_ticks, tick_labels
    
    def plot_prediction(self, historical_df, pred_df, symbol, is_future_forecast=False, save_plot=True,
                       plot_lookback=1500, enable_focus_mode=False, plot_lookback_days=None, prediction_highlight=True,
                       raw_historical_df=None, run_dir=None):
        """
        绘制预测结果 - 智能时间轴显示，区分预测和回测模式
        """
        try:
            logger.info("开始绘制预测图表...")
            logger.info(f"输入参数: symbol={symbol}, is_future_forecast={is_future_forecast}, save_plot={save_plot}, plot_lookback={plot_lookback}")
            logger.info(f"历史数据形状: {historical_df.shape if historical_df is not None else 'None'}")
            logger.info(f"预测数据形状: {pred_df.shape if pred_df is not None else 'None'}")

            # 验证输入数据
            if pred_df is None or pred_df.empty:
                logger.error("预测数据为空，无法绘制图表")
                return None

            if historical_df is None or historical_df.empty:
                logger.error("历史数据为空，无法绘制图表")
                return None

            # 设置matplotlib后端为非交互式
            logger.info("配置matplotlib...")
            import matplotlib
            matplotlib.use('Agg')  # 确保使用非交互式后端
            import matplotlib.pyplot as plt
            logger.info(f"matplotlib后端: {matplotlib.get_backend()}")

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            logger.info("matplotlib配置完成")
            
            # 准备绘图数据
            start_pred_time = pred_df.index.min()
            logger.info(f"预测开始时间: {start_pred_time}, 类型: {type(start_pred_time)}")

            # 检查数据类型
            logger.info(f"历史数据timestamps类型: {historical_df['timestamps'].dtype}")
            logger.info(f"预测数据索引类型: {pred_df.index.dtype}")

            # 确保时间戳格式一致并筛选数据
            try:
                hist_timestamps = pd.to_datetime(historical_df['timestamps'])
                pred_start = pd.to_datetime(start_pred_time)

                if enable_focus_mode and plot_lookback_days:
                    focus_start = pred_start - pd.Timedelta(days=plot_lookback_days)
                    focus_end = pred_df.index.max() + pd.Timedelta(days=FOCUS_MODE_MARGIN_DAYS)
                    historical_plot_df = historical_df[
                        (hist_timestamps >= focus_start) & (hist_timestamps < pred_start)
                    ]
                    if len(historical_plot_df) < 3:
                        logger.info(f"专注模式数据不足({len(historical_plot_df)}条)，回退到普通模式")
                        historical_plot_df = historical_df[hist_timestamps < pred_start].tail(plot_lookback)
                    else:
                        logger.info(f"专注模式: 显示预测前{plot_lookback_days}天的历史数据({len(historical_plot_df)}条)")
                else:
                    historical_plot_df = historical_df[hist_timestamps < pred_start].tail(plot_lookback)

                logger.info(f"历史数据点数: {len(historical_plot_df)}")
            except Exception as e:
                logger.error(f"时间戳处理失败: {e}")
                return None

            # 合并所有时间点用于智能刻度计算
            timestamps_list = []
            if not historical_plot_df.empty:
                timestamps_list.append(historical_plot_df['timestamps'])
            timestamps_list.append(pd.Series(pred_df.index))

            all_timestamps = pd.concat(timestamps_list).sort_values().drop_duplicates()
            logger.info(f"总时间点数: {len(all_timestamps)}")

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
            logger.info("matplotlib图形创建完成")
            
            # --- 价格图 ---
            # 使用原始历史数据（如果提供）来确保正确的价格尺度显示
            plot_hist_df = historical_plot_df
            if raw_historical_df is not None and not raw_historical_df.empty:
                # 尝试匹配时间戳来获取原始数据
                try:
                    # 根据时间戳匹配原始历史数据
                    hist_timestamps = historical_plot_df['timestamps']
                    raw_plot_df = raw_historical_df[raw_historical_df['timestamps'].isin(hist_timestamps)]
                    if not raw_plot_df.empty:
                        plot_hist_df = raw_plot_df.copy()
                        logger.info("使用原始历史数据进行图表绘制，确保价格尺度正确")
                except Exception as e:
                    logger.warning(f"无法使用原始数据绘图，使用预处理数据: {e}")

            ax1.plot(plot_hist_df['timestamps'], plot_hist_df['close'], label='历史价格', color='blue', linewidth=1.5)

            pred_len = len(pred_df)
            is_short_pred = pred_len <= 3

            # 预测价格线条（支持高亮）
            if prediction_highlight:
                pred_start_time = pred_df.index.min()
                pred_end_time = pred_df.index.max()

                if is_short_pred:
                    # 短预测用更宽的高亮带提升可见度
                    highlight_margin = pd.Timedelta(days=1)
                    ax1.axvspan(pred_start_time - highlight_margin, pred_end_time + highlight_margin,
                               alpha=0.15, color='red', label='预测区域')
                else:
                    ax1.axvspan(pred_start_time, pred_end_time, alpha=0.1, color='red', label='预测区域')

                marker_size = 10 if is_short_pred else 4
                ax1.plot(pred_df.index, pred_df['close'], label='预测价格',
                        color='red', linewidth=3, linestyle='--', marker='o',
                        markersize=marker_size, alpha=0.9, zorder=5)

                # 单点预测时在旁边标注价格值
                if pred_len == 1:
                    price_val = pred_df['close'].iloc[0]
                    ax1.annotate(f'{price_val:.2f}',
                                xy=(pred_df.index[0], price_val),
                                xytext=(15, 15), textcoords='offset points',
                                fontsize=11, fontweight='bold', color='red',
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='red', alpha=0.9),
                                zorder=6)
            else:
                ax1.plot(pred_df.index, pred_df['close'], label='预测价格', color='red', linewidth=2, linestyle='--')

            # 回测模式添加真实价格
            if not is_future_forecast:
                true_values_df = historical_df[historical_df['timestamps'].isin(pred_df.index)]
                if not true_values_df.empty:
                    marker_size = 10 if is_short_pred else 4
                    ax1.plot(true_values_df['timestamps'], true_values_df['close'], label='真实价格',
                            color='green', linewidth=2, alpha=0.8,
                            marker='s' if is_short_pred else None,
                            markersize=marker_size, zorder=5)
                    if pred_len == 1:
                        true_val = true_values_df['close'].iloc[0]
                        ax1.annotate(f'{true_val:.2f}',
                                    xy=(true_values_df['timestamps'].iloc[0], true_val),
                                    xytext=(15, -20), textcoords='offset points',
                                    fontsize=11, fontweight='bold', color='green',
                                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew', edgecolor='green', alpha=0.9),
                                    zorder=6)

            current_time = pd.Timestamp.now()
            pred_start_time = pred_df.index.min()

            # 用单条分隔线标记预测起始
            ax1.axvline(pred_start_time, color='orange', linestyle='--', linewidth=1.5, alpha=0.8)

            # 先设置 Y 轴范围，再定位文本标签
            all_prices = pd.concat([
                plot_hist_df['close'],
                pred_df['close']
            ])
            if not is_future_forecast:
                true_prices = historical_df[historical_df['timestamps'].isin(pred_df.index)]['close']
                all_prices = pd.concat([all_prices, true_prices])

            price_min, price_max = all_prices.min(), all_prices.max()
            price_range = price_max - price_min
            if price_range > 0:
                margin = price_range * 0.1
                ax1.set_ylim(price_min - margin, price_max + margin)

            # Y 轴确定后再放置标签（使用 transAxes 相对坐标避免坐标系不一致）
            ax1.text(pred_start_time, ax1.get_ylim()[1] - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.02,
                    '预测开始', ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7))

            if is_future_forecast and abs(current_time - all_timestamps.max()) < pd.Timedelta(days=7):
                ax1.axvline(current_time, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
                ax1.text(current_time, ax1.get_ylim()[1] - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.12,
                        '当前时间', ha='center', va='top', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='purple', alpha=0.7),
                        color='white')

            ax1.set_ylabel('价格', fontsize=14)
            mode_name = '未来预测' if is_future_forecast else '历史回测'
            mode_color = 'orange' if is_future_forecast else 'blue'
            mode_desc = '预测未来趋势' if is_future_forecast else '验证历史表现'

            ax1.set_title(f'{symbol} 股票{mode_name}结果 - 智能时间轴', fontsize=16)

            ax1.text(0.02, 0.98, f'{mode_name}\n{mode_desc}',
                    transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=mode_color, alpha=0.8),
                    color='white', fontweight='bold')

            ax1.legend(loc='upper center', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # --- 成交量图 ---
            has_volume = 'volume' in plot_hist_df.columns and 'volume' in pred_df.columns
            if has_volume:
                ax2.plot(plot_hist_df['timestamps'], plot_hist_df['volume'], label='历史成交量', color='blue', linewidth=1.5)
                vol_marker = 'o' if is_short_pred else None
                vol_ms = 8 if is_short_pred else 4
                ax2.plot(pred_df.index, pred_df['volume'], label='预测成交量',
                        color='red', linewidth=2, linestyle='--',
                        marker=vol_marker, markersize=vol_ms, zorder=5)

                if not is_future_forecast:
                    true_values_df = historical_df[historical_df['timestamps'].isin(pred_df.index)]
                    if not true_values_df.empty and 'volume' in true_values_df.columns:
                        ax2.plot(true_values_df['timestamps'], true_values_df['volume'], label='真实成交量',
                                color='green', linewidth=2, alpha=0.8,
                                marker='s' if is_short_pred else None,
                                markersize=vol_ms, zorder=5)
            else:
                ax2.text(0.5, 0.5, '无成交量数据', transform=ax2.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')

            ax2.axvline(pred_start_time, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

            if is_future_forecast and abs(current_time - all_timestamps.max()) < pd.Timedelta(days=7):
                ax2.axvline(current_time, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
            
            ax2.set_ylabel('成交量', fontsize=14)
            ax2.set_xlabel('时间', fontsize=14)
            ax2.legend(loc='upper left', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # 设置智能时间轴刻度
            smart_ticks, tick_labels = self._calculate_smart_xticks(
                all_timestamps,
                pred_start_time=pred_df.index.min(),
                max_ticks=12,
                is_future_forecast=is_future_forecast
            )
            if smart_ticks:
                ax2.set_xticks(smart_ticks)
                ax2.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=9)

            # 放大子图：仅当历史数据足够多、且预测区域相对较短时才显示
            total_duration = all_timestamps.max() - all_timestamps.min()
            pred_duration = pred_df.index.max() - pred_df.index.min()
            hist_count = len(historical_plot_df)

            show_zoom = (
                total_duration > pd.Timedelta(days=20) and
                pred_duration < total_duration * 0.3 and
                hist_count > 10
            )
            if show_zoom:
                # 放大窗口：预测前5天到预测结束后1天
                zoom_margin_before = pd.Timedelta(days=5)
                zoom_margin_after = pd.Timedelta(days=max(1, pred_duration.days + 1))
                zoom_start = pred_df.index.min() - zoom_margin_before
                zoom_end = pred_df.index.max() + zoom_margin_after

                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_zoom = inset_axes(ax1, width="35%", height="30%", loc='upper right',
                                    bbox_to_anchor=(0.02, 0.02, 0.96, 0.96),
                                    bbox_transform=ax1.transAxes)

                zoom_hist = historical_plot_df[
                    (historical_plot_df['timestamps'] >= zoom_start) &
                    (historical_plot_df['timestamps'] <= zoom_end)
                ]
                zoom_pred = pred_df[
                    (pred_df.index >= zoom_start) &
                    (pred_df.index <= zoom_end)
                ]

                if not zoom_hist.empty:
                    ax_zoom.plot(zoom_hist['timestamps'], zoom_hist['close'],
                               color='blue', linewidth=1.2, alpha=0.8)
                if not zoom_pred.empty:
                    ax_zoom.plot(zoom_pred.index, zoom_pred['close'],
                               color='red', linewidth=2, linestyle='--', marker='o',
                               markersize=6, alpha=0.9)

                if not is_future_forecast:
                    zoom_true = historical_df[
                        (historical_df['timestamps'].isin(pred_df.index)) &
                        (historical_df['timestamps'] >= zoom_start) &
                        (historical_df['timestamps'] <= zoom_end)
                    ]
                    if not zoom_true.empty:
                        ax_zoom.plot(zoom_true['timestamps'], zoom_true['close'],
                                   color='green', linewidth=1.5, marker='s', markersize=5, alpha=0.8)

                ax_zoom.set_title('预测区域放大', fontsize=9, pad=3)
                ax_zoom.tick_params(labelsize=7)
                ax_zoom.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
                ax_zoom.tick_params(axis='x', rotation=20)
                ax_zoom.grid(True, alpha=0.3)
                for spine in ax_zoom.spines.values():
                    spine.set_edgecolor('orange')
                    spine.set_linewidth(1.5)

            plt.tight_layout()
            
            # 保存图表
            plot_path = None
            if save_plot:
                try:
                    mode_prefix = 'forecast' if is_future_forecast else 'backtest'
                    
                    if run_dir:
                        symbol_results_dir = run_dir
                        plot_filename = f"{symbol}_{mode_prefix}_chart.png"
                    else:
                        # 兼容旧逻辑
                        mode_dir = 'future_forecast' if is_future_forecast else 'backtest'
                        symbol_results_dir = os.path.join(self.results_dir, symbol, mode_dir)
                        os.makedirs(symbol_results_dir, exist_ok=True)
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        plot_filename = f"{symbol}_{mode_prefix}_chart_{timestamp}.png"
                        
                    logger.info(f"创建或使用结果目录: {symbol_results_dir}")
                    plot_path = os.path.join(symbol_results_dir, plot_filename)

                    logger.info(f"正在保存图表到: {plot_path}")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

                    # 验证文件是否成功保存
                    if os.path.exists(plot_path):
                        file_size = os.path.getsize(plot_path)
                        logger.info(f"✅ 图表保存成功: {plot_path} ({file_size} bytes)")
                    else:
                        logger.error(f"❌ 图表文件不存在: {plot_path}")
                        plot_path = None

                except Exception as save_error:
                    logger.error(f"保存图表失败: {str(save_error)}")
                    plot_path = None

            logger.info("图表绘制流程完成")
            return plot_path

        except Exception as e:
            logger.error(f"绘制图表失败: {str(e)}")
            logger.error("详细错误信息:", exc_info=True)
            return None
        finally:
            plt.close('all')
    
    def save_prediction_results(self, pred_df, symbol, metadata=None, is_future_forecast=False, run_dir=None):
        """
        保存预测结果
        
        Args:
            pred_df (pd.DataFrame): 预测结果
            symbol (str): 股票代码
            metadata (dict): 元数据
            
        Returns:
            str: 结果文件路径
        """
        try:
            mode_prefix = 'forecast' if is_future_forecast else 'backtest'
            
            if run_dir:
                symbol_results_dir = run_dir
                timestamp = os.path.basename(run_dir)
                csv_filename = f"{symbol}_{mode_prefix}_data.csv"
                json_filename = f"{symbol}_{mode_prefix}_metadata.json"
            else:
                # 兼容旧逻辑
                mode_dir = 'future_forecast' if is_future_forecast else 'backtest'
                symbol_results_dir = os.path.join(self.results_dir, symbol, mode_dir)
                os.makedirs(symbol_results_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                csv_filename = f"{symbol}_{mode_prefix}_data_{timestamp}.csv"
                json_filename = f"{symbol}_{mode_prefix}_metadata_{timestamp}.json"

            csv_path = os.path.join(symbol_results_dir, csv_filename)
            json_path = os.path.join(symbol_results_dir, json_filename)
            
            # 将索引重置为列，并确保列名为'timestamps'
            save_df = pred_df.reset_index()
            if 'index' in save_df.columns:
                save_df = save_df.rename(columns={'index': 'timestamps'})
            
            save_df.to_csv(csv_path, index=False)
            
            # 保存元数据
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'symbol': symbol,
                'prediction_time': timestamp,
                'data_points': len(pred_df),
                'columns': list(pred_df.columns)
            })
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            
            mode_desc = "未来预测" if is_future_forecast else "历史回测"
            print(f"✅ {mode_desc}结果已保存至专门文件夹:")
            print(f"   📂 目录: {symbol_results_dir}")
            print(f"   📄 数据文件: {os.path.basename(csv_path)}")
            print(f"   📋 元数据文件: {os.path.basename(json_path)}")
            print(f"   🎯 模式: {mode_desc}")
            
            return csv_path
            
        except Exception as e:
            print(f"保存结果失败: {e}")
            return None
    
    def analyze_prediction(self, historical_df, pred_df, symbol, is_future_forecast):
        """
        全面分析预测结果
        """
        try:
            logger.info("开始详细分析预测结果...")

            # 获取历史数据的最后一个点用于比较
            last_historical_point = historical_df.iloc[-1]
            last_close = last_historical_point['close']
            
            # === 1. 基础统计分析 ===
            pred_close_stats = self._calculate_price_statistics(pred_df)
            pred_volume_stats = self._calculate_volume_statistics(pred_df)

            # === 2. 趋势和变化分析 ===
            trend_analysis = self._analyze_trend_and_changes(historical_df, pred_df, is_future_forecast)

            # === 3. 风险和波动性分析 ===
            risk_analysis = self._analyze_risk_and_volatility(historical_df, pred_df)

            # === 4. 预测质量评估 ===
            quality_metrics = self._evaluate_prediction_quality(historical_df, pred_df, is_future_forecast)

            # === 5. 技术指标分析 ===
            technical_analysis = self._calculate_technical_indicators(pred_df, historical_df)

            # 整合所有分析结果
            analysis = {
                'symbol': symbol,
                'historical_last_close': last_close,
                'prediction_period_days': len(pred_df),
                'is_future_forecast': is_future_forecast,
                'price_analysis': {
                    **pred_close_stats,
                    **trend_analysis['price']
                },
                'volume_analysis': pred_volume_stats,
                'risk_analysis': risk_analysis,
                'quality_metrics': quality_metrics,
                'technical_analysis': technical_analysis,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"预测分析完成 - 趋势: {analysis['price_analysis']['trend']}, "
                       f"预期变化: {analysis['price_analysis']['price_change_percentage']:.2f}%")
            return analysis

        except Exception as e:
            logger.error(f"预测结果分析失败: {str(e)}")
            return None

    def _calculate_price_statistics(self, pred_df):
        """计算价格基础统计"""
        close_prices = pred_df['close']

        return {
            'mean': close_prices.mean(),
            'std': close_prices.std(),
            'min': close_prices.min(),
            'max': close_prices.max(),
            'median': close_prices.median(),
            'q25': close_prices.quantile(0.25),
            'q75': close_prices.quantile(0.75),
            'range': close_prices.max() - close_prices.min(),
            'cv': close_prices.std() / close_prices.mean() if close_prices.mean() != 0 else 0
        }

    def _calculate_volume_statistics(self, pred_df):
        """计算成交量统计"""
        if 'volume' not in pred_df.columns:
            return {'trend': '无数据', 'avg_volume': 0}

        volumes = pred_df['volume']

        return {
            'mean': volumes.mean(),
            'std': volumes.std(),
            'min': volumes.min(),
            'max': volumes.max(),
            'trend': '增加' if volumes.iloc[-1] > volumes.iloc[0] else '减少',
            'volatility': volumes.std() / volumes.mean() if volumes.mean() != 0 else 0
        }

    def _analyze_trend_and_changes(self, historical_df, pred_df, is_future_forecast):
        """分析趋势和变化"""
        # 确定比较基准点
        if is_future_forecast:
            baseline_close = historical_df.iloc[-1]['close']
        else:
            comparison_point = historical_df[historical_df['timestamps'] < pred_df.index.min()]
            baseline_close = comparison_point.iloc[-1]['close'] if not comparison_point.empty else historical_df.iloc[-1]['close']

        pred_start = pred_df['close'].iloc[0]
        pred_end = pred_df['close'].iloc[-1]

        price_change = pred_end - baseline_close
        price_change_pct = (price_change / baseline_close) * 100 if baseline_close != 0 else 0

        pred_trend_strength = abs(pred_end - pred_start) / pred_start if (pred_start != 0 and len(pred_df) > 1) else 0

        return {
            'price': {
                'baseline_close': baseline_close,
                'pred_start': pred_start,
                'pred_end': pred_end,
                    'price_change': price_change,
                    'price_change_percentage': price_change_pct,
                'trend': '上涨' if pred_end > pred_start else '下跌',
                'trend_strength': pred_trend_strength,
                'direction_consistency': '一致' if (pred_end > pred_start) == (price_change > 0) else '不一致'
            }
        }

    def _analyze_risk_and_volatility(self, historical_df, pred_df):
        """分析风险和波动性"""
        pred_returns = pred_df['close'].pct_change().dropna()

        # 预测点太少时使用历史波动率作为近似
        hist_returns = historical_df['close'].pct_change().dropna().tail(100)
        hist_volatility = hist_returns.std() * (252 ** 0.5) if len(hist_returns) > 0 else 0

        if len(pred_returns) < 2:
            return {
                'pred_volatility': 0,
                'hist_volatility': hist_volatility,
                'volatility_ratio': 0,
                'var_95': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }

        pred_volatility = pred_returns.std() * (252 ** 0.5)
        var_95 = np.percentile(pred_returns, 5)
        max_drawdown = self._calculate_max_drawdown(pred_df['close'])

        return {
            'pred_volatility': pred_volatility,
            'hist_volatility': hist_volatility,
            'volatility_ratio': pred_volatility / hist_volatility if hist_volatility != 0 else float('inf'),
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': pred_returns.mean() / pred_returns.std() * (252 ** 0.5) if pred_returns.std() != 0 else 0
        }

    def _evaluate_prediction_quality(self, historical_df, pred_df, is_future_forecast):
        """评估预测质量"""
        quality_scores = {}
        returns = pred_df['close'].pct_change().dropna()

        if len(returns) < 2:
            quality_scores['smoothness'] = 1.0
            quality_scores['reasonableness'] = 0.5
            quality_scores['continuity'] = 1.0
            quality_scores['overall_score'] = 0.83
            return quality_scores

        smoothness = 1 / (1 + returns.std())
        quality_scores['smoothness'] = smoothness

        hist_volatility = historical_df['close'].pct_change().dropna().tail(50).std()
        pred_volatility = returns.std()

        if hist_volatility > 0 and pred_volatility > 0:
            volatility_ratio = pred_volatility / hist_volatility
            if 0.5 <= volatility_ratio <= 2.0:
                reasonableness = 1.0
            else:
                reasonableness = max(0, 1 - abs(np.log(volatility_ratio)))
        else:
            reasonableness = 0.5

        quality_scores['reasonableness'] = reasonableness

        jumps = (returns.abs() > 0.1).sum()
        continuity = max(0, 1 - jumps / len(returns))
        quality_scores['continuity'] = continuity

        quality_scores['overall_score'] = np.mean([smoothness, reasonableness, continuity])

        return quality_scores

    def _calculate_technical_indicators(self, pred_df, historical_df=None):
        """
        计算技术指标
        如果提供了historical_df，会将其与pred_df合并以计算更准确的指标（如MA20需要至少20天数据）
        """
        indicators = {}
        
        # 准备计算用的数据
        if historical_df is not None and not historical_df.empty:
            # 取历史数据的最后一部分，确保足够计算长周期指标 (如MACD需要26+9=35天，MA60需要60天)
            # 取100天应该足够
            hist_subset = historical_df.tail(100)[['close']].copy()
            pred_subset = pred_df[['close']].copy()
            combined_df = pd.concat([hist_subset, pred_subset])
        else:
            combined_df = pred_df[['close']].copy()

        if len(combined_df) < 5:  # 数据点太少
            return indicators

        try:
            # 使用TechnicalAnalyzer计算
            # MA
            ma5 = TechnicalAnalyzer.calculate_ma(combined_df['close'], 5)
            ma10 = TechnicalAnalyzer.calculate_ma(combined_df['close'], 10)
            ma20 = TechnicalAnalyzer.calculate_ma(combined_df['close'], 20)
            
            # MACD
            macd, signal, hist = TechnicalAnalyzer.calculate_macd(combined_df['close'])
            
            # RSI
            rsi = TechnicalAnalyzer.calculate_rsi(combined_df['close'])
            
            # 获取预测部分的最后一个值
            indicators['ma5'] = ma5.iloc[-1]
            indicators['ma10'] = ma10.iloc[-1]
            indicators['ma20'] = ma20.iloc[-1]
            
            indicators['macd'] = macd.iloc[-1]
            indicators['macd_signal'] = signal.iloc[-1]
            indicators['macd_hist'] = hist.iloc[-1]
            
            indicators['rsi'] = rsi.iloc[-1]
            
            # 添加趋势判断
            indicators['trend_ma'] = 'bullish' if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] else 'bearish'
            indicators['trend_macd'] = 'bullish' if macd.iloc[-1] > signal.iloc[-1] else 'bearish'
            indicators['trend_rsi'] = 'overbought' if rsi.iloc[-1] > 70 else ('oversold' if rsi.iloc[-1] < 30 else 'neutral')
            
        except Exception as e:
            self.logger.warning(f"计算技术指标失败: {e}")
            
        return indicators

    def _calculate_max_drawdown(self, prices):
        """计算最大回撤"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def run_prediction_pipeline(self, historical_df, x_df, x_timestamp, y_timestamp,
                               is_future_forecast, symbol, pred_len,
                               T=0.5, top_p=0.5, sample_count=1, plot_lookback=1500,
                               enable_advanced_preprocessing=False, price_normalization="none",
                               trend_adjustment=False, volatility_filter=False, config=None):
        """
        运行完整的预测流程

        Args:
            config: 配置字典，包含图表显示等设置
        """
        mode_name = "未来预测" if is_future_forecast else "历史回测"
        logger.info(f"🚀 开始 {symbol} 的{mode_name}流程...")

        try:
            # === 数据验证和预处理 ===
            logger.info("📊 数据验证和预处理...")

            # 验证历史数据
            is_valid, error_msg = self.validate_data(historical_df, "historical_data")
            if not is_valid:
                logger.error(f"历史数据验证失败: {error_msg}")
                return None

            # 验证输入数据（预测输入的最小点数按实际lookback长度放宽）
            input_df = x_df.copy()
            input_df[self.data_config['timestamp_column']] = x_timestamp
            input_min_points = max(10, len(x_df))
            is_valid, error_msg = self.validate_data(input_df, "input_data",
                                                      min_points_override=input_min_points)
            if not is_valid:
                logger.error(f"输入数据验证失败: {error_msg}")
                return None

            # 【关键修复】保存原始历史数据用于后续分析和绘图
            original_historical_df = historical_df.copy()

            # 预处理历史数据
            historical_df = self.preprocess_data(
                historical_df,
                enable_advanced=enable_advanced_preprocessing,
                normalization=price_normalization,
                trend_adjustment=trend_adjustment,
                volatility_filter=volatility_filter
            )
            x_df = self.preprocess_data(
                input_df,
                enable_advanced=enable_advanced_preprocessing,
                normalization=price_normalization,
                trend_adjustment=trend_adjustment,
                volatility_filter=volatility_filter,
                min_points_override=input_min_points
            )
            # 仅保留在输入中存在的必要列
            valid_cols = [c for c in REQUIRED_COLUMNS if c in x_df.columns]
            x_df = x_df[valid_cols]

            # 确保 y_timestamp 是正确的类型
            y_timestamp_series = pd.Series(pd.to_datetime(y_timestamp))

            # === 1. 进行预测 ===
            logger.info("🔮 开始模型推理...")
            pred_df = self.predict(x_df, x_timestamp, y_timestamp_series, pred_len, T, top_p, sample_count)
            if pred_df is None:
                logger.error("预测失败，返回None")
                return None

            # === 2. 确保 pred_df 的索引是 DatetimeIndex ===
            pred_df.index = pd.to_datetime(pred_df.index)

            # === 3. 分析预测结果 ===
            logger.info("📈 分析预测结果...")
            # 【关键修复】使用原始历史数据进行分析
            analysis = self.analyze_prediction(original_historical_df, pred_df, symbol, is_future_forecast)
            if analysis is None:
                logger.error("预测结果分析失败")
                return None

            # 【新增强化】为本次预测运行生成唯一的目录结构
            run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_dir = "future_forecast" if is_future_forecast else "backtest"
            run_dir = os.path.join(self.results_dir, symbol, mode_dir, run_timestamp)
            os.makedirs(run_dir, exist_ok=True)
            logger.info(f"本次运行独立目录: {run_dir}")

            # === 4. 绘制图表 ===
            logger.info("📊 生成可视化图表...")
            plot_path = self.plot_prediction(
                original_historical_df, pred_df, symbol, is_future_forecast,
                plot_lookback=plot_lookback,
                enable_focus_mode=config.get('enable_focus_mode', False) if config else False,
                plot_lookback_days=config.get('plot_lookback_days') if config else None,
                prediction_highlight=config.get('prediction_highlight', True) if config else True,
                raw_historical_df=original_historical_df,  # 传入原始历史数据用于正确显示价格尺度
                run_dir=run_dir  # 传入本次运行专属目录
            )
            if plot_path is None:
                logger.error("❌ 图表生成失败，plot_path为None")
            else:
                logger.info(f"✅ 图表生成成功: {plot_path}")

            # === 5. 保存结果 ===
            logger.info("💾 保存预测结果...")
            metadata = {
                'analysis': analysis,
                'plot_path': plot_path,
                'parameters': {
                    'pred_len': pred_len,
                    'T': T,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'is_future_forecast': is_future_forecast
                },
                'data_quality': {
                    'input_points': len(x_df),
                    'prediction_points': len(pred_df),
                    'processing_timestamp': datetime.now().isoformat()
                }
            }

            csv_path = self.save_prediction_results(pred_df, symbol, metadata, is_future_forecast, run_dir=run_dir)

            # === 6. 返回完整结果 ===
            results = {
                'symbol': symbol,
                'prediction': pred_df,
                'analysis': analysis,
                'files': {
                    'csv_path': csv_path,
                    'plot_path': plot_path
                },
                'metadata': metadata
            }

            logger.info(f"🎉 {mode_name}流程全部完成！")
            logger.info(f"📈 {mode_name}图表已保存至: {plot_path}")
            logger.info(f"📄 {mode_name}数据已保存至: {csv_path}")

            return results

        except Exception as e:
            logger.error(f"预测流程执行失败: {str(e)}")
            logger.error("详细错误信息:", exc_info=True)
            return None

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Kronos 股票预测器 - 独立运行版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python stock_predictor.py                                    # 使用默认设置运行示例
  python stock_predictor.py --device cpu                      # 使用 CPU 运行
  python stock_predictor.py --device cuda:0                   # 使用 GPU 运行
  python stock_predictor.py --data-path /path/to/data.csv --symbol 000001  # 使用自定义数据

参数说明:
  device: 计算设备选择
    - auto: 自动检测 (默认，推荐)
    - cpu: 使用 CPU
    - cuda:0: 使用第一个 CUDA GPU
    - mps: 使用 Apple Silicon GPU

  data-path: 自定义数据文件路径 (可选)
    如果不指定，将使用项目中的示例数据

  symbol: 股票代码 (可选)
    与 data-path 配合使用，默认 '600977'
        """
    )

    parser.add_argument(
        "--device", "-d",
        default="auto",
        choices=["auto", "cpu", "cuda", "cuda:0", "mps"],
        help="计算设备 (默认: auto)"
    )

    parser.add_argument(
        "--data-path",
        help="自定义数据文件路径 (CSV格式，包含OHLCV数据)"
    )

    parser.add_argument(
        "--symbol", "-s",
        default="600977",
        help="股票代码 (默认: 600977)"
    )

    parser.add_argument(
        "--lookback", "-l",
        type=int,
        default=1500,
        help="历史数据点数量 (默认: 1500)"
    )

    parser.add_argument(
        "--pred-len", "-p",
        type=int,
        default=96,
        help="预测数据点数量 (默认: 96)"
    )

    parser.add_argument(
        "--future-forecast",
        action="store_true",
        help="未来预测模式 (默认: False，回测模式)"
    )

    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    print("="*60)
    print("🎯 Kronos 股票预测器 - 独立运行版本")
    print("="*60)
    print(f"📋 使用参数: device={args.device}, symbol={args.symbol}")

    # 1. 创建预测器
    try:
        print(f"\n🚀 正在初始化预测器 (device: {args.device})...")
        predictor = StockPredictor(device=args.device)
        print("✅ 预测器初始化成功！")
    except Exception as e:
        print(f"❌ 预测器初始化失败: {e}")
        print("\n🔧 可能的原因:")
        print("  1. 缺少依赖包，请运行: pip install -r requirements.txt")
        print("  2. Kronos 模型下载失败，请检查网络连接")
        print(f"  3. 设备 '{args.device}' 不可用，尝试使用 'cpu'")
        print("\n💡 建议:")
        print("  python stock_predictor.py --device cpu")
        return

    # 2. 确定数据文件路径
    if args.data_path:
        data_path = args.data_path
        symbol = args.symbol
        print(f"\n📂 使用自定义数据文件: {data_path}")
        print(f"📈 股票代码: {symbol}")
    else:
        # 使用示例数据
        data_path = os.path.join("examples", "data", "XSHG_5min_600977.csv")
        symbol = "600977"
        print(f"\n📂 使用示例数据文件: {data_path}")
        print(f"📈 股票代码: {symbol}")

    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        print("\n💡 建议解决方案:")
        if args.data_path:
            print("  1. 检查文件路径是否正确")
            print("  2. 确保文件包含必要的列: timestamps, open, high, low, close, volume, amount")
        else:
            print("  1. 运行数据获取脚本获取股票数据:")
            print("     python my_stock_predictor/run_my_prediction.py")
        return

    # 3. 加载数据
    print(f"\n📖 正在加载数据...")
    df = predictor.load_data(data_path)
    if df is None:
        print("❌ 数据加载失败")
        return

    # 4. 准备预测数据
    lookback = args.lookback
    pred_len = args.pred_len
    is_future_forecast = args.future_forecast

    print(f"\n⚙️ 预测参数:")
    print(f"   - 历史数据点: {lookback}")
    print(f"   - 预测长度: {pred_len}")
    print(f"   - 预测模式: {'未来预测' if is_future_forecast else '历史回测'}")

    # 检查数据是否足够
    if len(df) < lookback + pred_len:
        print(f"⚠️ 警告: 数据点不足 (需要 {lookback + pred_len}, 实际 {len(df)})")
        # 自动调整参数
        available_points = len(df)
        lookback = min(lookback, available_points // 2)
        pred_len = min(pred_len, available_points - lookback)
        print(f"🔧 自动调整参数: lookback={lookback}, pred_len={pred_len}")

    x_df, x_timestamp, y_timestamp = predictor.prepare_prediction_data(df, lookback, pred_len)

    # 5. 运行预测流程
    print("\n🔮 开始预测流程...")
    start_time = datetime.now()

    results = predictor.run_prediction_pipeline(
        historical_df=df,
        x_df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        is_future_forecast=is_future_forecast,
        symbol=symbol,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        plot_lookback=lookback
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    if results:
        print(f"\n🎉 预测完成！用时 {duration:.1f} 秒")
        print(f"📁 结果已保存到 prediction_results/{symbol}/ 目录")
        print("   - 查看生成的图表和数据文件")
        # 打印结果概览
        analysis = results.get('analysis', {})
        if analysis:
            price_change_pct = analysis.get('price_analysis', {}).get('price_change_percentage', 0)
            trend = analysis.get('price_analysis', {}).get('trend', '未知')
            print("\n📊 预测概览:")
            print(f"   - 价格变化: {price_change_pct:.2f}%")
            print(f"   - 趋势: {trend}")
    else:
        print(f"\n❌ 预测流程失败，用时 {duration:.1f} 秒")

if __name__ == "__main__":
    main()
