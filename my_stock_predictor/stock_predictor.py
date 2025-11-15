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
except ImportError:
    print("错误: 无法导入Kronos模型，请确保模型文件存在")
    sys.exit(1)

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

        if normalized == "auto":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                print("✅ 检测到 Apple Silicon MPS，自动使用 'mps' 设备。")
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
                return "mps"
            print("⚠️ 请求使用 MPS (Apple Silicon)，但当前环境不支持，已自动回退到 CPU。")
            return "cpu"

        return device
    
    def validate_data(self, df: pd.DataFrame, context: str = "general") -> Tuple[bool, str]:
        """
        验证数据质量和完整性

        Args:
            df: 输入数据框
            context: 验证上下文描述

        Returns:
            (is_valid, error_message)
        """
        try:
            # 1. 检查必要列是否存在
            missing_cols = set(self.data_config['required_columns']) - set(df.columns)
            if missing_cols:
                return False, f"缺少必要列: {missing_cols}"

            # 2. 检查时间戳列
            if self.data_config['timestamp_column'] not in df.columns:
                return False, f"缺少时间戳列: {self.data_config['timestamp_column']}"

            # 3. 检查数据量是否足够
            if len(df) < self.data_config['min_data_points']:
                return False, f"数据点过少: {len(df)} < {self.data_config['min_data_points']}"

            # 4. 检查NaN值比例
            nan_ratio = df[self.data_config['required_columns']].isnull().sum().sum() / (len(df) * len(self.data_config['required_columns']))
            if nan_ratio > self.data_config['max_nan_ratio']:
                return False, f"NaN值比例过高: {nan_ratio:.2%} > {self.data_config['max_nan_ratio']:.2%}"

            # 5. 检查价格数据合理性
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    return False, f"发现非正价格值在列 {col}"

                # 检查high >= max(open, close), low <= min(open, close)
                if col == 'high':
                    invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
                    if invalid_high.any():
                        return False, f"发现high价格低于open或close"
                elif col == 'low':
                    invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
                    if invalid_low.any():
                        return False, f"发现low价格高于open或close"

            # 6. 检查时间戳排序
            if not df[self.data_config['timestamp_column']].is_monotonic_increasing:
                return False, "时间戳不是单调递增的"

            return True, "数据验证通过"

        except Exception as e:
            return False, f"数据验证过程中发生错误: {str(e)}"

    def preprocess_data(self, df: pd.DataFrame, detect_outliers: bool = True) -> pd.DataFrame:
        """
        数据预处理和清理

        Args:
            df: 原始数据框
            detect_outliers: 是否检测和处理异常值

        Returns:
            处理后的数据框
        """
        try:
            logger.info("开始数据预处理...")

            # 首先验证输入数据
            is_valid, error_msg = self.validate_data(df, "preprocessing_input")
            if not is_valid:
                raise ValueError(f"输入数据验证失败: {error_msg}")

            processed_df = df.copy()

            # 1. 处理缺失值
            numeric_cols = self.data_config['required_columns']
            for col in numeric_cols:
                if processed_df[col].isnull().any():
                    # 使用前向填充，然后后向填充
                    processed_df[col] = processed_df[col].ffill().bfill()
                    # 如果仍有NaN，使用中位数填充
                    if processed_df[col].isnull().any():
                        median_val = processed_df[col].median()
                        processed_df[col] = processed_df[col].fillna(median_val)
                        logger.warning(f"列 {col} 使用中位数 {median_val:.2f} 填充剩余NaN值")

            # 2. 异常值检测和处理
            if detect_outliers:
                processed_df = self._handle_outliers(processed_df)

            # 3. 添加数据平滑处理（优化措施）
            processed_df = self._smooth_price_data(processed_df)

            # 4. 确保数据类型正确
            processed_df[self.data_config['timestamp_column']] = pd.to_datetime(processed_df[self.data_config['timestamp_column']])
            for col in numeric_cols:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

            # 5. 排序数据
            processed_df = processed_df.sort_values(self.data_config['timestamp_column']).reset_index(drop=True)

            # 6. 最终验证处理后的数据
            is_valid, error_msg = self.validate_data(processed_df, "preprocessing_output")
            if not is_valid:
                logger.warning(f"预处理后数据存在问题: {error_msg}，但继续执行")

            logger.info(f"数据预处理完成，处理后数据量: {len(processed_df)}")
            return processed_df

        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise

    def _smooth_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对价格数据进行平滑处理，减少噪声，提高预测稳定性
        """
        smoothed_df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']

        for col in price_cols:
            if col in smoothed_df.columns:
                # 使用指数移动平均进行轻微平滑（alpha=0.1，避免数据泄露）
                smoothed_df[col] = smoothed_df[col].ewm(alpha=0.1, adjust=True).mean()

        logger.info("已应用数据平滑处理以提高预测稳定性")
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

            # IQR异常值检测：低于Q1-1.5*IQR或高于Q3+1.5*IQR的值
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

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
        """加载Kronos模型"""
        try:
            print(f"正在加载Kronos模型... (device: {self.device})")

            # 设置环境变量解决SSL问题
            import os
            os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
            os.environ['REQUESTS_CA_BUNDLE'] = ''
            os.environ['SSL_CERT_FILE'] = ''

            # 加载分词器和模型
            self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-base")

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
            if "SSL" in str(e).upper():
                print("  1. SSL证书问题，尝试: pip install --upgrade requests urllib3")
                print("  2. 或设置环境变量跳过SSL验证: export HF_HUB_DISABLE_SSL_VERIFICATION=1")
            elif "timeout" in str(e).lower():
                print("  1. 网络超时，检查网络连接或使用代理")
                print("  2. 或下载模型到本地后使用离线模式")
            elif "disk" in str(e).lower():
                print("  1. 磁盘空间不足，清理磁盘空间")
                print("  2. 或设置HF_HOME到其他目录")
            else:
                print("  1. 检查HuggingFace token是否正确设置")
                print("  2. 尝试重新安装transformers库")
                print("  3. 检查是否有足够的内存和磁盘空间")
            raise RuntimeError(f"Kronos模型加载失败: {e}")
    
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
            
            # 删除无效数据
            df = df.dropna()
            
            # 按时间排序
            df = df.sort_values('timestamps').reset_index(drop=True)
            
            print(f"成功加载 {len(df)} 条数据")
            print(f"时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
            
            return df
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def prepare_prediction_data(self, df, lookback=1500, pred_len=96):
        """
        准备预测数据
        
        Args:
            df (pd.DataFrame): 股票数据
            lookback (int): 回看窗口大小
            pred_len (int): 预测长度
            
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
        x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
        x_timestamp = df.loc[:lookback-1, 'timestamps']
        y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
        
        return x_df, x_timestamp, y_timestamp
    
    def predict(self, x_df, x_timestamp, y_timestamp, pred_len, T=0.3, top_p=0.8, sample_count=5):
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
            
        except MemoryError:
            self.logger.error("内存不足，无法完成预测")
            self.optimize_memory_usage()
            return None
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
            self.logger.info("自适应参数调优已禁用，使用用户指定的参数")
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

        # 2. 确保OHLC关系合理
        processed_df['high'] = processed_df[['open', 'close', 'high']].max(axis=1)
        processed_df['low'] = processed_df[['open', 'close', 'low']].min(axis=1)

        # 3. 确保成交量不为负数
        if 'volume' in processed_df.columns:
            processed_df['volume'] = processed_df['volume'].clip(lower=0)
        if 'amount' in processed_df.columns:
            processed_df['amount'] = processed_df['amount'].clip(lower=0)

        return processed_df

    def _calculate_smart_xticks(self, timestamps, max_ticks=15, is_future_forecast=False):
        """
        计算智能时间轴刻度，越接近当前时间显示越详细

        Args:
            timestamps: 所有时间戳
            max_ticks: 最大刻度数量

        Returns:
            智能刻度位置和标签
        """
        if timestamps.empty:
            return [], []

        timestamps = pd.to_datetime(timestamps)
        start_time = timestamps.min()
        end_time = timestamps.max()
        total_duration = end_time - start_time

        # 计算关键时间点
        current_time = pd.Timestamp.now()
        pred_start = timestamps[timestamps >= start_time].min() if len(timestamps) > 0 else end_time

        # 根据时间跨度确定刻度密度
        total_hours = total_duration.total_seconds() / 3600

        if total_hours <= 24:  # 1天内
            # 每小时一个刻度，重点区域每15分钟
            base_freq = 'H'
            dense_freq = '15min'
        elif total_hours <= 168:  # 1周内
            # 每天一个刻度，重点区域每小时
            base_freq = 'D'
            dense_freq = 'H'
        elif total_hours <= 720:  # 1月内
            # 每周一个刻度，重点区域每天
            base_freq = 'W'
            dense_freq = 'D'
        else:  # 更长时间
            # 每月一个刻度，重点区域每周
            base_freq = 'M'
            dense_freq = 'W'

        # 生成基础刻度
        base_ticks = pd.date_range(start=start_time, end=end_time, freq=base_freq)

        # 在关键区域添加密集刻度
        dense_ticks = []

        # 预测开始时间前后的密集刻度
        pred_dense_start = pred_start - pd.Timedelta(hours=min(total_hours * 0.1, 24))
        pred_dense_end = pred_start + pd.Timedelta(hours=min(total_hours * 0.1, 24))
        if pred_dense_start < end_time and pred_dense_end > start_time:
            pred_dense = pd.date_range(
                start=max(pred_dense_start, start_time),
                end=min(pred_dense_end, end_time),
                freq=dense_freq
            )
            dense_ticks.extend(pred_dense)

        # 当前时间前后的密集刻度（如果是未来预测）
        if is_future_forecast and abs(current_time - end_time) < pd.Timedelta(days=30):
            current_dense_start = current_time - pd.Timedelta(hours=min(total_hours * 0.15, 48))
            current_dense_end = min(current_time + pd.Timedelta(hours=min(total_hours * 0.15, 48)), end_time)
            if current_dense_start < end_time and current_dense_end > start_time:
                current_dense = pd.date_range(
                    start=max(current_dense_start, start_time),
                    end=current_dense_end,
                    freq=dense_freq
                )
                dense_ticks.extend(current_dense)

        # 合并所有刻度并去重
        all_ticks = sorted(set(base_ticks).union(set(dense_ticks)))
        all_ticks = [t for t in all_ticks if start_time <= t <= end_time]

        # 限制刻度数量
        if len(all_ticks) > max_ticks:
            # 优先保留关键区域的刻度
            key_ticks = []
            other_ticks = []

            for tick in all_ticks:
                if (abs(tick - pred_start) < pd.Timedelta(hours=24) or
                    (is_future_forecast and abs(tick - current_time) < pd.Timedelta(hours=24))):
                    key_ticks.append(tick)
                else:
                    other_ticks.append(tick)

            # 保留所有关键刻度，然后均匀采样其他刻度
            remaining_slots = max_ticks - len(key_ticks)
            if remaining_slots > 0 and other_ticks:
                step = max(1, len(other_ticks) // remaining_slots)
                sampled_other = other_ticks[::step][:remaining_slots]
                all_ticks = sorted(set(key_ticks + sampled_other))
            else:
                all_ticks = sorted(key_ticks)

        # 生成刻度标签
        tick_labels = []
        for tick in all_ticks:
            if total_hours <= 24:  # 1天内显示时分
                tick_labels.append(tick.strftime('%m-%d %H:%M'))
            elif total_hours <= 168:  # 1周内显示日期和小时
                tick_labels.append(tick.strftime('%m-%d %H:00'))
            else:  # 更长时间显示日期
                tick_labels.append(tick.strftime('%m-%d'))

        return all_ticks, tick_labels
    
    def plot_prediction(self, historical_df, pred_df, symbol, is_future_forecast=False, save_plot=True, plot_lookback=1500):
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
            ax1.plot(historical_plot_df['timestamps'], historical_plot_df['close'], label='历史价格', color='blue', linewidth=1.5)
            ax1.plot(pred_df.index, pred_df['close'], label='预测价格', color='red', linewidth=2, linestyle='--')
            
            # 在回测模式下，添加真实价格曲线
            if not is_future_forecast:
                true_values_df = historical_df[historical_df['timestamps'].isin(pred_df.index)]
                ax1.plot(true_values_df['timestamps'], true_values_df['close'], label='真实价格', color='green', linewidth=2, alpha=0.8)

            # 添加关键时间区域标记
            current_time = pd.Timestamp.now()

            # 标记预测开始时间
            pred_start_time = pred_df.index.min()
            ax1.axvline(pred_start_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
            ax1.text(pred_start_time, ax1.get_ylim()[1]*0.95, '预测开始',
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7))

            # 标记当前时间（如果是未来预测且接近当前时间）
            if is_future_forecast and abs(current_time - all_timestamps.max()) < pd.Timedelta(days=7):
                ax1.axvline(current_time, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)
                ax1.text(current_time, ax1.get_ylim()[1]*0.90, '当前时间',
                        ha='center', va='top', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='purple', alpha=0.7))

            # 标记历史数据结束时间
            if not historical_plot_df.empty:
                hist_end_time = historical_plot_df['timestamps'].iloc[-1]
                ax1.axvline(hist_end_time, color='gray', linestyle='--', linewidth=1, alpha=0.8)
                ax1.text(hist_end_time, ax1.get_ylim()[0]*1.05, '历史结束',
                        ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='gray', alpha=0.6))
            
            ax1.set_ylabel('价格', fontsize=14)
            mode_name = '未来预测' if is_future_forecast else '历史回测'
            mode_color = 'orange' if is_future_forecast else 'blue'
            mode_desc = '预测未来趋势' if is_future_forecast else '验证历史表现'

            ax1.set_title(f'{symbol} 股票{mode_name}结果 - 智能时间轴', fontsize=16)

            # 在图表左上角添加模式标识框
            ax1.text(0.02, 0.98, f'{mode_name}\n{mode_desc}',
                    transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=mode_color, alpha=0.8),
                    color='white', fontweight='bold')

            ax1.legend(loc='upper left', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            # --- 成交量图 ---
            ax2.plot(historical_plot_df['timestamps'], historical_plot_df['volume'], label='历史成交量', color='blue', linewidth=1.5)
            ax2.plot(pred_df.index, pred_df['volume'], label='预测成交量', color='red', linewidth=2, linestyle='--')
            
            if not is_future_forecast:
                true_values_df = historical_df[historical_df['timestamps'].isin(pred_df.index)]
                ax2.plot(true_values_df['timestamps'], true_values_df['volume'], label='真实成交量', color='green', linewidth=2, alpha=0.8)

            # 在成交量图上也添加关键时间标记
            ax2.axvline(pred_start_time, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

            if is_future_forecast and abs(current_time - all_timestamps.max()) < pd.Timedelta(days=7):
                ax2.axvline(current_time, color='purple', linestyle=':', linewidth=1.5, alpha=0.7)

            if not historical_plot_df.empty:
                hist_end_time = historical_plot_df['timestamps'].iloc[-1]
                ax2.axvline(hist_end_time, color='gray', linestyle='--', linewidth=1, alpha=0.8)
            
            ax2.set_ylabel('成交量', fontsize=14)
            ax2.set_xlabel('时间', fontsize=14)
            ax2.legend(loc='upper left', fontsize=12)
            ax2.grid(True, alpha=0.3)

            # 设置智能时间轴刻度
            smart_ticks, tick_labels = self._calculate_smart_xticks(all_timestamps, max_ticks=20, is_future_forecast=is_future_forecast)
            if smart_ticks:
                ax2.set_xticks(smart_ticks)
                ax2.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=10)

            # 添加局部放大视图（如果时间跨度较大且预测区域较短）
            total_duration = all_timestamps.max() - all_timestamps.min()
            pred_duration = pred_df.index.max() - pred_df.index.min()

            if total_duration > pd.Timedelta(days=30) and pred_duration < total_duration * 0.3:
                # 创建局部放大子图
                # 计算预测区域的扩展范围（前后各延长10%）
                pred_range = pred_df.index.max() - pred_df.index.min()
                zoom_margin = pred_range * 0.2

                zoom_start = max(all_timestamps.min(), pred_df.index.min() - zoom_margin)
                zoom_end = min(all_timestamps.max(), pred_df.index.max() + zoom_margin)

                # 创建子图位置
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_zoom = inset_axes(ax1, width="35%", height="25%", loc='upper right',
                                    bbox_to_anchor=(0.02, 0.02, 0.96, 0.96),
                                    bbox_transform=ax1.transAxes)

                # 在子图中绘制局部放大的价格数据
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
                               color='blue', linewidth=1, alpha=0.7)
                if not zoom_pred.empty:
                    ax_zoom.plot(zoom_pred.index, zoom_pred['close'],
                               color='red', linewidth=1.5, linestyle='--', alpha=0.8)

                # 子图样式设置
                ax_zoom.set_title('预测区域放大', fontsize=9, pad=2)
                ax_zoom.tick_params(labelsize=8)
                ax_zoom.grid(True, alpha=0.3)

                # 添加边框
                for spine in ax_zoom.spines.values():
                    spine.set_edgecolor('orange')
                    spine.set_linewidth(1)

            plt.tight_layout()
            
            # 保存图表
            plot_path = None
            if save_plot:
                try:
                    # --- 创建模式区分的结果子文件夹 ---
                    mode_dir = 'future_forecast' if is_future_forecast else 'backtest'
                    symbol_results_dir = os.path.join(self.results_dir, symbol, mode_dir)
                    os.makedirs(symbol_results_dir, exist_ok=True)
                    logger.info(f"创建结果目录: {symbol_results_dir}")

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    mode_prefix = 'forecast' if is_future_forecast else 'backtest'
                    plot_filename = f"{symbol}_{mode_prefix}_chart_{timestamp}.png"
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

            # 在非交互环境中不显示图表，避免阻塞
            # 注意：我们已经设置了Agg后端，所以plt.show()不会实际显示图像
            try:
                plt.show()  # 在Agg后端下这不会显示任何东西，只是为了兼容性
                logger.info("图表显示调用完成")
            except Exception as show_error:
                logger.warning(f"图表显示时出现警告: {str(show_error)}")
            
            logger.info("图表绘制流程完成")
            return plot_path
            
        except Exception as e:
            logger.error(f"绘制图表失败: {str(e)}")
            logger.error("详细错误信息:", exc_info=True)
            return None
    
    def save_prediction_results(self, pred_df, symbol, metadata=None, is_future_forecast=False):
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
            # --- 创建模式区分的结果子文件夹 ---
            mode_dir = 'future_forecast' if is_future_forecast else 'backtest'
            symbol_results_dir = os.path.join(self.results_dir, symbol, mode_dir)
            os.makedirs(symbol_results_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            mode_prefix = 'forecast' if is_future_forecast else 'backtest'

            # 保存预测数据
            csv_filename = f"{symbol}_{mode_prefix}_data_{timestamp}.csv"
            csv_path = os.path.join(symbol_results_dir, csv_filename)
            
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
            
            json_filename = f"{symbol}_{mode_prefix}_metadata_{timestamp}.json"
            json_path = os.path.join(symbol_results_dir, json_filename)
            
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
            technical_analysis = self._calculate_technical_indicators(pred_df)

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

        # 计算预测的变化
        pred_start = pred_df['close'].iloc[0]
        pred_end = pred_df['close'].iloc[-1]

        price_change = pred_end - baseline_close
        price_change_pct = (price_change / baseline_close) * 100 if baseline_close != 0 else 0

        # 趋势强度分析
        pred_trend_strength = abs(pred_end - pred_start) / pred_start if pred_start != 0 else 0

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
        # 计算预测数据的波动性
        pred_returns = pred_df['close'].pct_change().dropna()
        pred_volatility = pred_returns.std() * (252 ** 0.5)  # 年化波动率

        # 计算历史数据的波动性作为对比
        hist_returns = historical_df['close'].pct_change().dropna().tail(100)  # 最近100个点
        hist_volatility = hist_returns.std() * (252 ** 0.5) if len(hist_returns) > 0 else 0

        # 风险指标
        var_95 = np.percentile(pred_returns, 5)  # 95% VaR
        max_drawdown = self._calculate_max_drawdown(pred_df['close'])

        return {
            'pred_volatility': pred_volatility,
            'hist_volatility': hist_volatility,
            'volatility_ratio': pred_volatility / hist_volatility if hist_volatility != 0 else float('inf'),
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': pred_returns.mean() / pred_returns.std() if pred_returns.std() != 0 else 0
        }

    def _evaluate_prediction_quality(self, historical_df, pred_df, is_future_forecast):
        """评估预测质量"""
        quality_scores = {}

        # 1. 平滑性评分 (0-1, 1表示非常平滑)
        returns = pred_df['close'].pct_change().dropna()
        smoothness = 1 / (1 + returns.std())  # 波动率越低，平滑性越高
        quality_scores['smoothness'] = smoothness

        # 2. 合理性评分 (基于历史波动性)
        hist_volatility = historical_df['close'].pct_change().dropna().tail(50).std()
        pred_volatility = returns.std()

        if hist_volatility > 0:
            volatility_ratio = pred_volatility / hist_volatility
            # 理想的波动率比应该在0.5-2.0之间
            if 0.5 <= volatility_ratio <= 2.0:
                reasonableness = 1.0
            else:
                reasonableness = max(0, 1 - abs(np.log(volatility_ratio)))
        else:
            reasonableness = 0.5

        quality_scores['reasonableness'] = reasonableness

        # 3. 连续性评分 (检查是否有异常跳跃)
        jumps = (returns.abs() > 0.1).sum()  # 单步变化超过10%的次数
        continuity = max(0, 1 - jumps / len(returns))
        quality_scores['continuity'] = continuity

        # 4. 整体质量评分
        quality_scores['overall_score'] = np.mean([smoothness, reasonableness, continuity])

        return quality_scores

    def _calculate_technical_indicators(self, pred_df):
        """计算技术指标"""
        indicators = {}

        if len(pred_df) < 14:  # 数据点太少，无法计算技术指标
            return indicators

        close_prices = pred_df['close']

        # 简单移动平均
        indicators['sma_5'] = close_prices.rolling(5).mean().iloc[-1] if len(close_prices) >= 5 else None
        indicators['sma_10'] = close_prices.rolling(10).mean().iloc[-1] if len(close_prices) >= 10 else None

        # RSI (简化版)
        if len(close_prices) >= 14:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            indicators['rsi_14'] = 100 - (100 / (1 + rs)).iloc[-1]
        else:
            indicators['rsi_14'] = None

        return indicators

    def _calculate_max_drawdown(self, prices):
        """计算最大回撤"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def run_prediction_pipeline(self, historical_df, x_df, x_timestamp, y_timestamp,
                               is_future_forecast, symbol, pred_len,
                               T=1.0, top_p=0.9, sample_count=1, plot_lookback=1500):
        """
        运行完整的预测流程
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

            # 验证输入数据
            input_df = x_df.copy()
            input_df[self.data_config['timestamp_column']] = x_timestamp
            is_valid, error_msg = self.validate_data(input_df, "input_data")
            if not is_valid:
                logger.error(f"输入数据验证失败: {error_msg}")
                return None

            # 预处理历史数据
            historical_df = self.preprocess_data(historical_df)
            x_df = self.preprocess_data(input_df)[self.data_config['required_columns']]

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
            analysis = self.analyze_prediction(historical_df, pred_df, symbol, is_future_forecast)
            if analysis is None:
                logger.error("预测结果分析失败")
                return None

            # === 4. 绘制图表 ===
            logger.info("📊 生成可视化图表...")
            plot_path = self.plot_prediction(historical_df, pred_df, symbol, is_future_forecast, plot_lookback=plot_lookback)
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

            csv_path = self.save_prediction_results(pred_df, symbol, metadata, is_future_forecast)

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
