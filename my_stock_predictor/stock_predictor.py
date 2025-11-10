#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票预测分析模块
基于Kronos模型进行股票预测并保存结果
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import torch
import warnings
warnings.filterwarnings('ignore')

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
    
    def __init__(self, device="auto", max_context=512, results_dir="my_stock_predictor/prediction_results"):
        """
        初始化预测器
        
        Args:
            device (str): 计算设备，可为 'cpu'、'cuda:0' 或 'auto'
            max_context (int): 最大上下文长度
            results_dir (str): 结果保存目录
        """
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.max_context = max_context
        self.results_dir = results_dir
        self.model = None
        self.tokenizer = None
        self.predictor = None
        
        # 确保结果目录存在
        self.ensure_results_dir()
        
        # 初始化模型
        self.load_model()
    
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
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

            # 创建预测器
            self.predictor = KronosPredictor(
                self.model,
                self.tokenizer,
                device=self.device,
                max_context=self.max_context
            )

            print("模型加载成功！")

        except Exception as e:
            print(f"模型加载失败: {e}")
            print("\n🔧 解决方案:")
            print("  1. 检查网络连接是否正常")
            print("  2. 尝试使用代理: export HTTPS_PROXY=http://your-proxy:port")
            print("  3. 或者下载模型到本地后设置 local_files_only=True")
            print("  4. 如果是SSL问题，可以尝试: pip install --upgrade requests urllib3")
            raise
    
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
    
    def predict(self, x_df, x_timestamp, y_timestamp, pred_len, T=1.0, top_p=0.9, sample_count=1):
        """
        进行预测
        
        Args:
            x_df (pd.DataFrame): 输入特征
            x_timestamp (pd.Series): 输入时间戳
            y_timestamp (pd.Series): 预测时间戳
            pred_len (int): 预测长度
            T (float): 采样温度
            top_p (float): 核采样概率
            sample_count (int): 采样次数
            
        Returns:
            pd.DataFrame: 预测结果
        """
        try:
            print("正在进行预测...")
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=T,
                top_p=top_p,
                sample_count=sample_count,
                verbose=True
            )
            
            print("预测完成！")
            return pred_df
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def plot_prediction(self, historical_df, pred_df, symbol, is_future_forecast=False, save_plot=True, plot_lookback=1500):
        """
        绘制预测结果
        """
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 准备绘图数据
            start_pred_time = pred_df.index.min()
            historical_plot_df = historical_df[historical_df['timestamps'] < start_pred_time].tail(plot_lookback)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
            
            # --- 价格图 ---
            ax1.plot(historical_plot_df['timestamps'], historical_plot_df['close'], label='历史价格', color='blue', linewidth=1.5)
            ax1.plot(pred_df.index, pred_df['close'], label='预测价格', color='red', linewidth=1.5, linestyle='--')
            
            # 在回测模式下，添加真实价格曲线
            if not is_future_forecast:
                true_values_df = historical_df[historical_df['timestamps'].isin(pred_df.index)]
                ax1.plot(true_values_df['timestamps'], true_values_df['close'], label='真实价格', color='green', linewidth=1.5, alpha=0.7)
            
            ax1.set_ylabel('价格', fontsize=14)
            ax1.set_title(f'{symbol} 股票预测结果', fontsize=16)
            ax1.legend(loc='upper left', fontsize=12)
            ax1.grid(True, alpha=0.3)
            if not historical_plot_df.empty:
                ax1.axvline(historical_plot_df['timestamps'].iloc[-1], color='gray', linestyle='--', linewidth=1)
            
            # --- 成交量图 ---
            ax2.plot(historical_plot_df['timestamps'], historical_plot_df['volume'], label='历史成交量', color='blue', linewidth=1.5)
            ax2.plot(pred_df.index, pred_df['volume'], label='预测成交量', color='red', linewidth=1.5, linestyle='--')
            
            if not is_future_forecast:
                true_values_df = historical_df[historical_df['timestamps'].isin(pred_df.index)]
                ax2.plot(true_values_df['timestamps'], true_values_df['volume'], label='真实成交量', color='green', linewidth=1.5, alpha=0.7)
            
            ax2.set_ylabel('成交量', fontsize=14)
            ax2.set_xlabel('时间', fontsize=14)
            ax2.legend(loc='upper left', fontsize=12)
            ax2.grid(True, alpha=0.3)
            if not historical_plot_df.empty:
                ax2.axvline(historical_plot_df['timestamps'].iloc[-1], color='gray', linestyle='--', linewidth=1)
            
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            
            # 保存图表
            plot_path = None
            if save_plot:
                # --- 修正: 创建股票专属的结果子文件夹 ---
                symbol_results_dir = os.path.join(self.results_dir, symbol)
                os.makedirs(symbol_results_dir, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_filename = f"{symbol}_prediction_chart_{timestamp}.png"
                plot_path = os.path.join(symbol_results_dir, plot_filename)
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"图表已保存: {plot_path}")
            
            plt.show()
            
            return plot_path
            
        except Exception as e:
            print(f"绘制图表失败: {e}")
            return None
    
    def save_prediction_results(self, pred_df, symbol, metadata=None):
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
            # --- 修正: 创建股票专属的结果子文件夹 ---
            symbol_results_dir = os.path.join(self.results_dir, symbol)
            os.makedirs(symbol_results_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存预测数据
            csv_filename = f"{symbol}_prediction_data_{timestamp}.csv"
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
            
            json_filename = f"{symbol}_metadata_{timestamp}.json"
            json_path = os.path.join(symbol_results_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"预测结果已保存至 '{symbol_results_dir}' 目录:")
            print(f"  - 数据文件: {os.path.basename(csv_path)}")
            print(f"  - 元数据文件: {os.path.basename(json_path)}")
            
            return csv_path
            
        except Exception as e:
            print(f"保存结果失败: {e}")
            return None
    
    def analyze_prediction(self, historical_df, pred_df, symbol, is_future_forecast):
        """
        分析预测结果
        """
        try:
            # 获取历史数据的最后一个点用于比较
            last_historical_point = historical_df.iloc[-1]
            last_close = last_historical_point['close']
            
            # 预测数据的统计信息
            pred_close_stats = {
                'mean': pred_df['close'].mean(),
                'std': pred_df['close'].std(),
                'min': pred_df['close'].min(),
                'max': pred_df['close'].max(),
                'trend': '上涨' if pred_df['close'].iloc[-1] > pred_df['close'].iloc[0] else '下跌'
            }
            
            pred_volume_stats = {
                'mean': pred_df['volume'].mean(),
                'std': pred_df['volume'].std(),
                'min': pred_df['volume'].min(),
                'max': pred_df['volume'].max()
            }
            
            # 价格变化分析
            # 如果是未来预测，与最后一个历史点比较
            # 如果是回测，与预测开始前的那个点比较
            if is_future_forecast:
                comparison_close = last_close
            else:
                # 找到预测开始前的最后一个点
                comparison_point = historical_df[historical_df['timestamps'] < pred_df.index.min()]
                if not comparison_point.empty:
                    comparison_close = comparison_point.iloc[-1]['close']
                else:
                    comparison_close = last_close # Fallback

            price_change = pred_df['close'].iloc[-1] - comparison_close
            price_change_pct = (price_change / comparison_close) * 100 if comparison_close != 0 else 0
            
            # 生成分析报告
            analysis = {
                'symbol': symbol,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'historical_last_close': last_close,
                'historical_last_volume': last_historical_point['volume'], # 使用最后一个历史点的成交量
                'prediction_periods': len(pred_df),
                'price_analysis': {
                    'predicted_last_close': pred_df['close'].iloc[-1],
                    'price_change': price_change,
                    'price_change_percentage': price_change_pct,
                    'trend': pred_close_stats['trend'],
                    'volatility': pred_close_stats['std']
                },
                'volume_analysis': {
                    'predicted_avg_volume': pred_volume_stats['mean'],
                    'volume_trend': '增加' if pred_volume_stats['mean'] > last_historical_point['volume'] else '减少'
                },
                'statistics': {
                    'close_stats': pred_close_stats,
                    'volume_stats': pred_volume_stats
                }
            }
            
            # 打印分析结果
            print("\n" + "="*60)
            print(f"📊 {symbol} 预测分析报告")
            print("="*60)
            print(f"📈 价格趋势: {analysis['price_analysis']['trend']}")
            print(f"💰 预测价格变化: {price_change:.4f} ({price_change_pct:.2f}%)")
            print(f"📊 价格波动性: {pred_close_stats['std']:.4f}")
            print(f"📈 成交量趋势: {analysis['volume_analysis']['volume_trend']}")
            print(f"📊 预测数据点: {len(pred_df)}")
            print("="*60)
            
            return analysis
            
        except Exception as e:
            print(f"分析预测结果失败: {e}")
            return None
    
    def run_prediction_pipeline(self, historical_df, x_df, x_timestamp, y_timestamp,
                               is_future_forecast, symbol, pred_len,
                               T=1.0, top_p=0.9, sample_count=1, plot_lookback=1500):
        """
        运行完整的预测流程
        """
        print(f"🚀 开始 {symbol} 的预测流程...")
        
        # --- 核心修正: 确保 y_timestamp 始终是 Series ---
        # 原始模型需要 Series 类型的时间戳输入
        if isinstance(y_timestamp, pd.DatetimeIndex):
            y_timestamp_series = pd.Series(y_timestamp, index=y_timestamp)
        else:
            y_timestamp_series = y_timestamp

        # 1. 进行预测
        pred_df = self.predict(x_df, x_timestamp, y_timestamp_series, pred_len, T, top_p, sample_count)
        if pred_df is None:
            return None
        
        # 2. 确保 pred_df 的索引是 DatetimeIndex
        pred_df.index = pd.to_datetime(pred_df.index)
        
        # 3. 分析预测结果
        analysis = self.analyze_prediction(historical_df, pred_df, symbol, is_future_forecast)
        
        # 4. 绘制图表
        plot_path = self.plot_prediction(historical_df, pred_df, symbol, is_future_forecast, plot_lookback=plot_lookback)
        
        # 5. 保存结果
        metadata = {
            'analysis': analysis,
            'plot_path': plot_path,
            'parameters': {
                'pred_len': pred_len,
                'T': T,
                'top_p': top_p,
                'sample_count': sample_count,
                'is_future_forecast': is_future_forecast
            }
        }
        
        csv_path = self.save_prediction_results(pred_df, symbol, metadata)
        
        # 6. 返回完整结果
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
        
        print(f"✅ {symbol} 预测流程完成！")
        return results

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
