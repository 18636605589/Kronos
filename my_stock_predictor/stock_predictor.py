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
from datetime import datetime, timedelta
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
    
    def __init__(self, device="cpu", max_context=512, results_dir="my_stock_predictor/prediction_results"):
        """
        初始化预测器
        
        Args:
            device (str): 计算设备，'cpu' 或 'cuda:0'
            max_context (int): 最大上下文长度
            results_dir (str): 结果保存目录
        """
        self.device = device
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
    
    def load_model(self):
        """加载Kronos模型"""
        try:
            print("正在加载Kronos模型...")
            
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
    
    def prepare_prediction_data(self, df, lookback=400, pred_len=120):
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
    
    def plot_prediction(self, historical_df, pred_df, symbol, is_future_forecast=False, save_plot=True):
        """
        绘制预测结果
        """
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 准备绘图数据
            start_pred_time = pred_df.index.min()
            historical_plot_df = historical_df[historical_df['timestamps'] < start_pred_time].tail(400)
            
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
                               T=1.0, top_p=0.9, sample_count=1):
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
        plot_path = self.plot_prediction(historical_df, pred_df, symbol, is_future_forecast)
        
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

def main():
    """主函数示例"""
    # 创建预测器
    predictor = StockPredictor(device="cpu")
    
    # 示例：使用示例数据进行预测
    print("="*60)
    print("示例：使用Kronos示例数据进行预测")
    print("="*60)
    
    # 使用项目中的示例数据
    example_data_path = os.path.join("examples", "data", "XSHG_5min_600977.csv")
    
    if os.path.exists(example_data_path):
        # 加载数据
        df = predictor.load_data(example_data_path)
        if df is None:
            print("无法加载示例数据，请检查文件路径。")
            return

        # 准备预测数据
        lookback = 400
        pred_len = 120
        x_df, x_timestamp, y_timestamp = predictor.prepare_prediction_data(df, lookback, pred_len)

        # 运行预测流程
        results = predictor.run_prediction_pipeline(
            historical_df=df, # 传入完整的df用于绘图和分析
            x_df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            is_future_forecast=False, # 示例数据是历史数据，不是未来预测
            symbol="600977",
            pred_len=pred_len,
            T=1.0,
            top_p=0.9,
            sample_count=1
        )
        
        if results:
            print("\n🎉 预测完成！结果已保存到prediction_results目录")
    else:
        print(f"示例数据文件不存在: {example_data_path}")
        print("请先运行数据获取模块获取股票数据")

if __name__ == "__main__":
    main()
