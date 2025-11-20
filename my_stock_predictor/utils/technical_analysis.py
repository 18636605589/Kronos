import pandas as pd
import numpy as np

class TechnicalAnalyzer:
    """
    Technical Analysis Utility Class
    Calculates various technical indicators like MA, MACD, RSI.
    """

    @staticmethod
    def calculate_ma(series: pd.Series, window: int) -> pd.Series:
        """
        Calculate Moving Average (Simple)
        """
        return series.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(series: pd.Series, span: int) -> pd.Series:
        """
        Calculate Exponential Moving Average
        """
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        Returns: macd_line, signal_line, histogram
        """
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Handle division by zero if loss is 0
        rsi = rsi.fillna(100) 
        return rsi

    @staticmethod
    def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
        """
        Calculate Bollinger Bands
        Returns: upper_band, middle_band, lower_band
        """
        middle_band = series.rolling(window=window).mean()
        std_dev = series.rolling(window=window).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3):
        """
        Calculate KDJ Indicator
        Returns: K, D, J
        """
        low_min = df['low'].rolling(window=n).min()
        high_max = df['high'].rolling(window=n).max()
        
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        # Handle NaN
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k, d, j

    @staticmethod
    def add_all_indicators(df: pd.DataFrame, price_col='close') -> pd.DataFrame:
        """
        Add common indicators to the DataFrame
        """
        df_copy = df.copy()
        
        # MA
        df_copy['MA5'] = TechnicalAnalyzer.calculate_ma(df_copy[price_col], 5)
        df_copy['MA10'] = TechnicalAnalyzer.calculate_ma(df_copy[price_col], 10)
        df_copy['MA20'] = TechnicalAnalyzer.calculate_ma(df_copy[price_col], 20)
        
        # MACD
        macd, signal, hist = TechnicalAnalyzer.calculate_macd(df_copy[price_col])
        df_copy['MACD'] = macd
        df_copy['MACD_Signal'] = signal
        df_copy['MACD_Hist'] = hist
        
        # RSI
        df_copy['RSI'] = TechnicalAnalyzer.calculate_rsi(df_copy[price_col])

        # Bollinger Bands
        upper, middle, lower = TechnicalAnalyzer.calculate_bollinger_bands(df_copy[price_col])
        df_copy['BB_Upper'] = upper
        df_copy['BB_Middle'] = middle
        df_copy['BB_Lower'] = lower

        # KDJ
        k, d, j = TechnicalAnalyzer.calculate_kdj(df_copy)
        df_copy['K'] = k
        df_copy['D'] = d
        df_copy['J'] = j
        
        return df_copy

    @staticmethod
    def analyze_market_condition(df: pd.DataFrame, model_prediction_trend: str = None) -> dict:
        """
        Analyze market condition and generate signals
        
        Args:
            df: DataFrame with calculated indicators
            model_prediction_trend: 'up' or 'down' (optional) from model prediction
        """
        if len(df) < 2:
            return {"summary": "数据不足", "signals": [], "warnings": []}
            
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]
        
        signals = []
        warnings = []
        trend_score = 0  # >0 bullish, <0 bearish
        
        # 1. MA Trend Analysis
        if last_row['MA5'] > last_row['MA10'] > last_row['MA20']:
            signals.append("均线多头排列 (强看涨)")
            trend_score += 2
        elif last_row['MA5'] < last_row['MA10'] < last_row['MA20']:
            signals.append("均线空头排列 (强看跌)")
            trend_score -= 2
        elif last_row['close'] > last_row['MA20']:
            signals.append("价格位于20日均线上方 (偏多)")
            trend_score += 1
        else:
            signals.append("价格位于20日均线下方 (偏空)")
            trend_score -= 1
            
        # 2. MACD Analysis
        if last_row['MACD'] > last_row['MACD_Signal']:
            if prev_row['MACD'] <= prev_row['MACD_Signal']:
                signals.append("MACD金叉 (买入信号)")
                trend_score += 2
            else:
                signals.append("MACD多头区域")
                trend_score += 1
        else:
            if prev_row['MACD'] >= prev_row['MACD_Signal']:
                signals.append("MACD死叉 (卖出信号)")
                trend_score -= 2
            else:
                signals.append("MACD空头区域")
                trend_score -= 1
                
        # 3. RSI Analysis & Risk Warning
        if last_row['RSI'] > 80:
            warnings.append(f"RSI超买 ({last_row['RSI']:.1f}) - 短期回调风险大")
            trend_score -= 1 # Overbought is bearish for short term
        elif last_row['RSI'] > 70:
            warnings.append(f"RSI偏高 ({last_row['RSI']:.1f}) - 注意风险")
        elif last_row['RSI'] < 20:
            warnings.append(f"RSI超卖 ({last_row['RSI']:.1f}) - 短期反弹机会")
            trend_score += 1 # Oversold is bullish for short term
        elif last_row['RSI'] < 30:
            signals.append(f"RSI低位 ({last_row['RSI']:.1f})")
            
        # 4. KDJ Analysis
        if last_row['K'] > last_row['D'] and prev_row['K'] <= prev_row['D']:
             signals.append("KDJ金叉")
             trend_score += 1
        elif last_row['K'] < last_row['D'] and prev_row['K'] >= prev_row['D']:
             signals.append("KDJ死叉")
             trend_score -= 1
             
        # 5. Bollinger Bands Analysis
        if last_row['close'] > last_row['BB_Upper']:
            warnings.append("价格突破布林带上轨 - 极端行情，注意回调")
        elif last_row['close'] < last_row['BB_Lower']:
            warnings.append("价格跌破布林带下轨 - 极端行情，可能反弹")
            
        # 6. Combined Analysis with Model Prediction
        advice = ""
        if model_prediction_trend == 'up':
            if trend_score >= 2:
                advice = "✅ 【强力买入信号】模型预测上涨 + 技术面强力看涨 -> 极高置信度"
            elif trend_score > 0:
                advice = "✅ 【买入信号】模型预测上涨 + 技术面偏多 -> 较高置信度"
            elif trend_score < -1:
                advice = "⚠️ 【谨慎看涨】模型预测上涨，但技术面看跌 -> 可能处于反转初期或模型误判"
            else:
                advice = "⚖️ 【震荡看涨】模型预测上涨，技术面震荡"
                
            # Check for conflicts
            if last_row['RSI'] > 75:
                advice += "\n   ⚠️ 风险提示: 虽然看涨，但RSI超买，建议分批建仓或等待回调"
                
        elif model_prediction_trend == 'down':
            if trend_score <= -2:
                advice = "✅ 【强力卖出信号】模型预测下跌 + 技术面强力看跌 -> 极高置信度"
            elif trend_score < 0:
                advice = "✅ 【卖出信号】模型预测下跌 + 技术面偏空 -> 较高置信度"
            elif trend_score > 1:
                advice = "⚠️ 【谨慎看跌】模型预测下跌，但技术面看涨 -> 可能处于顶部反转或模型误判"
            else:
                advice = "⚖️ 【震荡看跌】模型预测下跌，技术面震荡"
                
            if last_row['RSI'] < 25:
                advice += "\n   ⚠️ 风险提示: 虽然看跌，但RSI超卖，谨防超跌反弹"

        return {
            "trend_score": trend_score,
            "signals": signals,
            "warnings": warnings,
            "advice": advice
        }
