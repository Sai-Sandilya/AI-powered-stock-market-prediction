"""
Multi-Timeframe Signal Generation
Generate trading signals across different timeframes for various strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import talib
from dataclasses import dataclass

@dataclass
class SignalResult:
    """Container for signal results."""
    timeframe: str
    signal_type: str
    signal_value: float
    confidence: float
    timestamp: pd.Timestamp
    description: str

class MultiTimeframeSignals:
    def __init__(self):
        self.signals = {}
        
    def generate_scalping_signals(self, df: pd.DataFrame) -> Dict[str, SignalResult]:
        """
        Generate scalping signals (1-5 minute timeframe).
        """
        print("⚡ Generating scalping signals...")
        
        signals = {}
        
        # 1-minute RSI scalping
        rsi_1m = talib.RSI(df['Close'], timeperiod=14)
        signals['RSI_Scalping'] = SignalResult(
            timeframe='1m',
            signal_type='RSI',
            signal_value=rsi_1m.iloc[-1],
            confidence=0.7 if abs(rsi_1m.iloc[-1] - 50) > 20 else 0.3,
            timestamp=df.index[-1],
            description=f"RSI: {rsi_1m.iloc[-1]:.2f} - {'Oversold' if rsi_1m.iloc[-1] < 30 else 'Overbought' if rsi_1m.iloc[-1] > 70 else 'Neutral'}"
        )
        
        # MACD scalping
        macd, macd_signal, macd_hist = talib.MACD(df['Close'])
        signals['MACD_Scalping'] = SignalResult(
            timeframe='1m',
            signal_type='MACD',
            signal_value=macd_hist.iloc[-1],
            confidence=0.8 if abs(macd_hist.iloc[-1]) > 0.5 else 0.4,
            timestamp=df.index[-1],
            description=f"MACD Histogram: {macd_hist.iloc[-1]:.4f} - {'Bullish' if macd_hist.iloc[-1] > 0 else 'Bearish'}"
        )
        
        # Volume spike scalping
        volume_ma = df['Volume'].rolling(20).mean()
        volume_ratio = df['Volume'].iloc[-1] / volume_ma.iloc[-1]
        signals['Volume_Scalping'] = SignalResult(
            timeframe='1m',
            signal_type='Volume',
            signal_value=volume_ratio,
            confidence=0.9 if volume_ratio > 3 else 0.6 if volume_ratio > 2 else 0.3,
            timestamp=df.index[-1],
            description=f"Volume Ratio: {volume_ratio:.2f}x average - {'High Activity' if volume_ratio > 2 else 'Normal'}"
        )
        
        return signals
    
    def generate_day_trading_signals(self, df: pd.DataFrame) -> Dict[str, SignalResult]:
        """
        Generate day trading signals (5-15 minute timeframe).
        """
        print("📈 Generating day trading signals...")
        
        signals = {}
        
        # Bollinger Bands day trading
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'])
        bb_position = (df['Close'].iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        signals['Bollinger_Day'] = SignalResult(
            timeframe='5m',
            signal_type='Bollinger',
            signal_value=bb_position,
            confidence=0.8 if bb_position > 0.95 or bb_position < 0.05 else 0.5,
            timestamp=df.index[-1],
            description=f"BB Position: {bb_position:.2f} - {'Sell' if bb_position > 0.95 else 'Buy' if bb_position < 0.05 else 'Hold'}"
        )
        
        # Stochastic day trading
        stoch_k, stoch_d = talib.STOCH(df['High'], df['Low'], df['Close'])
        signals['Stochastic_Day'] = SignalResult(
            timeframe='5m',
            signal_type='Stochastic',
            signal_value=stoch_k.iloc[-1],
            confidence=0.7 if stoch_k.iloc[-1] < 20 or stoch_k.iloc[-1] > 80 else 0.4,
            timestamp=df.index[-1],
            description=f"Stoch K: {stoch_k.iloc[-1]:.2f} - {'Oversold' if stoch_k.iloc[-1] < 20 else 'Overbought' if stoch_k.iloc[-1] > 80 else 'Neutral'}"
        )
        
        # Support/Resistance day trading
        support = df['Low'].rolling(20).min().iloc[-1]
        resistance = df['High'].rolling(20).max().iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        support_distance = (current_price - support) / current_price
        resistance_distance = (resistance - current_price) / current_price
        
        signals['Support_Resistance_Day'] = SignalResult(
            timeframe='5m',
            signal_type='Support_Resistance',
            signal_value=support_distance,
            confidence=0.9 if support_distance < 0.01 else 0.7 if resistance_distance < 0.01 else 0.4,
            timestamp=df.index[-1],
            description=f"Price vs S/R - Support: {support_distance:.3f}, Resistance: {resistance_distance:.3f}"
        )
        
        return signals
    
    def generate_swing_trading_signals(self, df: pd.DataFrame) -> Dict[str, SignalResult]:
        """
        Generate swing trading signals (daily timeframe).
        """
        print("🔄 Generating swing trading signals...")
        
        signals = {}
        
        # Moving average crossover
        ma_20 = df['Close'].rolling(20).mean()
        ma_50 = df['Close'].rolling(50).mean()
        ma_cross = (ma_20.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1]
        
        signals['MA_Crossover_Swing'] = SignalResult(
            timeframe='1d',
            signal_type='MA_Crossover',
            signal_value=ma_cross,
            confidence=0.8 if abs(ma_cross) > 0.05 else 0.5,
            timestamp=df.index[-1],
            description=f"MA Cross: {ma_cross:.3f} - {'Bullish' if ma_cross > 0 else 'Bearish'}"
        )
        
        # RSI swing trading
        rsi_daily = talib.RSI(df['Close'], timeperiod=14)
        signals['RSI_Swing'] = SignalResult(
            timeframe='1d',
            signal_type='RSI',
            signal_value=rsi_daily.iloc[-1],
            confidence=0.9 if rsi_daily.iloc[-1] < 25 or rsi_daily.iloc[-1] > 75 else 0.6,
            timestamp=df.index[-1],
            description=f"Daily RSI: {rsi_daily.iloc[-1]:.2f} - {'Strong Buy' if rsi_daily.iloc[-1] < 25 else 'Strong Sell' if rsi_daily.iloc[-1] > 75 else 'Neutral'}"
        )
        
        # MACD swing trading
        macd_daily, macd_signal_daily, macd_hist_daily = talib.MACD(df['Close'])
        macd_trend = macd_daily.iloc[-1] - macd_daily.iloc[-5]  # 5-day trend
        
        signals['MACD_Swing'] = SignalResult(
            timeframe='1d',
            signal_type='MACD',
            signal_value=macd_trend,
            confidence=0.8 if abs(macd_trend) > 0.5 else 0.5,
            timestamp=df.index[-1],
            description=f"MACD Trend: {macd_trend:.4f} - {'Strengthening' if macd_trend > 0 else 'Weakening'}"
        )
        
        return signals
    
    def generate_position_trading_signals(self, df: pd.DataFrame) -> Dict[str, SignalResult]:
        """
        Generate position trading signals (weekly timeframe).
        """
        print("📊 Generating position trading signals...")
        
        signals = {}
        
        # Weekly trend analysis
        weekly_returns = df['Close'].pct_change(5)  # 5-day returns
        trend_strength = weekly_returns.rolling(20).mean().iloc[-1]
        trend_volatility = weekly_returns.rolling(20).std().iloc[-1]
        
        signals['Trend_Position'] = SignalResult(
            timeframe='1w',
            signal_type='Trend',
            signal_value=trend_strength,
            confidence=0.9 if abs(trend_strength) > 0.02 else 0.6,
            timestamp=df.index[-1],
            description=f"Weekly Trend: {trend_strength:.4f} - {'Strong Up' if trend_strength > 0.02 else 'Strong Down' if trend_strength < -0.02 else 'Sideways'}"
        )
        
        # Monthly momentum
        monthly_momentum = df['Close'].pct_change(20).iloc[-1]  # 20-day momentum
        signals['Momentum_Position'] = SignalResult(
            timeframe='1w',
            signal_type='Momentum',
            signal_value=monthly_momentum,
            confidence=0.8 if abs(monthly_momentum) > 0.05 else 0.5,
            timestamp=df.index[-1],
            description=f"Monthly Momentum: {monthly_momentum:.4f} - {'Strong' if abs(monthly_momentum) > 0.05 else 'Weak'}"
        )
        
        # Volatility regime
        volatility = df['Close'].pct_change().rolling(20).std().iloc[-1]
        vol_percentile = (df['Close'].pct_change().rolling(20).std() < volatility).mean()
        
        signals['Volatility_Position'] = SignalResult(
            timeframe='1w',
            signal_type='Volatility',
            signal_value=vol_percentile,
            confidence=0.7 if vol_percentile > 0.8 or vol_percentile < 0.2 else 0.4,
            timestamp=df.index[-1],
            description=f"Volatility Percentile: {vol_percentile:.2f} - {'High' if vol_percentile > 0.8 else 'Low' if vol_percentile < 0.2 else 'Normal'}"
        )
        
        return signals
    
    def generate_risk_management_signals(self, df: pd.DataFrame) -> Dict[str, SignalResult]:
        """
        Generate risk management signals.
        """
        print("🛡️ Generating risk management signals...")
        
        signals = {}
        
        # Maximum drawdown
        rolling_max = df['Close'].expanding().max()
        drawdown = (df['Close'] - rolling_max) / rolling_max
        current_drawdown = drawdown.iloc[-1]
        
        signals['Drawdown_Risk'] = SignalResult(
            timeframe='All',
            signal_type='Risk',
            signal_value=current_drawdown,
            confidence=0.9 if current_drawdown < -0.15 else 0.7 if current_drawdown < -0.10 else 0.4,
            timestamp=df.index[-1],
            description=f"Current Drawdown: {current_drawdown:.2%} - {'High Risk' if current_drawdown < -0.15 else 'Medium Risk' if current_drawdown < -0.10 else 'Low Risk'}"
        )
        
        # Volatility risk
        volatility = df['Close'].pct_change().rolling(20).std().iloc[-1]
        vol_risk = volatility * np.sqrt(252)  # Annualized volatility
        
        signals['Volatility_Risk'] = SignalResult(
            timeframe='All',
            signal_type='Risk',
            signal_value=vol_risk,
            confidence=0.8 if vol_risk > 0.4 else 0.6 if vol_risk > 0.25 else 0.4,
            timestamp=df.index[-1],
            description=f"Annualized Volatility: {vol_risk:.2%} - {'High' if vol_risk > 0.4 else 'Medium' if vol_risk > 0.25 else 'Low'}"
        )
        
        # Correlation risk (if multiple assets)
        signals['Correlation_Risk'] = SignalResult(
            timeframe='All',
            signal_type='Risk',
            signal_value=0.5,  # Placeholder
            confidence=0.6,
            timestamp=df.index[-1],
            description="Portfolio correlation risk assessment"
        )
        
        return signals
    
    def generate_all_signals(self, df: pd.DataFrame) -> Dict[str, Dict[str, SignalResult]]:
        """
        Generate all signals across all timeframes.
        """
        print("🎯 Generating comprehensive multi-timeframe signals...")
        
        all_signals = {
            'scalping': self.generate_scalping_signals(df),
            'day_trading': self.generate_day_trading_signals(df),
            'swing_trading': self.generate_swing_trading_signals(df),
            'position_trading': self.generate_position_trading_signals(df),
            'risk_management': self.generate_risk_management_signals(df)
        }
        
        self.signals = all_signals
        return all_signals
    
    def get_signal_summary(self) -> pd.DataFrame:
        """
        Get a summary of all signals.
        """
        if not self.signals:
            return pd.DataFrame()
        
        summary_data = []
        for strategy, signals in self.signals.items():
            for signal_name, signal in signals.items():
                summary_data.append({
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Signal': signal_name.replace('_', ' ').title(),
                    'Timeframe': signal.timeframe,
                    'Value': signal.signal_value,
                    'Confidence': signal.confidence,
                    'Description': signal.description,
                    'Timestamp': signal.timestamp
                })
        
        return pd.DataFrame(summary_data)
    
    def get_high_confidence_signals(self, min_confidence: float = 0.7) -> List[SignalResult]:
        """
        Get signals with high confidence.
        """
        high_confidence = []
        for strategy, signals in self.signals.items():
            for signal_name, signal in signals.items():
                if signal.confidence >= min_confidence:
                    high_confidence.append(signal)
        
        return sorted(high_confidence, key=lambda x: x.confidence, reverse=True)
    
    def get_trading_recommendations(self) -> Dict[str, str]:
        """
        Get trading recommendations based on signal consensus.
        """
        recommendations = {}
        
        # Scalping recommendation
        scalping_signals = self.signals.get('scalping', {})
        if scalping_signals:
            avg_confidence = np.mean([s.confidence for s in scalping_signals.values()])
            recommendations['Scalping'] = 'Active' if avg_confidence > 0.6 else 'Wait'
        
        # Day trading recommendation
        day_signals = self.signals.get('day_trading', {})
        if day_signals:
            avg_confidence = np.mean([s.confidence for s in day_signals.values()])
            recommendations['Day Trading'] = 'Active' if avg_confidence > 0.6 else 'Wait'
        
        # Swing trading recommendation
        swing_signals = self.signals.get('swing_trading', {})
        if swing_signals:
            avg_confidence = np.mean([s.confidence for s in swing_signals.values()])
            recommendations['Swing Trading'] = 'Active' if avg_confidence > 0.6 else 'Wait'
        
        # Position trading recommendation
        position_signals = self.signals.get('position_trading', {})
        if position_signals:
            avg_confidence = np.mean([s.confidence for s in position_signals.values()])
            recommendations['Position Trading'] = 'Active' if avg_confidence > 0.6 else 'Wait'
        
        # Risk assessment
        risk_signals = self.signals.get('risk_management', {})
        if risk_signals:
            risk_level = 'High' if any(s.signal_value < -0.15 for s in risk_signals.values()) else 'Medium'
            recommendations['Risk Level'] = risk_level
        
        return recommendations

def generate_multi_timeframe_signals(df: pd.DataFrame) -> MultiTimeframeSignals:
    """
    Convenience function to generate all multi-timeframe signals.
    """
    signal_generator = MultiTimeframeSignals()
    signal_generator.generate_all_signals(df)
    return signal_generator 