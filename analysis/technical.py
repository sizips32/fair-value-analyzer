"""
기술적 분석 모듈
통합된 기술적 지표 계산 및 매매 신호 생성
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available. Some advanced indicators will be skipped.")
from ta import add_all_ta_features
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import TechnicalConfig

class TechnicalAnalyzer:
    """통합 기술적 분석기"""

    def __init__(self, config: TechnicalConfig):
        self.config = config
        self.indicators = {}
        self.signals = {}

    def calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 계산

        Args:
            data: OHLCV 데이터

        Returns:
            기술적 지표가 추가된 데이터프레임
        """
        enriched_data = data.copy()

        # 이동평균선
        enriched_data = self._add_moving_averages(enriched_data)

        # 모멘텀 지표
        enriched_data = self._add_momentum_indicators(enriched_data)

        # 추세 지표
        enriched_data = self._add_trend_indicators(enriched_data)

        # 변동성 지표
        enriched_data = self._add_volatility_indicators(enriched_data)

        # 거래량 지표
        enriched_data = self._add_volume_indicators(enriched_data)

        # 사용자 정의 지표
        enriched_data = self._add_custom_indicators(enriched_data)

        return enriched_data

    def _add_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """이동평균선 추가"""
        for period in self.config.ma_periods:
            # 단순 이동평균
            data[f'SMA_{period}'] = SMAIndicator(
                close=data['Close'], window=period
            ).sma_indicator()

            # 지수 이동평균
            data[f'EMA_{period}'] = EMAIndicator(
                close=data['Close'], window=period
            ).ema_indicator()

        return data

    def _add_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 지표 추가"""
        # RSI
        rsi = RSIIndicator(close=data['Close'], window=self.config.rsi_period)
        data['RSI'] = rsi.rsi()

        # Stochastic
        stoch = StochasticOscillator(
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )
        data['Stoch_K'] = stoch.stoch()
        data['Stoch_D'] = stoch.stoch_signal()

        # MFI (Money Flow Index)
        mfi = MFIIndicator(
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            volume=data['Volume']
        )
        data['MFI'] = mfi.money_flow_index()

        # Williams %R
        data['Williams_R'] = ((data['High'].rolling(window=14).max() - data['Close']) /
                             (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())) * -100

        return data

    def _add_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """추세 지표 추가"""
        # MACD
        macd = MACD(
            close=data['Close'],
            window_fast=self.config.macd_fast,
            window_slow=self.config.macd_slow,
            window_sign=self.config.macd_signal
        )
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Histogram'] = macd.macd_diff()

        # ADX (Average Directional Index) - only if TA-Lib is available
        if TALIB_AVAILABLE and len(data) >= 14:
            try:
                high_values = data['High'].values
                low_values = data['Low'].values
                close_values = data['Close'].values

                data['ADX'] = talib.ADX(high_values, low_values, close_values, timeperiod=14)
                data['Plus_DI'] = talib.PLUS_DI(high_values, low_values, close_values, timeperiod=14)
                data['Minus_DI'] = talib.MINUS_DI(high_values, low_values, close_values, timeperiod=14)

                # Parabolic SAR
                data['PSAR'] = talib.SAR(data['High'].values, data['Low'].values)
            except Exception as e:
                print(f"Warning: TA-Lib indicator calculation failed: {e}")

        return data

    def _add_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """변동성 지표 추가"""
        # Bollinger Bands
        bb = BollingerBands(
            close=data['Close'],
            window=self.config.bollinger_period,
            window_dev=self.config.bollinger_std
        )
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Middle'] = bb.bollinger_mavg()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Middle']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])

        # Average True Range
        atr = AverageTrueRange(
            high=data['High'],
            low=data['Low'],
            close=data['Close']
        )
        data['ATR'] = atr.average_true_range()

        # Keltner Channels
        data['KC_Upper'] = data[f'EMA_20'] + (2 * data['ATR'])
        data['KC_Lower'] = data[f'EMA_20'] - (2 * data['ATR'])

        return data

    def _add_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """거래량 지표 추가"""
        # On Balance Volume
        obv = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume'])
        data['OBV'] = obv.on_balance_volume()

        # Volume Rate of Change
        data['Volume_ROC'] = data['Volume'].pct_change(periods=10) * 100

        # Volume Moving Average
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']

        # VWAP (Volume Weighted Average Price)
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        data['VWAP'] = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()

        return data

    def _add_custom_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """사용자 정의 지표 추가"""
        # Ichimoku Cloud
        if len(data) >= 52:
            # Tenkan-sen (Conversion Line)
            high_9 = data['High'].rolling(window=9).max()
            low_9 = data['Low'].rolling(window=9).min()
            data['Tenkan_Sen'] = (high_9 + low_9) / 2

            # Kijun-sen (Base Line)
            high_26 = data['High'].rolling(window=26).max()
            low_26 = data['Low'].rolling(window=26).min()
            data['Kijun_Sen'] = (high_26 + low_26) / 2

            # Senkou Span A (Leading Span A)
            data['Senkou_Span_A'] = ((data['Tenkan_Sen'] + data['Kijun_Sen']) / 2).shift(26)

            # Senkou Span B (Leading Span B)
            high_52 = data['High'].rolling(window=52).max()
            low_52 = data['Low'].rolling(window=52).min()
            data['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)

            # Chikou Span (Lagging Span)
            data['Chikou_Span'] = data['Close'].shift(-26)

        # Supertrend
        data = self._calculate_supertrend(data)

        return data

    def _calculate_supertrend(self, data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Supertrend 지표 계산"""
        hl2 = (data['High'] + data['Low']) / 2
        atr = data['ATR']

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        # Initialize
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)

        for i in range(len(data)):
            if i == 0:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = 1
            else:
                if data['Close'].iloc[i] <= supertrend.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = -1

        data['Supertrend'] = supertrend
        data['Supertrend_Direction'] = direction

        return data

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        매매 신호 생성

        Returns:
            매매 신호가 추가된 데이터프레임
        """
        signals = pd.DataFrame(index=data.index)

        # RSI 신호
        signals['RSI_Signal'] = self._rsi_signals(data)

        # MACD 신호
        signals['MACD_Signal'] = self._macd_signals(data)

        # 볼린저 밴드 신호
        signals['BB_Signal'] = self._bollinger_signals(data)

        # 이동평균 신호
        signals['MA_Signal'] = self._moving_average_signals(data)

        # Stochastic 신호
        signals['Stoch_Signal'] = self._stochastic_signals(data)

        # 종합 신호
        signals['Composite_Signal'] = self._composite_signal(signals)

        # 신호 강도
        signals['Signal_Strength'] = self._calculate_signal_strength(signals)

        return signals

    def _rsi_signals(self, data: pd.DataFrame) -> pd.Series:
        """RSI 기반 신호"""
        signals = pd.Series(0, index=data.index)
        signals[data['RSI'] < 30] = 1   # 과매도 -> 매수
        signals[data['RSI'] > 70] = -1  # 과매수 -> 매도
        return signals

    def _macd_signals(self, data: pd.DataFrame) -> pd.Series:
        """MACD 기반 신호"""
        signals = pd.Series(0, index=data.index)

        # MACD 라인이 시그널 라인을 상향 돌파
        macd_cross_up = (
            (data['MACD'] > data['MACD_Signal']) &
            (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
        )

        # MACD 라인이 시그널 라인을 하향 돌파
        macd_cross_down = (
            (data['MACD'] < data['MACD_Signal']) &
            (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))
        )

        signals[macd_cross_up] = 1
        signals[macd_cross_down] = -1

        return signals

    def _bollinger_signals(self, data: pd.DataFrame) -> pd.Series:
        """볼린저 밴드 기반 신호"""
        signals = pd.Series(0, index=data.index)

        # 하단 밴드 접촉 -> 매수
        signals[data['Close'] <= data['BB_Lower']] = 1

        # 상단 밴드 접촉 -> 매도
        signals[data['Close'] >= data['BB_Upper']] = -1

        return signals

    def _moving_average_signals(self, data: pd.DataFrame) -> pd.Series:
        """이동평균 기반 신호"""
        signals = pd.Series(0, index=data.index)

        # 단기 이동평균이 장기 이동평균을 상향 돌파 (골든 크로스)
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            golden_cross = (
                (data['SMA_20'] > data['SMA_50']) &
                (data['SMA_20'].shift(1) <= data['SMA_50'].shift(1))
            )

            # 단기 이동평균이 장기 이동평균을 하향 돌파 (데드 크로스)
            dead_cross = (
                (data['SMA_20'] < data['SMA_50']) &
                (data['SMA_20'].shift(1) >= data['SMA_50'].shift(1))
            )

            signals[golden_cross] = 1
            signals[dead_cross] = -1

        return signals

    def _stochastic_signals(self, data: pd.DataFrame) -> pd.Series:
        """Stochastic 기반 신호"""
        signals = pd.Series(0, index=data.index)

        # %K가 %D를 상향 돌파하면서 과매도 구간
        stoch_buy = (
            (data['Stoch_K'] > data['Stoch_D']) &
            (data['Stoch_K'].shift(1) <= data['Stoch_D'].shift(1)) &
            (data['Stoch_K'] < 20)
        )

        # %K가 %D를 하향 돌파하면서 과매수 구간
        stoch_sell = (
            (data['Stoch_K'] < data['Stoch_D']) &
            (data['Stoch_K'].shift(1) >= data['Stoch_D'].shift(1)) &
            (data['Stoch_K'] > 80)
        )

        signals[stoch_buy] = 1
        signals[stoch_sell] = -1

        return signals

    def _composite_signal(self, signals: pd.DataFrame) -> pd.Series:
        """종합 신호 생성"""
        # 각 신호의 가중치
        weights = {
            'RSI_Signal': 0.2,
            'MACD_Signal': 0.3,
            'BB_Signal': 0.2,
            'MA_Signal': 0.2,
            'Stoch_Signal': 0.1
        }

        composite = pd.Series(0.0, index=signals.index)

        for signal_name, weight in weights.items():
            if signal_name in signals.columns:
                composite += signals[signal_name] * weight

        # 임계값 적용
        result = pd.Series(0, index=signals.index)
        result[composite > 0.3] = 1   # 매수
        result[composite < -0.3] = -1  # 매도

        return result

    def _calculate_signal_strength(self, signals: pd.DataFrame) -> pd.Series:
        """신호 강도 계산"""
        signal_columns = ['RSI_Signal', 'MACD_Signal', 'BB_Signal', 'MA_Signal', 'Stoch_Signal']
        existing_columns = [col for col in signal_columns if col in signals.columns]

        if not existing_columns:
            return pd.Series(0, index=signals.index)

        # 동일한 방향 신호의 개수
        buy_signals = (signals[existing_columns] == 1).sum(axis=1)
        sell_signals = (signals[existing_columns] == -1).sum(axis=1)

        # 강도 계산 (0-1 범위)
        strength = pd.Series(0.0, index=signals.index)
        strength += buy_signals / len(existing_columns)
        strength -= sell_signals / len(existing_columns)

        return strength

    def get_latest_analysis(self, data: pd.DataFrame) -> Dict:
        """최신 데이터 기반 분석 결과"""
        if data.empty:
            return {}

        latest = data.iloc[-1]
        signals = self.generate_signals(data)
        latest_signals = signals.iloc[-1] if not signals.empty else {}

        analysis = {
            'price_info': {
                'current_price': latest['Close'],
                'change': latest['Close'] - data['Close'].iloc[-2] if len(data) > 1 else 0,
                'change_percent': ((latest['Close'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100) if len(data) > 1 else 0
            },
            'technical_indicators': {
                'rsi': latest.get('RSI', 0),
                'macd': latest.get('MACD', 0),
                'macd_signal': latest.get('MACD_Signal', 0),
                'bb_position': latest.get('BB_Position', 0),
                'stoch_k': latest.get('Stoch_K', 0),
                'stoch_d': latest.get('Stoch_D', 0)
            },
            'signals': {
                'rsi_signal': latest_signals.get('RSI_Signal', 0),
                'macd_signal': latest_signals.get('MACD_Signal', 0),
                'bb_signal': latest_signals.get('BB_Signal', 0),
                'ma_signal': latest_signals.get('MA_Signal', 0),
                'composite_signal': latest_signals.get('Composite_Signal', 0),
                'signal_strength': latest_signals.get('Signal_Strength', 0)
            }
        }

        return analysis