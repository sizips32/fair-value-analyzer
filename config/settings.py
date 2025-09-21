"""
설정 관리 모듈
분석 파라미터, 시장 설정, 모델 설정을 중앙 관리
"""

import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class MarketConfig:
    """시장별 설정"""
    ticker: str
    name: str
    timezone: str
    trading_hours: str
    currency: str
    data_source: str = "yfinance"

@dataclass
class TechnicalConfig:
    """기술적 분석 설정"""
    rsi_period: int = 14
    ma_periods: List[int] = None
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    def __post_init__(self):
        if self.ma_periods is None:
            self.ma_periods = [20, 50, 200]

@dataclass
class MonteCarloConfig:
    """몬테카를로 시뮬레이션 설정"""
    simulations: int = 10000
    forecast_days: int = 126  # 6개월
    confidence_levels: List[int] = None
    method: str = "geometric_brownian"

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [90, 95, 99]

@dataclass
class MLConfig:
    """머신러닝 모델 설정"""
    models: List[str] = None
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    sequence_length: int = 60
    train_test_split: float = 0.8

    def __post_init__(self):
        if self.models is None:
            self.models = ["LSTM", "GRU", "Prophet"]

class ConfigManager:
    """설정 관리자"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._load_config()

    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로"""
        return str(Path(__file__).parent / "config.yaml")

    def _load_config(self):
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            config_data = self._get_default_config()
            self._save_config(config_data)

        self.markets = {
            name: MarketConfig(**data)
            for name, data in config_data.get('markets', {}).items()
        }
        self.technical = TechnicalConfig(**config_data.get('technical', {}))
        self.monte_carlo = MonteCarloConfig(**config_data.get('monte_carlo', {}))
        self.ml = MLConfig(**config_data.get('ml', {}))

    def _get_default_config(self) -> Dict:
        """기본 설정값"""
        return {
            'markets': {
                'kospi': {
                    'ticker': '^KS11',
                    'name': 'KOSPI',
                    'timezone': 'Asia/Seoul',
                    'trading_hours': '09:00-15:30',
                    'currency': 'KRW'
                },
                'sp500': {
                    'ticker': '^GSPC',
                    'name': 'S&P 500',
                    'timezone': 'US/Eastern',
                    'trading_hours': '09:30-16:00',
                    'currency': 'USD'
                },
                'nasdaq': {
                    'ticker': '^IXIC',
                    'name': 'NASDAQ',
                    'timezone': 'US/Eastern',
                    'trading_hours': '09:30-16:00',
                    'currency': 'USD'
                }
            },
            'technical': {
                'rsi_period': 14,
                'ma_periods': [20, 50, 200],
                'bollinger_period': 20,
                'bollinger_std': 2.0,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            'monte_carlo': {
                'simulations': 10000,
                'forecast_days': 126,
                'confidence_levels': [90, 95, 99],
                'method': 'geometric_brownian'
            },
            'ml': {
                'models': ['LSTM', 'GRU', 'Prophet'],
                'lstm_epochs': 100,
                'lstm_batch_size': 32,
                'sequence_length': 60,
                'train_test_split': 0.8
            }
        }

    def _save_config(self, config_data: Dict):
        """설정 파일 저장"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

    def get_market_config(self, market_name: str) -> MarketConfig:
        """특정 시장 설정 조회"""
        if market_name not in self.markets:
            raise ValueError(f"Unknown market: {market_name}")
        return self.markets[market_name]

    def update_config(self, section: str, **kwargs):
        """설정 업데이트"""
        if section == 'technical':
            for key, value in kwargs.items():
                if hasattr(self.technical, key):
                    setattr(self.technical, key, value)
        elif section == 'monte_carlo':
            for key, value in kwargs.items():
                if hasattr(self.monte_carlo, key):
                    setattr(self.monte_carlo, key, value)
        elif section == 'ml':
            for key, value in kwargs.items():
                if hasattr(self.ml, key):
                    setattr(self.ml, key, value)

        # 설정 파일에 저장
        config_data = self._config_to_dict()
        self._save_config(config_data)

    def _config_to_dict(self) -> Dict:
        """설정을 딕셔너리로 변환"""
        return {
            'markets': {
                name: {
                    'ticker': config.ticker,
                    'name': config.name,
                    'timezone': config.timezone,
                    'trading_hours': config.trading_hours,
                    'currency': config.currency,
                    'data_source': config.data_source
                }
                for name, config in self.markets.items()
            },
            'technical': {
                'rsi_period': self.technical.rsi_period,
                'ma_periods': self.technical.ma_periods,
                'bollinger_period': self.technical.bollinger_period,
                'bollinger_std': self.technical.bollinger_std,
                'macd_fast': self.technical.macd_fast,
                'macd_slow': self.technical.macd_slow,
                'macd_signal': self.technical.macd_signal
            },
            'monte_carlo': {
                'simulations': self.monte_carlo.simulations,
                'forecast_days': self.monte_carlo.forecast_days,
                'confidence_levels': self.monte_carlo.confidence_levels,
                'method': self.monte_carlo.method
            },
            'ml': {
                'models': self.ml.models,
                'lstm_epochs': self.ml.lstm_epochs,
                'lstm_batch_size': self.ml.lstm_batch_size,
                'sequence_length': self.ml.sequence_length,
                'train_test_split': self.ml.train_test_split
            }
        }

# 전역 설정 인스턴스
config_manager = ConfigManager()