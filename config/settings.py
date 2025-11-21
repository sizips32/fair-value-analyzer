"""
설정 관리 모듈
분석 파라미터, 시장 설정, 모델 설정을 중앙 관리
환경 변수와 YAML 설정을 통합 관리
"""

import yaml
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path

# 환경 변수 지원
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

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
    """설정 관리자 (환경 변수 + YAML 통합)"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        # 환경 변수에서 기본값 로드
        self._load_env_config()
        self._load_config()
    
    def _load_env_config(self):
        """환경 변수에서 설정 로드"""
        # 환경 변수는 YAML 설정보다 우선순위가 낮음 (YAML이 기본값)
        # 하지만 특정 설정은 환경 변수로 오버라이드 가능
        self.env_config = {
            'cache_dir': os.getenv('CACHE_DIR', './cache'),
            'cache_max_size_mb': int(os.getenv('CACHE_MAX_SIZE_MB', '500')),
            'cache_ttl_hours': int(os.getenv('CACHE_TTL_HOURS', '24')),
            'log_level': os.getenv('LOG_LEVEL', 'INFO').upper(),
            'log_dir': os.getenv('LOG_DIR', './logs'),
            'log_to_file': os.getenv('LOG_TO_FILE', 'true').lower() == 'true',
            'log_to_console': os.getenv('LOG_TO_CONSOLE', 'true').lower() == 'true',
        }

    def _get_default_config_path(self) -> str:
        """기본 설정 파일 경로"""
        return str(Path(__file__).parent / "config.yaml")

    def _validate_config(self, config_data: Dict) -> bool:
        """
        설정값 검증
        
        Args:
            config_data: 설정 데이터 딕셔너리
            
        Returns:
            검증 성공 여부
            
        Raises:
            ValueError: 필수 섹션이나 필드가 없을 때
        """
        # 필수 섹션 확인
        required_sections = ['markets', 'technical', 'monte_carlo']
        for section in required_sections:
            if section not in config_data:
                raise ValueError(f"Missing required section: {section}")

        # 시장 설정 검증
        markets = config_data.get('markets', {})
        if not markets:
            raise ValueError("At least one market configuration is required")

        for market_name, market_data in markets.items():
            if not isinstance(market_data, dict):
                raise ValueError(f"Market {market_name} configuration must be a dictionary")
            
            required_fields = ['ticker', 'name', 'currency']
            for field in required_fields:
                if field not in market_data:
                    raise ValueError(f"Market '{market_name}' missing required field: {field}")
            
            # 티커 유효성 검사
            if not market_data.get('ticker') or not isinstance(market_data['ticker'], str):
                raise ValueError(f"Market '{market_name}' ticker must be a non-empty string")

        # 기술적 분석 설정 검증
        technical = config_data.get('technical', {})
        if technical:
            if 'rsi_period' in technical and (not isinstance(technical['rsi_period'], int) or technical['rsi_period'] < 1):
                raise ValueError("RSI period must be a positive integer")
            if 'ma_periods' in technical:
                if not isinstance(technical['ma_periods'], list) or not all(isinstance(p, int) and p > 0 for p in technical['ma_periods']):
                    raise ValueError("MA periods must be a list of positive integers")

        # 몬테카를로 설정 검증
        monte_carlo = config_data.get('monte_carlo', {})
        if monte_carlo:
            if 'simulations' in monte_carlo and (not isinstance(monte_carlo['simulations'], int) or monte_carlo['simulations'] < 1):
                raise ValueError("Monte Carlo simulations must be a positive integer")
            if 'forecast_days' in monte_carlo and (not isinstance(monte_carlo['forecast_days'], int) or monte_carlo['forecast_days'] < 1):
                raise ValueError("Forecast days must be a positive integer")

        return True

    def _load_config(self):
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except FileNotFoundError:
            config_data = self._get_default_config()
            self._save_config(config_data)

        # 설정 검증
        self._validate_config(config_data)

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

    def get_env_config(self, key: str, default: Any = None) -> Any:
        """환경 변수 설정 조회"""
        return self.env_config.get(key, default)
    
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
