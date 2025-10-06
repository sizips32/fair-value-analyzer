"""
데이터 수집 모듈
다양한 데이터 소스로부터 금융 데이터를 수집하고 전처리
"""

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from functools import lru_cache
from pathlib import Path
try:
    import streamlit as st
except ImportError:
    # Streamlit이 없는 환경에서도 작동하도록
    st = None

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MarketConfig

class DataCollector:
    """통합 데이터 수집기"""

    def __init__(self, market_config: MarketConfig):
        self.config = market_config
        self.logger = logging.getLogger(__name__)

    def fetch_historical_data(self,
                            start_date: datetime,
                            end_date: datetime,
                            interval: str = "1d") -> pd.DataFrame:
        """
        과거 데이터 수집

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            interval: 데이터 간격 (1d, 1h, 5m 등)

        Returns:
            OHLCV 데이터프레임
        """
        try:
            ticker = yf.Ticker(self.config.ticker)
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            if data.empty:
                raise ValueError(f"No data found for {self.config.ticker}")

            # 데이터 정리
            data = self._clean_data(data)

            self.logger.info(f"Successfully fetched {len(data)} rows for {self.config.ticker}")
            return data

        except Exception as e:
            self.logger.error(f"Error fetching data for {self.config.ticker}: {e}")
            raise

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정리 및 전처리"""
        # 열 이름 정리 - yfinance에서 반환되는 실제 열 이름 확인
        print(f"Original columns: {list(data.columns)}")

        # yfinance에서 반환되는 표준 열 이름에 맞춰 조정
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if len(data.columns) != len(expected_columns):
            # 실제 데이터의 열 이름을 그대로 사용하되, 필요한 열만 선택
            available_cols = [col for col in expected_columns if col in data.columns]
            if len(available_cols) >= 4:  # 최소한 OHLC는 있어야 함
                data = data[available_cols]
            else:
                raise ValueError(f"Required columns not found. Available: {list(data.columns)}")
        else:
            data.columns = expected_columns

        # 결측값 처리
        data = data.dropna()

        # 이상치 제거 (3 시그마 규칙)
        for col in ['Open', 'High', 'Low', 'Close']:
            mean = data[col].mean()
            std = data[col].std()
            data = data[abs(data[col] - mean) <= 3 * std]

        # 수익률 계산
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

        # 변동성 계산 (20일 롤링)
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)

        return data.dropna()

    def fetch_real_time_data(self) -> Dict:
        """실시간 데이터 수집"""
        try:
            ticker = yf.Ticker(self.config.ticker)
            info = ticker.info

            # 현재 가격 정보
            current_data = {
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'long_name': info.get('longName', self.config.ticker),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'timestamp': datetime.now()
            }

            # 변화량 계산
            if current_data['previous_close'] > 0:
                change = current_data['current_price'] - current_data['previous_close']
                change_percent = (change / current_data['previous_close']) * 100
                current_data['change'] = change
                current_data['change_percent'] = change_percent

            return current_data

        except Exception as e:
            self.logger.error(f"Error fetching real-time data: {e}")
            return {}

class DataProcessor:
    """데이터 전처리 및 검증"""

    @staticmethod
    def validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        데이터 품질 검증

        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        errors = []

        # 기본 검증
        if data.empty:
            errors.append("데이터가 비어있습니다")
            return False, errors

        # 필수 컬럼 확인
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"필수 컬럼 누락: {missing_columns}")

        # 가격 논리 검증
        for idx, row in data.iterrows():
            if row['High'] < row['Low']:
                errors.append(f"고가가 저가보다 낮음: {idx}")
            if row['Close'] > row['High'] or row['Close'] < row['Low']:
                errors.append(f"종가가 고저가 범위를 벗어남: {idx}")
            if row['Open'] > row['High'] or row['Open'] < row['Low']:
                errors.append(f"시가가 고저가 범위를 벗어남: {idx}")

        # 결측값 확인
        missing_data = data.isnull().sum()
        if missing_data.any():
            errors.append(f"결측값 발견: {missing_data.to_dict()}")

        # 중복 인덱스 확인
        if data.index.duplicated().any():
            errors.append("중복된 날짜 인덱스 발견")

        return len(errors) == 0, errors

    @staticmethod
    def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        데이터 정규화

        Args:
            data: 정규화할 데이터
            method: 정규화 방법 ('minmax', 'zscore', 'robust')
        """
        normalized_data = data.copy()

        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            normalized_data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(
                data[['Open', 'High', 'Low', 'Close']]
            )
        elif method == 'zscore':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(
                data[['Open', 'High', 'Low', 'Close']]
            )
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            normalized_data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(
                data[['Open', 'High', 'Low', 'Close']]
            )

        return normalized_data

    @staticmethod
    def create_features(data: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        featured_data = data.copy()

        # 가격 기반 피처
        featured_data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
        featured_data['OC_Ratio'] = (data['Open'] - data['Close']) / data['Close']

        # 변동성 피처
        featured_data['Price_Range'] = data['High'] - data['Low']
        featured_data['True_Range'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift()),
                abs(data['Low'] - data['Close'].shift())
            )
        )

        # 거래량 피처
        featured_data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        featured_data['Volume_Ratio'] = data['Volume'] / featured_data['Volume_MA']

        # 모멘텀 피처
        for period in [5, 10, 20]:
            featured_data[f'Return_{period}d'] = data['Close'].pct_change(period)
            featured_data[f'Volatility_{period}d'] = (
                data['Returns'].rolling(window=period).std() * np.sqrt(252)
            )

        return featured_data.dropna()

class MultiMarketCollector:
    """다중 시장 데이터 수집기"""

    def __init__(self, market_configs: Dict[str, MarketConfig]):
        self.collectors = {
            name: DataCollector(config)
            for name, config in market_configs.items()
        }

    async def fetch_multiple_markets(self,
                                   start_date: datetime,
                                   end_date: datetime) -> Dict[str, pd.DataFrame]:
        """여러 시장 데이터 동시 수집"""
        tasks = []

        for market_name, collector in self.collectors.items():
            task = asyncio.create_task(
                self._fetch_single_market(collector, start_date, end_date, market_name)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 정리
        market_data = {}
        for i, (market_name, result) in enumerate(zip(self.collectors.keys(), results)):
            if isinstance(result, Exception):
                logging.error(f"Error fetching {market_name}: {result}")
            else:
                market_data[market_name] = result

        return market_data

    async def _fetch_single_market(self,
                                 collector: DataCollector,
                                 start_date: datetime,
                                 end_date: datetime,
                                 market_name: str) -> pd.DataFrame:
        """단일 시장 데이터 수집 (비동기)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            collector.fetch_historical_data,
            start_date,
            end_date
        )

class DataCache:
    """데이터 캐싱 시스템"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_cache_key(self, ticker: str, start_date: datetime, end_date: datetime) -> str:
        """캐시 키 생성"""
        return f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    def save_to_cache(self, key: str, data: pd.DataFrame):
        """캐시에 데이터 저장"""
        cache_file = self.cache_dir / f"{key}.parquet"
        data.to_parquet(cache_file)

    def load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 로드"""
        cache_file = self.cache_dir / f"{key}.parquet"
        if cache_file.exists():
            # 캐시 파일이 24시간 이내인지 확인
            if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).hours < 24:
                return pd.read_parquet(cache_file)
        return None
