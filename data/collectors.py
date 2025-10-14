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
                'financial_currency': info.get('financialCurrency', info.get('currency', 'USD')),
                'long_name': info.get('longName', self.config.ticker),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'dividend_yield': info.get('dividendYield', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'ex_dividend_date': info.get('exDividendDate', None),
                'payout_ratio': info.get('payoutRatio', 0),
                # 밸류에이션 비율 추가
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'ev_ebitda': info.get('enterpriseToEbitda', 0),
                # 재무구조 정보 추가
                'total_revenue': info.get('totalRevenue', 0),
                'gross_profit': info.get('grossProfits', 0),
                'operating_income': info.get('operatingIncome', 0),
                'net_income': info.get('netIncomeToCommon', info.get('netIncome', 0)),
                'ebitda': info.get('ebitda', 0),
                # 수익성 지표
                'gross_margin': info.get('grossMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'profit_margin': info.get('profitMargins', 0),
                'ebitda_margin': info.get('ebitdaMargins', 0),
                # 성장률 지표
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
                'revenue_per_share': info.get('revenuePerShare', 0),
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
    def validate_dividend_data(dividend_info: Dict) -> Tuple[bool, List[str]]:
        """
        배당 관련 데이터 검증
        
        Args:
            dividend_info: 배당 정보 딕셔너리
            
        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        errors = []
        
        dividend_yield = dividend_info.get('dividend_yield', 0)
        dividend_rate = dividend_info.get('dividend_rate', 0)
        payout_ratio = dividend_info.get('payout_ratio', 0)
        
        # 배당수익률 검증 (0-100% 범위)
        # yfinance의 dividendYield는 이미 백분율로 제공됨 (0-100 범위)
        if dividend_yield < 0 or dividend_yield > 100:
            errors.append(f"배당수익률이 비정상적입니다: {dividend_yield:.2f}%")
        
        # 배당률 검증 (양수여야 함)
        if dividend_rate < 0:
            errors.append(f"배당률이 음수입니다: {dividend_rate}")
            
        # 배당성향 검증 (0-100% 범위)
        if payout_ratio < 0 or payout_ratio > 1:
            errors.append(f"배당성향이 비정상적입니다: {payout_ratio*100:.2f}%")
            
        # 배당수익률과 배당률 일관성 검증
        if dividend_yield > 0 and dividend_rate > 0:
            # 대략적인 일관성 검사 (정확한 계산은 주가에 따라 달라짐)
            if dividend_yield > 20:  # 20% 이상은 비정상적으로 높음
                errors.append(f"배당수익률이 비정상적으로 높습니다: {dividend_yield:.2f}%")
        
        return len(errors) == 0, errors

    @staticmethod
    def format_large_number(value: float, currency: str) -> str:
        """
        큰 숫자 포맷팅 (억, 조 단위)
        
        Args:
            value: 포맷팅할 값
            currency: 통화 코드
            
        Returns:
            포맷팅된 문자열
        """
        if currency == 'KRW':
            # 한국: 억, 조 단위
            if abs(value) >= 1e12:  # 조
                return f"{value/1e12:.1f}조"
            elif abs(value) >= 1e8:  # 억
                return f"{value/1e8:.1f}억"
            else:
                return f"{value:,.0f}"
        else:
            # 기타 통화: B, T 단위
            if abs(value) >= 1e12:  # T
                return f"{value/1e12:.1f}T"
            elif abs(value) >= 1e9:  # B
                return f"{value/1e9:.1f}B"
            elif abs(value) >= 1e6:  # M
                return f"{value/1e6:.1f}M"
            else:
                return f"{value:,.0f}"

    @staticmethod
    def calculate_financial_metrics(financial_data: Dict, currency: str) -> Dict:
        """
        재무구조 지표 계산
        
        Args:
            financial_data: 재무 데이터 딕셔너리
            currency: 통화 코드
            
        Returns:
            계산된 재무 지표 딕셔너리
        """
        total_revenue = financial_data.get('total_revenue', 0)
        gross_profit = financial_data.get('gross_profit', 0)
        operating_income = financial_data.get('operating_income', 0)
        net_income = financial_data.get('net_income', 0)
        ebitda = financial_data.get('ebitda', 0)
        
        # 수익성 지표 계산
        gross_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        operating_margin = (operating_income / total_revenue * 100) if total_revenue > 0 else 0
        net_margin = (net_income / total_revenue * 100) if total_revenue > 0 else 0
        ebitda_margin = (ebitda / total_revenue * 100) if total_revenue > 0 else 0
        
        # 포맷팅된 값들
        formatted_revenue = DataProcessor.format_large_number(total_revenue, currency)
        formatted_gross_profit = DataProcessor.format_large_number(gross_profit, currency)
        formatted_operating_income = DataProcessor.format_large_number(operating_income, currency)
        formatted_net_income = DataProcessor.format_large_number(net_income, currency)
        formatted_ebitda = DataProcessor.format_large_number(ebitda, currency)
        
        return {
            'total_revenue': total_revenue,
            'gross_profit': gross_profit,
            'operating_income': operating_income,
            'net_income': net_income,
            'ebitda': ebitda,
            'formatted_revenue': formatted_revenue,
            'formatted_gross_profit': formatted_gross_profit,
            'formatted_operating_income': formatted_operating_income,
            'formatted_net_income': formatted_net_income,
            'formatted_ebitda': formatted_ebitda,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'net_margin': net_margin,
            'ebitda_margin': ebitda_margin,
            'revenue_growth': financial_data.get('revenue_growth', 0) * 100,
            'earnings_growth': financial_data.get('earnings_growth', 0) * 100,
            'earnings_quarterly_growth': financial_data.get('earnings_quarterly_growth', 0) * 100
        }

    @staticmethod
    def format_currency(value: float, currency: str) -> str:
        """
        통화별 포맷팅
        
        Args:
            value: 포맷팅할 값
            currency: 통화 코드
            
        Returns:
            포맷팅된 문자열
        """
        if currency == 'USD':
            return f"${value:.2f}"
        elif currency == 'KRW':
            return f"₩{value:,.0f}"
        elif currency == 'JPY':
            return f"¥{value:,.0f}"
        elif currency == 'EUR':
            return f"€{value:.2f}"
        elif currency == 'GBP':
            return f"£{value:.2f}"
        else:
            return f"{value:.2f} {currency}"

    @staticmethod
    def calculate_dividend_metrics(dividend_info: Dict, current_price: float) -> Dict:
        """
        배당 관련 지표 계산
        
        Args:
            dividend_info: 배당 정보 딕셔너리
            current_price: 현재 주가
            
        Returns:
            계산된 배당 지표 딕셔너리
        """
        dividend_yield = dividend_info.get('dividend_yield', 0)
        dividend_rate = dividend_info.get('dividend_rate', 0)
        payout_ratio = dividend_info.get('payout_ratio', 0)
        
        metrics = {
            'dividend_yield_percent': dividend_yield,  # 이미 백분율로 제공됨
            'dividend_rate': dividend_rate,
            'payout_ratio_percent': payout_ratio * 100,
            'annual_dividend_per_share': dividend_rate,
            'dividend_per_share_yield': (dividend_rate / current_price * 100) if current_price > 0 else 0
        }
        
        # 배당 안정성 지표
        if payout_ratio > 0:
            if payout_ratio < 0.3:
                metrics['dividend_sustainability'] = '높음'
            elif payout_ratio < 0.6:
                metrics['dividend_sustainability'] = '보통'
            else:
                metrics['dividend_sustainability'] = '낮음'
        else:
            metrics['dividend_sustainability'] = 'N/A'
            
        return metrics

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
